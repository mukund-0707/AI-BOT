import logging
import re

import httpx
from openai import OpenAI

from django.conf import settings

logger = logging.getLogger(__name__)

# Reasoning models normally return their trace in a separate field, but they
# occasionally inline it in the answer. Strip it either way.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"


def strip_reasoning(text):

    if not text:
        return ""

    return THINK_BLOCK.sub("", text).strip()


def _unfinished_tag(text, tag):
    """Length of the trailing run that could still grow into `tag`."""

    for size in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:size]):
            return size

    return 0


class ReasoningFilter:
    """Removes <think>...</think> from a stream of deltas.

    `strip_reasoning` can regex the finished answer, but a stream never holds
    the whole tag at once: the model emits "<th" and then "ink>", so searching
    each delta on its own never matches and the entire trace reaches the user's
    screen - and, because the caller assembles the answer from what it yields,
    the session history too.

    Text is therefore emitted only once it cannot be the start of a tag.
    """

    def __init__(self):
        self.buffer = ""
        self.inside = False

    def feed(self, delta):
        """The visible part of `delta`, which may be empty."""

        self.buffer += delta
        visible = []

        while True:
            if self.inside:
                end = self.buffer.find(THINK_CLOSE)

                if end == -1:
                    # Discard the trace, but keep anything that could still
                    # turn out to be the closing tag.
                    keep = _unfinished_tag(self.buffer, THINK_CLOSE)
                    self.buffer = self.buffer[len(self.buffer) - keep :]
                    break

                self.buffer = self.buffer[end + len(THINK_CLOSE) :]
                self.inside = False
                continue

            start = self.buffer.find(THINK_OPEN)

            if start == -1:
                keep = _unfinished_tag(self.buffer, THINK_OPEN)
                cut = len(self.buffer) - keep

                if cut:
                    visible.append(self.buffer[:cut])

                self.buffer = self.buffer[cut:]
                break

            if start:
                visible.append(self.buffer[:start])

            self.buffer = self.buffer[start + len(THINK_OPEN) :]
            self.inside = True

        return "".join(visible)

    def flush(self):
        """Whatever is left when the stream ends.

        An unclosed tag means the model never came back out of its trace, so
        what is held is reasoning rather than answer, and is dropped.
        """

        tail = "" if self.inside else self.buffer

        self.buffer = ""

        return tail


class NVIDIAProvider:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.NVIDIA_API_KEY,
            base_url=settings.NVIDIA_BASE_URL,
        )

    def simple_chat(self, prompt: str):

        response = self.client.chat.completions.create(
            model=settings.NVIDIA_CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=300,
        )

        return response.choices[0].message.content

    def generate_embedding(self, text: str):

        response = self.client.embeddings.create(
            model=settings.NVIDIA_EMBEDDING_MODEL,
            input=text,
            extra_body={"input_type": "query"},
        )

        embedding = response.data[0].embedding
        # print("RESPONSE EMMBD: \n", response)

        return embedding

    def generate_embeddings_batch(self, texts: list[str]):

        response = self.client.embeddings.create(
            model=settings.NVIDIA_EMBEDDING_MODEL,
            input=texts,
            extra_body={"input_type": "passage"},
        )
        # print("RESPONSE BATCH: \n", response)

        return [item.embedding for item in response.data]

    def rerank(self, question: str, chunks: list[dict], top_n: int, floor: float):
        """Score every candidate against the question with a cross-encoder.

        Unlike cosine similarity, these logits are comparable across queries,
        so `floor` can be a constant: relevant passages land around -1 to -6,
        while a question the corpus cannot answer tops out near -12.
        """

        if not chunks:
            return []

        response = httpx.post(
            RERANK_URL,
            headers={"Authorization": f"Bearer {settings.NVIDIA_API_KEY}"},
            json={
                "model": settings.NVIDIA_RERANK_MODEL,
                "query": {"text": question},
                "passages": [{"text": chunk["text"]} for chunk in chunks],
            },
            timeout=30,
        )
        response.raise_for_status()

        rankings = sorted(
            response.json()["rankings"],
            key=lambda ranking: -ranking["logit"],
        )

        rerankings_chunk = [
            {**chunks[ranking["index"]], "score": ranking["logit"]}
            for ranking in rankings[:top_n]
            if ranking["logit"] >= floor
        ]

        return rerankings_chunk

    def classify(self, system_prompt: str, user_prompt: str, max_tokens: int):
        """A routing call, not an answer: greedy, tiny, and quick to give up.

        The caller treats any failure as "this is a question", so a short timeout
        costs a slightly worse route, never a failed request.
        """

        response = self.client.chat.completions.create(
            model=settings.NVIDIA_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            timeout=settings.RAG_INTENT_TIMEOUT,
        )

        return strip_reasoning(response.choices[0].message.content)

    def stream_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict] | None = None,
    ):
        """Same as generate_answer but yields tokens as they arrive.

        The caller is responsible for assembling the full text if it needs to
        store the answer after streaming completes.
        """

        messages = [{"role": "system", "content": system_prompt}]

        for entry in history or []:
            messages.append({"role": entry["role"], "content": entry["text"]})

        messages.append({"role": "user", "content": user_prompt})

        # A tag is routinely split across deltas, so the filter holds text back
        # rather than deciding one delta at a time.
        reasoning = ReasoningFilter()

        with self.client.chat.completions.create(
            model=settings.NVIDIA_CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            max_tokens=1200,
            stream=True,
        ) as stream:
            for chunk in stream:
                # Guard: some chunks arrive with an empty choices list
                # (e.g. the final [DONE] sentinel from the API).
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta.content or ""

                if not delta:
                    continue

                visible = reasoning.feed(delta)

                if visible:
                    yield visible

        tail = reasoning.flush()

        if tail:
            yield tail

    def generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict] | None = None,
    ):
        """`history` is sent as real chat turns, not pasted into the prompt.

        The model treats prior turns as things that were said; text inside the
        user message reads as material to answer from.
        """

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        for entry in history or []:
            messages.append(
                {
                    "role": entry["role"],
                    "content": entry["text"],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        response = self.client.chat.completions.create(
            model=settings.NVIDIA_CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            max_tokens=1200,
        )
        choice = response.choices[0]
        logger.debug("Chat finish_reason=%s", choice.finish_reason)
        answer = strip_reasoning(choice.message.content)

        if choice.finish_reason == "length" and not answer:
            return (
                "The answer was cut off before it could be written. "
                "Please ask a narrower question."
            )

        return answer
