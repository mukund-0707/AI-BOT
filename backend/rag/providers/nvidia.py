import re

import httpx
from openai import OpenAI

from django.conf import settings

# Reasoning models normally return their trace in a separate field, but they
# occasionally inline it in the answer. Strip it either way.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"


def strip_reasoning(text):

    if not text:
        return ""

    return THINK_BLOCK.sub("", text).strip()


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
        )

        embedding = response.data[0].embedding
        # print("RESPONSE EMMBD: \n", response)

        return embedding

    def generate_embeddings_batch(self, texts: list[str]):

        response = self.client.embeddings.create(
            model=settings.NVIDIA_EMBEDDING_MODEL,
            input=texts,
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

        return [
            {**chunks[ranking["index"]], "score": ranking["logit"]}
            for ranking in rankings[:top_n]
            if ranking["logit"] >= floor
        ]

    def generate_answer(self, system_prompt: str, user_prompt: str):

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
            temperature=0.2,
            top_p=0.9,
            max_tokens=1200,
        )
        print("NVIDIA response:", response.choices[0].message.content)
        choice = response.choices[0]
        answer = strip_reasoning(choice.message.content)

        if choice.finish_reason == "length" and not answer:
            return (
                "The answer was cut off before it could be written. "
                "Please ask a narrower question."
            )

        return answer
