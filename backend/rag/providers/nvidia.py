import re

from openai import OpenAI

from django.conf import settings

# Reasoning models normally return their trace in a separate field, but they
# occasionally inline it in the answer. Strip it either way.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


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
