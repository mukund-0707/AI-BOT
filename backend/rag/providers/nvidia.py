from openai import OpenAI

from django.conf import settings


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

        return embedding

    def generate_embeddings_batch(self, texts: list[str]):

        response = self.client.embeddings.create(
            model=settings.NVIDIA_EMBEDDING_MODEL,
            input=texts,
        )

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
            max_tokens=2048,
        )
        print("NVIDIA response:", response.choices[0].message.content)
        return response.choices[0].message.content
