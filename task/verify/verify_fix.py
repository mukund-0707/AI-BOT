"""Prove whether input_type + instruction prefix actually fixes retrieval separation."""

import os
import sys

sys.path.insert(0, r"D:\mukund\AI bot\AI bot\backend")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
os.chdir(r"D:\mukund\AI bot\AI bot\backend")

import django

django.setup()
from django.conf import settings
from openai import OpenAI

client = OpenAI(api_key=settings.NVIDIA_API_KEY, base_url=settings.NVIDIA_BASE_URL)
MODEL = settings.NVIDIA_EMBEDDING_MODEL

INSTRUCT = (
    "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
)


def embed(text, input_type=None):
    kw = {}
    if input_type:
        kw["extra_body"] = {"input_type": input_type, "truncate": "NONE"}
    return client.embeddings.create(model=MODEL, input=[text], **kw).data[0].embedding


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)


Q = "Who is the contact person for ReMo?"
GOLD = "Name (Title/Position): Verena Wanner. Email: verena.wanner@syngenta.com. Telephone: +41-613236961"
DISTRACTOR = "This runbook describes the registration monitoring process and its operational steps."
UNRELATED = "The company cafeteria serves lunch between 12:00 and 14:00 on weekdays."

configs = [
    ("A. CURRENT CODE (no input_type, no instruction)", None, None, False),
    ("B. + input_type only (query/passage)", "query", "passage", False),
    ("C. + input_type AND instruction prefix", "query", "passage", True),
]

print(f"{'config':<48} {'gold':>8} {'distract':>9} {'unrel':>8} {'margin':>8}")
print("-" * 86)

rows = []
for name, qt, pt, instruct in configs:
    q_text = (INSTRUCT + Q) if instruct else Q
    vq = embed(q_text, qt)
    g = cos(vq, embed(GOLD, pt))
    d = cos(vq, embed(DISTRACTOR, pt))
    u = cos(vq, embed(UNRELATED, pt))
    margin = g - max(d, u)
    rows.append((name, g, d, u, margin))
    print(f"{name:<48} {g:>8.4f} {d:>9.4f} {u:>8.4f} {margin:>+8.4f}")

print("-" * 86)
print("gold   = the table chunk that actually answers the question")
print("margin = gold - best_wrong_chunk. NEGATIVE margin => wrong chunk ranks FIRST.")
print(f"\nRAG_MIN_SCORE currently = {settings.RAG_MIN_SCORE}")
print(
    "Gate in rag/services.py:  if results[0]['score'] < RAG_MIN_SCORE -> 'couldn't find information'"
)
for name, g, d, u, m in rows:
    verdict = "PASSES gate" if g >= settings.RAG_MIN_SCORE else "BLOCKED by gate"
    print(f"  {name:<48} gold={g:.4f} -> {verdict}")
