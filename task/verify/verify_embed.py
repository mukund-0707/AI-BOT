"""Probe the real NVIDIA embedding endpoint to ground the analysis in actual behaviour."""

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
print(f"model = {MODEL}")
print(f"chat  = {settings.NVIDIA_CHAT_MODEL}")


def embed(text, input_type=None):
    kw = {}
    if input_type:
        kw["extra_body"] = {"input_type": input_type, "truncate": "NONE"}
    r = client.embeddings.create(model=MODEL, input=[text], **kw)
    return r.data[0].embedding


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * y for x, y in zip(b, b)) ** 0.5
    return dot / (na * nb)


print("\n--- 1) exactly what the code does today (NO input_type) ---")
try:
    v = embed("What is the ReMo runbook escalation contact?")
    print(f"  OK, dim={len(v)}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\n--- 2) with input_type=query ---")
try:
    vq = embed("What is the ReMo runbook escalation contact?", "query")
    print(f"  OK, dim={len(vq)}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    vq = None

print("\n--- 3) with input_type=passage ---")
try:
    vp = embed(
        "The escalation contact is Verena Wanner, verena.wanner@syngenta.com", "passage"
    )
    print(f"  OK, dim={len(vp)}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    vp = None

print("\n--- 4) does input_type actually change the vector? ---")
try:
    a = embed("escalation contact", "query")
    b = embed("escalation contact", "passage")
    same = a == b
    print(f"  query-vec == passage-vec for identical text? {same}")
    if not same:
        print(f"  cosine(query_enc, passage_enc) = {cos(a, b):.4f}")
        print("  -> asymmetric encoding IS active; code ignoring it is a real bug")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\n--- 5) REAL score check: the exact question the user cares about ---")
try:
    q = "Who is the contact person for ReMo?"
    good = "Name (Title/Position): Verena Wanner. Email: verena.wanner@syngenta.com. Telephone: +41-613236961"
    generic = "This runbook describes the registration monitoring process and its operational steps."
    vq2 = embed(q)
    vgood = embed(good)
    vgen = embed(generic)
    print(f"  cos(question, table-contact-chunk) = {cos(vq2, vgood):.4f}")
    print(f"  cos(question, generic-prose-chunk) = {cos(vq2, vgen):.4f}")
    print(f"  RAG_MIN_SCORE gate in settings     = {settings.RAG_MIN_SCORE}")
    print("  NOTE: Qdrant COSINE scores are compared against this same threshold.")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\n--- 6) summarize-style query vs a real chunk (user's reported failure) ---")
try:
    vs = embed("give me a brief summary")
    vchunk = embed(
        "The escalation contact is Verena Wanner and the runbook covers registration monitoring."
    )
    print(f"  cos('give me a brief summary', chunk) = {cos(vs, vchunk):.4f}")
    print(
        f"  threshold = {settings.RAG_MIN_SCORE} -> "
        f"{'PASSES' if cos(vs, vchunk) >= settings.RAG_MIN_SCORE else 'BLOCKED (returns not-found)'}"
    )
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
