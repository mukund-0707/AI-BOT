"""Reproduce suspected bugs in rag.chunking.create_chunks without touching the project."""

import sys

sys.path.insert(0, r"D:\mukund\AI bot\AI bot\backend")

from rag.chunking import create_chunks

print("=" * 60)
print("TEST 1: normal prose -- baseline sanity")
text = " ".join(f"word{i}" for i in range(1000))
chunks = create_chunks([{"page_number": 1, "text": text}])
print(f"  input len={len(text)} -> {len(chunks)} chunks")
print(f"  chunk lens: {[len(c['text']) for c in chunks][:8]}")

print("=" * 60)
print("TEST 2: negative-start bug -- space early, then long unbroken run")
# space at index 5, then 2000 chars with no space at all
text2 = "abcde " + ("x" * 2000)
chunks2 = create_chunks([{"page_number": 1, "text": text2}])
print(f"  input len={len(text2)} -> {len(chunks2)} chunks")
for c in chunks2[:5]:
    t = c["text"]
    print(f"   idx={c['chunk_index']} len={len(t)} head={t[:20]!r} tail={t[-12:]!r}")
total_chars = sum(len(c["text"]) for c in chunks2)
print(f"  TOTAL chars across chunks = {total_chars} (input was {len(text2)})")

print("=" * 60)
print("TEST 3: infinite-loop probe -- space exactly at overlap boundary")
# space at index 150 only, nothing after -> start = end - overlap = 0 forever
text3 = ("y" * 150) + " " + ("z" * 3000)
print(f"  input len={len(text3)}; space at index {text3.find(' ')}")
import threading

result = {}


def run():
    try:
        result["chunks"] = create_chunks([{"page_number": 1, "text": text3}])
    except Exception as e:  # noqa: BLE001
        result["err"] = repr(e)


t = threading.Thread(target=run, daemon=True)
t.start()
t.join(timeout=5)
if t.is_alive():
    print(
        "  >>> HANG CONFIRMED: create_chunks did not return within 5s (infinite loop)"
    )
else:
    print(
        f"  returned: {len(result.get('chunks', []))} chunks, err={result.get('err')}"
    )
