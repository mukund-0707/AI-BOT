# Mukii RAG Chatbot — Deep-Dive Analysis

**Date:** 2026-07-17
**Scope:** Full `backend/` codebase — ingestion → chunking → embeddings → vector search → context construction → answer generation.
**Method:** Every conclusion below is backed by either a code reference (`file:line`) or a measurement actually executed against this codebase and the live NVIDIA API. Reproduction scripts are described in §9.

> **Headline:** There is no single "prompt problem". There are **four independent, individually-fatal defects** stacked on top of each other. Two of them (the score threshold and the missing `input_type`) were measured directly and each one alone is sufficient to produce every symptom you reported. Fixing the prompt would have changed nothing.

---

## 1. Current System Architecture

### 1.1 Modules

| App | Role | Key files |
|---|---|---|
| `documents` | Upload, extract, clean, chunk, embed, index | `services.py`, `extractors/*`, `text_cleaner.py` |
| `rag` | Retrieval + generation | `services.py`, `chunking.py`, `prompts.py`, `providers/nvidia.py`, `providers/qdrant.py` |
| `chat` | HTTP API + UI | `views.py`, `serializers.py`, `urls.py` |
| `accounts` | Session auth | `authenticate.py`, `views.py` |

### 1.2 Ingestion flow (request-to-index)

```
POST /api/documents/upload/          documents/views.py:18   (IsAuthenticated)
  └─ DocumentUploadSerializer         validate ext(.pdf/.docx/.txt), size<=10MB, size>0
  └─ DocumentService.upload()         services.py:19   Document row, status=UPLOADED
  └─ DocumentService.process_document()  services.py:45   *** SYNCHRONOUS, in-request ***
       ├─ status = PROCESSING
       ├─ extract_document()          services.py:32  -> extract_pdf / extract_docx / extract_text
       ├─ clean_text()                text_cleaner.py:4
       ├─ create_chunks()             rag/chunking.py:1   chunk_size=1000 chars, overlap=150
       ├─ generate_embeddings_batch() providers/nvidia.py:41  batches of 50
       ├─ ensure_collection(4096)     providers/qdrant.py:29  COSINE
       ├─ upsert_chunks()             providers/qdrant.py:47   payload: document_id, file_name,
       │                                                        page_number, chunk_index, text
       └─ status = READY
```

### 1.3 Query flow (request-to-response)

```
POST /api/chat/ask/                  chat/views.py:22   (AllowAny  <- note the asymmetry)
  └─ AskQuestionSerializer           question non-empty
  └─ answer_question(question)       rag/services.py:83
       ├─ nvidia.generate_embedding(question)   providers/nvidia.py:30   *** no input_type ***
       ├─ qdrant.search_chunks(emb)             providers/qdrant.py:74   limit=10, NO filter
       ├─ if not results  -> "couldn't find information"
       ├─ if results[0]["score"] < 0.70 -> "couldn't find information"   *** services.py:99-105 ***
       ├─ build_prompt(question, chunks)        prompts.py:31   ALL 10 chunks, unfiltered
       ├─ nvidia.generate_answer(SYSTEM, prompt)  temperature=0.2, max_tokens=2048
       └─ "Smart Source Hiding"                 services.py:117-138  string-matching on the answer
```

### 1.4 What is absent entirely

No reranking. No query rewriting/expansion. No hybrid (keyword/BM25) search. No metadata filtering at query time. No per-chunk score filtering. No context deduplication. No token budgeting. No conversation history (`chat/models.py` is empty — the model file contains only the default comment). No citations in the answer text. No evaluation harness. No logging (`print()` only). No tests (all four `tests.py` are 3 lines / empty).

---

## 2. Major Issues Found

| # | Severity | Issue | Location | Verified? |
|---|---|---|---|---|
| C1 | **CRITICAL** | `RAG_MIN_SCORE = 0.70` is ~3× higher than this model's real relevant-chunk score band (0.22–0.30) → gate rejects almost everything | `settings.py:170`, `rag/services.py:99-105` | **Measured** |
| C2 | **CRITICAL** | `input_type` (query/passage) never sent → asymmetric model used symmetrically → **wrong chunk ranks first** (margin −0.0350) | `providers/nvidia.py:30-48` | **Measured** |
| C3 | **CRITICAL** | DOCX extractor reads `document.paragraphs` only → **all table content silently dropped** (42.5% / 19.7% of your real files) | `extractors/docx.py:9-19` | **Measured** |
| C4 | **CRITICAL** | `create_chunks` can produce a **negative `start`** → silently skips text; and can **infinite-loop** → request hangs forever | `rag/chunking.py:30` | **Reproduced** |
| H1 | High | Only `results[0]` is score-gated; the other 9 chunks enter the prompt unfiltered → noise → generic answers | `rag/services.py:99-110` | Code |
| H2 | High | No dedup: same file uploaded 3× (`Connect_Applications_to_Gaia_1*.pdf`) → identical chunks fill top-10 | `media/documents/`, no dedup path | Observed |
| H3 | High | No reranking → embedding order is final order | absent | Code |
| H4 | High | Synchronous ingestion inside the HTTP request → timeouts on large files | `documents/views.py:29` | Code |
| H5 | High | No conversation history → every turn is stateless; follow-ups ("aur batao") lose referent | `chat/models.py` empty | Code |
| M1 | Medium | Prompt puts Question **before** Context; no instruction reinforcement after context | `prompts.py:31-46` | Code |
| M2 | Medium | "Smart Source Hiding" matches English substrings in the answer — brittle, silently drops sources | `rag/services.py:117-123` | Code |
| M3 | Medium | `page_number=None` for DOCX/TXT → every citation says "N/A"; `total_pages` is always 1 | `extractors/docx.py:17`, `text.py:7` | Code |
| M4 | Medium | Fixed `limit=10`, no dynamic top-k, no token budget for context | `providers/qdrant.py:76` | Code |
| M5 | Medium | `AskQuestionView` is `AllowAny` while upload requires auth → the knowledge base is world-readable | `chat/views.py:23` | Code |
| M6 | Medium | No payload index on `document_id`; `delete_chunks` filter scans | `providers/qdrant.py:107` | Code |
| L1 | Low | Dead/wrong import `from docx import document` | `rag/services.py:2` | Code |
| L2 | Low | Providers instantiated at import time (module-level singletons) | `rag/services.py:14-15` | Code |
| L3 | Low | `print()` instead of `logging`; no tests anywhere | throughout | Code |
| L4 | Low | `extract_text` hard-codes `encoding="utf-8"` → `UnicodeDecodeError` on cp1252 files | `extractors/text.py:2` | Code |

---

## 3. Root-Cause Analysis

### 3.1 The measurement that explains everything

I embedded a real question against real content from your own `RegistrationMonitoring-ReMo_Runbook_Full.docx` using **your** API key and **your** configured model (`nvidia/nv-embed-v1`, 4096-dim), and computed the exact cosine similarity Qdrant would compute:

```
config                                               gold  distract    unrel   margin
--------------------------------------------------------------------------------------
A. CURRENT CODE (no input_type, no instruction)    0.2608    0.2958   0.1340  -0.0350
B. + input_type only (query/passage)               0.2176    0.0563   0.0299  +0.1613
C. + input_type AND instruction prefix             0.2338    0.0721   0.0326  +0.1617
```

- `gold` = the chunk that actually answers *"Who is the contact person for ReMo?"*
- `distract` = generic prose from the same document
- `margin` = `gold − best_wrong_chunk`. **Negative margin means the wrong chunk ranks first.**

Two independent, fatal facts fall out of this table:

**(a) The threshold is impossible to pass.** Your relevant chunks score **0.22–0.30**. Your gate is **0.70** (`settings.py:170`). The line

```python
if results[0]["score"] < threshold:                       # rag/services.py:101
    return {"answer": "I couldn't find information about this in the knowledge base.", ...}
```

therefore fires on essentially **every** query. This is the direct, mechanical cause of *"Relevant context hone ke baad bhi proper answer nahi milta"* and *"Retrieval expected documents ya chunks return nahi karta"*. The retrieval is often working; the gate throws the result away.

The 0.70 value appears to have been carried over from a model with a different score distribution (the commented-out older code at `rag/services.py:47` used `0.40`). Cosine bands are **model-specific** — 0.70 is a reasonable gate for a normalized `e5`/`bge`-family model and a nonsense gate for `nv-embed-v1`.

**(b) Ranking is actively inverted.** In config A — which is exactly what your code does today — the *distractor* (0.2958) outranks the *correct answer* (0.2608). `nv-embed-v1` is an **asymmetric, instruction-tuned** retriever: queries and passages must be encoded differently. Encoding identical text both ways gives cosine **0.4257** — i.e. the two encodings are genuinely different spaces. Your code never sends `input_type` (`providers/nvidia.py:30-48`), so queries and passages are embedded the same way, and similarity collapses toward "which chunk looks most like a generic sentence" rather than "which chunk answers this".

Adding `input_type` flips the margin from **−0.0350 to +0.1613** — a **4.6× separation improvement** and, more importantly, a sign change: the right chunk starts ranking first.

Note config C: the instruction prefix adds almost nothing **once `input_type` is set** (+0.1617 vs +0.1613). `input_type` is the lever; don't over-invest in prompt-prefix tuning for the embedder.

### 3.2 Why information is missing even when retrieval works

`extract_docx` only walks `document.paragraphs` (`extractors/docx.py:9`). `python-docx` does **not** include table cells in `.paragraphs` — they live in `document.tables`. Measured against your actual files:

```
Data_Capture_Solution_Design_Proposal_1.docx
  extractor chars : 14666      tables: 17   non-empty cells: 427
  chars INSIDE tables (DROPPED): 10860   -> 42.5% of the document is LOST

RegistrationMonitoring-ReMo_Runbook_Full.docx
  extractor chars : 54458      tables: 37   non-empty cells: 486
  chars INSIDE tables (DROPPED): 13389   -> 19.7% of the document is LOST
  lost cells include: 'Verena Wanner', 'verena.wanner@syngenta.com', '+41-613236961'
```

This is the cleanest possible explanation of *"Answers mein important information miss ho jaati hai"*. If a user asks **"Who is the ReMo contact?"**, the answer physically does not exist in Qdrant — it was dropped at extraction time, before chunking, before embedding. No amount of prompt or threshold tuning can recover it. Contact tables, RACI matrices, config tables, and step tables are exactly the high-value factual content users ask about, and they are exactly what is being discarded.

### 3.3 Why chunks silently vanish (and why uploads can hang)

`rag/chunking.py:30` — `start = end - overlap` — assumes `end` is always near `start + chunk_size`. But `end` is first pulled back to the last space (`chunking.py:15-17`), and that space can be anywhere, including near `start`.

**Reproduced — silent data loss:**

```
input: "abcde " + "x"*2000        (len=2006)
chunks produced: text[0:5], text[705:1705], text[1555:2006]
>>> characters 5..705 are in NO chunk — ~700 chars silently dropped
```

Trace: `start=0` → `rfind(" ", 0, 1000)` returns **5** → `end=5` → `start = 5 - 150 = **-145**`. Python then interprets `text[-145:855]` as `text[1861:855]` → empty string → dropped by the `if chunk_text.strip()` guard at `chunking.py:21`, and the loop resumes from `start=705`. Everything between is lost, with no error and no log.

**Reproduced — infinite loop / hang:**

```
input: "y"*150 + " " + "z"*3000
>>> HANG CONFIRMED: create_chunks did not return within 5s
```

Trace: `start=0` → the only space is at index 150 → `end=150` → `start = 150 - 150 = 0` → identical iteration forever. Because ingestion runs **inside the HTTP request** (`documents/views.py:29`), this pins a worker thread at 100% CPU permanently. One such upload degrades the whole server.

Both triggers are realistic for real documents: dense tables flattened without spaces, long URLs, base64 blobs, code fences, CJK text, or any long run without a space.

### 3.4 Why answers are generic / don't use the context

Three compounding causes:

1. **Unfiltered context.** Only `results[0]` is score-checked (`rag/services.py:101`). If it squeaks past, **all 10** chunks — including ones scoring near zero — are concatenated into the prompt (`prompts.py:31-33`). The LLM receives mostly noise and hedges. There is no per-chunk filter anywhere.
2. **Duplicate context.** `media/documents/` contains `Connect_Applications_to_Gaia_1.pdf`, `..._c4DG8RV.pdf`, and `..._x37HZA2.pdf` — the same document uploaded three times. Nothing dedups on content hash, so a 10-chunk window can be three copies of the same passage. Effective context shrinks to ~3 distinct chunks while the token cost stays at 10.
3. **Prompt ordering.** `build_prompt` (`prompts.py:35-44`) emits **Question first, Context second**, and nothing after the context. With a long context, the instruction is far from the generation point. Standard practice is Context → Question (recency), because the model attends most strongly to the tail.

### 3.5 Why "summarize" queries fail (your original question)

Measured: `cos("give me a brief summary", <real chunk>) = 0.3151` — below the 0.70 gate, so it returns not-found. But note it is **also** in the same 0.2–0.3 band as a *good* factual match. This is the deeper point: **the score band is compressed for everything**, so no single threshold can separate "summarize" from "real question" from "irrelevant". A summarization intent is not a retrieval problem at all — vector search answers *"which chunk looks like the phrase 'give me a brief summary'"*, and no chunk ever does. This needs an **intent route**, not a threshold tweak (§6.3).

---

## 4. Code-Level Findings

### C1 — Threshold set for a different model's score distribution

1. **What:** `RAG_MIN_SCORE = 0.7` gates on an absolute cosine value that `nv-embed-v1` essentially never reaches for legitimate matches.
2. **Where:** `config/settings.py:170`; enforced at `rag/services.py:99-105`.
3. **Impact:** Near-100% false-negative rate. The canned "I couldn't find information about this in the knowledge base." is returned for correct retrievals.
4. **Verify:** Run the §9.1 probe — gold chunks score 0.22–0.30 against a 0.70 gate. Or temporarily set `RAG_MIN_SCORE=0.0` and observe that real answers immediately start appearing.
5. **Fix:** Calibrate empirically, don't guess. From the measurements, `~0.15` separates gold (0.22+) from distractors (0.05–0.07) **once C2 is fixed**. Better: drop the absolute gate in favour of a **relative** gate (top score vs. the score gap) plus a reranker score, which is model-portable.
6. **Change:** see §8.1.

### C2 — Asymmetric embedding model used symmetrically

1. **What:** `input_type` (`"query"` / `"passage"`) is never sent to the embeddings endpoint.
2. **Where:** `rag/providers/nvidia.py:30-39` (`generate_embedding`, used for queries) and `:41-48` (`generate_embeddings_batch`, used for passages). Both call `client.embeddings.create(...)` with no `extra_body`.
3. **Impact:** Ranking inversion — measured margin **−0.0350**, i.e. the distractor beats the correct chunk. This is why "relevant context exists but the answer is wrong/generic".
4. **Verify:** §9.1 config A vs B. Also: embedding the *same string* as `query` vs `passage` yields cosine **0.4257**, proving the encodings differ materially.
5. **Fix:** Pass `extra_body={"input_type": "query"|"passage", "truncate": "END"}`. **Re-index is mandatory** — existing vectors were written with the wrong encoding.
6. **Change:** see §8.2.

### C3 — DOCX tables silently discarded

1. **What:** Only `document.paragraphs` is read; `document.tables` is ignored. Headers/footers too.
2. **Where:** `documents/extractors/docx.py:9-19`.
3. **Impact:** 42.5% of `Data_Capture_Solution_Design_Proposal_1.docx` and 19.7% of the ReMo runbook — including all contact names, emails, phone numbers — never reach the index. Unanswerable by construction.
4. **Verify:** §9.2 — compare `len(extract_docx(f)[0]["text"])` against the sum of `cell.text` over `Document(f).tables`.
5. **Fix:** Walk the document **body in document order** (paragraphs *and* tables interleaved), and linearize tables in a retrieval-friendly, row-wise `Header: value` form rather than raw grid text — this keeps each row semantically self-contained for embedding.
6. **Change:** see §8.3.

### C4 — Chunker: negative start (data loss) + infinite loop (hang)

1. **What:** `start = end - overlap` with no lower bound and no forward-progress guarantee.
2. **Where:** `rag/chunking.py:12-30`, specifically line 30.
3. **Impact:** (a) silent, unlogged loss of arbitrary text spans (~700 chars in the repro); (b) permanent 100%-CPU hang of a request thread, which — because ingestion is synchronous (`documents/views.py:29`) — degrades the server.
4. **Verify:** §9.3 — the two inputs in §3.3 reproduce both deterministically.
5. **Fix:** Rewrite with a guaranteed-forward-progress loop, clamped `start`, a minimum viable chunk, and token-based (not character-based) sizing. Add a structural pre-split on paragraph boundaries so chunks stop cutting mid-sentence.
6. **Change:** see §8.4.

### H1 — Only `results[0]` gated; the rest pass through unfiltered

`rag/services.py:99-110`. The threshold guards the *response*, not the *context*. Every one of the 10 chunks is handed to `build_prompt`. Fix: filter per chunk, then decide.

### H2 — No content-level deduplication

`documents/services.py:19` creates a new `Document` per upload with no content hash. Evidence: three byte-identical Gaia PDFs in `media/documents/`. Fix: SHA-256 the file on upload; reject or supersede duplicates. Additionally dedup near-identical chunk text at query time (§8.5).

### H4 — Synchronous ingestion

`documents/views.py:29` calls `process_document` in-request: extract + N embedding round-trips + upsert. A 10MB PDF can exceed any sane HTTP timeout, and C4's hang becomes a permanent thread leak. Fix: move to a task queue (Celery/RQ/`django-q`), return `202` + poll `status`. The `Document.Status.PROCESSING` state and the commented-out `ProcessDocumentView` (`documents/views.py:45-75`) show this was the original intent.

### M1 — Prompt ordering

`rag/prompts.py:31-46`. Question precedes context; no post-context instruction. Fix in §8.6.

### M2 — "Smart Source Hiding" is substring matching on model output

`rag/services.py:117-123` inspects the generated answer for `"hello"`, `"couldn't find information"`, `"specify which topic"`. Any phrasing drift, any Hindi/Hinglish reply, or an answer that merely *quotes* the phrase, silently mis-classifies. Fix: make the *route* decide whether sources apply (§6.3), not a regex over prose.

### L1/L2 — Hygiene

`rag/services.py:2` imports `document` from `docx` and never uses it (and the correct symbol is `Document`). `rag/services.py:14-15` instantiates both providers at import time, so a bad API key becomes an import-time explosion and the objects are unmockable in tests.

---

## 5. Prompt Analysis

### 5.1 Problems with the current `SYSTEM_PROMPT` (`rag/prompts.py:1-11`)

| Problem | Detail |
|---|---|
| Rules 1–3 fight the pipeline | Greeting/vague/keyword handling is written as *prompt rules*, but the request never reaches the LLM for those cases — the 0.70 gate at `services.py:101` returns the canned string first. The rules are unreachable code in prose form. |
| Rule 5 duplicates a hard-coded string | The exact "not found" sentence exists in **two** places (`prompts.py:10` and `services.py:95/103`). §4/M2 then string-matches on it — three-way coupling that drifts. |
| No grounding/citation contract | Nothing tells the model to cite `[Source N]`, so answers can't be traced, and hallucination is undetectable. |
| No conflict policy | Nothing says what to do when two chunks disagree (common with three copies of the Gaia doc). |
| No partial-answer policy | It's all-or-nothing: either answer or the canned refusal. Real queries are often *partially* covered — the model should say what it found *and* what it couldn't. This is a direct cause of *"incomplete ya generic response"*. |
| No language instruction | User writes Hinglish; the model has no guidance to mirror the user's language. |
| Vague "comprehensively" (rule 2) | Undefined; encourages padding. |

### 5.2 Improved prompts

```python
# rag/prompts.py  (replacement)

SYSTEM_PROMPT = """
You are Mukii, a document-grounded assistant.

GROUNDING
- Answer ONLY from the numbered sources in the Context block. Never use outside knowledge.
- Every factual sentence must cite its source inline as [Source N]. Cite each source at most once per sentence.
- If sources conflict, present both and attribute each: "[Source 1] says X, while [Source 3] says Y."

COVERAGE
- If the Context fully answers the question, answer it completely and specifically. Prefer exact
  values (names, emails, dates, IDs, numbers) over paraphrase.
- If the Context answers the question only PARTIALLY, answer the part you can, then add one line:
  "Not covered by the available documents: <what is missing>."
- If the Context does not address the question at all, reply with exactly:
  NO_ANSWER
  and nothing else.

STYLE
- Reply in the same language the user used (including Hinglish).
- Be direct. No preamble, no restating the question, no filler.
- Use a short bullet list only when the answer is genuinely a list (e.g. steps, contacts, options).
""".strip()
```

Design notes — each line earns its place:

- **`NO_ANSWER` sentinel** instead of a natural-language refusal: the *code* now detects not-found reliably (`answer.strip() == "NO_ANSWER"`), which kills the brittle substring matching of M2 and the three-way string coupling of §5.1. The user-facing message is then rendered once, in one place.
- **Partial-answer clause** directly targets your "incomplete/generic response" symptom — it converts a silent refusal into a useful partial answer plus an explicit gap statement.
- **Citation contract** enables both source-grounding and a cheap hallucination check (§9.5): any sentence without a `[Source N]` is suspect.
- **Greeting/summarize rules are gone** — those belong to the router (§6.3), not the prompt. A rule the pipeline never reaches is worse than no rule.

```python
def build_prompt(question, chunks):
    """Context FIRST, question LAST — the model attends most strongly to the tail."""
    context = build_context(chunks)
    return f"""Context:

{context}

---
Using only the Context above, answer this question. Cite every fact as [Source N].

Question: {question}
""".strip()


def build_context(chunks):
    sections = []
    for index, chunk in enumerate(chunks, start=1):
        loc = f"page {chunk['page_number']}" if chunk.get("page_number") else chunk.get("section") or "n/a"
        sections.append(
            f"[Source {index}] {chunk['file_name']} ({loc})\n{chunk['text']}"
        )
    return "\n\n".join(sections)
```

---

## 6. Recommended Architecture

### 6.1 Ingestion

```
upload → sha256 dedup → async task
   → extract (PDF: per page | DOCX: body-order paragraphs + linearized tables | TXT: encoding-sniffed)
   → clean (dehyphenate, drop repeated headers/footers, normalize whitespace)
   → structural split (headings/paragraphs) → token-window chunks (512 tok, 64 overlap, forward-progress-safe)
   → enrich: prepend "file | section | heading" breadcrumb to each chunk's embedded text
   → embed as input_type="passage"
   → upsert with payload {document_id, file_name, page, section, chunk_index, text, content_hash}
   → payload index on document_id
```

Two cheap, high-leverage additions:

- **Breadcrumb enrichment.** Embed `"ReMo Runbook > Contacts > Escalation\n\n<chunk text>"` rather than the bare chunk. A table-row chunk like `Email: verena.wanner@syngenta.com` is nearly meaningless in isolation; with its breadcrumb it becomes retrievable by "ReMo contact".
- **Per-document summary field.** Add `Document.summary` (one LLM call at ingest, map-reduce for long docs). This is what makes "summarize" answerable at all — §6.3 routes to it directly instead of pretending vector search can find a summary.

### 6.2 Retrieval

```
question
  → intent route (§6.3)
  → query rewrite (resolve pronouns using last 2 turns; generate 2-3 paraphrases)
  → for each variant: embed(input_type="query") → Qdrant top-30
  → hybrid merge with BM25/keyword hits (Qdrant sparse vectors or Postgres FTS)   [phase 3]
  → Reciprocal Rank Fusion across variants
  → dedup by content hash + near-duplicate cosine (>0.97)
  → rerank top-30 → top-N (cross-encoder, e.g. nvidia/nv-rerankqa-mistral-4b-v3)
  → dynamic top-k: keep while rerank_score >= rel_gate AND token_budget allows
  → build context (rerank order, breadcrumbs, budgeted)
  → generate → parse NO_ANSWER / citations → attach sources
```

**Why reranking matters most here.** Your measured gold-vs-distractor margin is +0.16 *at best* — thin. A cross-encoder reads (query, chunk) **jointly** and typically produces a far cleaner separation, which makes the gate robust instead of knife-edge. It also removes the need to trust an absolute cosine value, which is exactly what broke in C1. **This is the single highest-value addition after the four criticals.**

### 6.3 Intent router (fixes the original "summarize" complaint)

| Intent | Detection | Action |
|---|---|---|
| `GREETING` / chit-chat | tiny classifier or short regex on a ≤4-token input | reply directly, `sources: []`, no retrieval |
| `SUMMARIZE` (whole doc / all docs) | rewriter flags it, or matches "summary/summarise/brief/overview" with no specific entity | read `Document.summary` fields — **never** vector search |
| `META` ("what documents do you have?") | pattern | query Postgres, not Qdrant |
| `QA` (default) | otherwise | full §6.2 pipeline |

The router — not the system prompt, and not the threshold — is the correct home for this logic. This is what your `project_summary.md` §3 was reaching for; the analysis here confirms it, with the addition that the router must come **before** the score gate, which is where the current design traps it.

---

## 7. Implementation Plan

### Phase 0 — Stop the bleeding (~1–2 hours, unblocks everything)

| Step | Change | Expected effect |
|---|---|---|
| 0.1 | Add `input_type` to both embed methods (§8.2) | Ranking margin −0.035 → +0.16 |
| 0.2 | Lower `RAG_MIN_SCORE` to `0.15` (§8.1) | Gate stops rejecting valid hits |
| 0.3 | Fix the chunker (§8.4) | No silent loss, no hangs |
| 0.4 | Fix DOCX tables (§8.3) | +20–43% of content becomes searchable |
| 0.5 | **Re-index everything** (delete collection, re-upload) | **Mandatory** — 0.1/0.3/0.4 all change stored vectors |
| 0.6 | Filter context per chunk, not just `results[0]` (§8.5) | Less noise → less generic |

> **0.5 is not optional.** Steps 0.1, 0.3 and 0.4 all change what gets written to Qdrant. Without a full re-index you are querying old, wrongly-encoded vectors of incomplete text, and the other fixes will appear to do nothing.

### Phase 1 — Correctness & grounding (~half day)
Prompt rewrite + `NO_ANSWER` sentinel (§5.2, §8.6); delete "Smart Source Hiding"; content-hash dedup on upload; async ingestion via task queue; replace `print()` with `logging` + a per-query retrieval trace.

### Phase 2 — Retrieval quality (~1–2 days)
Reranker; dynamic top-k + token budget; intent router; `Document.summary`; breadcrumb enrichment; conversation history model + query rewriting.

### Phase 3 — Scale & rigour (~2–3 days)
Hybrid sparse+dense with RRF; evaluation harness + golden set (§9.4); retrieval/answer metrics dashboards; payload indexes; `document_id` filtering for per-document chat.

---

## 8. Suggested Code Changes

### 8.1 Threshold (`config/settings.py:170`)

```python
# BEFORE
RAG_MIN_SCORE = 0.7

# AFTER — calibrated for nv-embed-v1's measured band (gold 0.22-0.30, distractor 0.05-0.07)
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.15"))   # vector-stage recall gate
RAG_RERANK_MIN_SCORE = float(os.getenv("RAG_RERANK_MIN_SCORE", "0.5"))  # phase 2, model-portable
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "30"))    # retrieve wide
RAG_CONTEXT_K = int(os.getenv("RAG_CONTEXT_K", "6"))  # keep narrow
RAG_CONTEXT_TOKEN_BUDGET = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "6000"))
```

> Treat `0.15` as a **starting point measured on one probe**, not a constant handed down from heaven. Calibrate it on the golden set (§9.4) — that is the whole point of building one. Once the reranker lands, `RAG_MIN_SCORE` becomes a loose recall filter and `RAG_RERANK_MIN_SCORE` becomes the real decision boundary.

### 8.2 Embeddings — the highest-value patch (`rag/providers/nvidia.py`)

```python
class NVIDIAProvider:
    def __init__(self, client=None):
        self.client = client or OpenAI(
            api_key=settings.NVIDIA_API_KEY,
            base_url=settings.NVIDIA_BASE_URL,
            timeout=60.0,
            max_retries=3,
        )

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """input_type MUST be 'query' or 'passage' — nv-embed-v1 is asymmetric.
        Measured: same text encoded both ways gives cosine 0.4257."""
        response = self.client.embeddings.create(
            model=settings.NVIDIA_EMBEDDING_MODEL,
            input=texts,
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "passage")
```

Note `sorted(..., key=lambda d: d.index)`: the current code (`nvidia.py:48`) assumes `response.data` comes back in request order. The API returns an `index` field precisely because that is not guaranteed — and a silent reordering here would misalign every chunk with its vector, which is an extremely hard bug to spot later.

Call sites: `rag/services.py:85` → `embed_query`; `documents/services.py:87` → `embed_passages`.

### 8.3 DOCX extraction (`documents/extractors/docx.py`)

```python
from docx import Document as DocxFile
from docx.table import Table
from docx.text.paragraph import Paragraph


def _iter_block_items(parent):
    """Yield Paragraphs and Tables in true document order."""
    from docx.oxml.ns import qn
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _linearize_table(table) -> str:
    """Row-wise 'Header: value' — keeps each row self-contained for embedding.
    A bare grid dump embeds terribly; 'Email: x@y.com' embeds well."""
    rows = [[c.text.strip() for c in r.cells] for r in table.rows]
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    header_is_labels = any(h for h in header) and len(rows) > 1
    out = []
    for row in (body if header_is_labels else rows):
        if not any(row):
            continue
        if header_is_labels:
            out.append(" | ".join(
                f"{h}: {v}" for h, v in zip(header, row) if v
            ))
        else:
            out.append(" | ".join(v for v in row if v))
    return "\n".join(out)


def extract_docx(file_path):
    doc = DocxFile(file_path)
    blocks, current_heading = [], None

    for item in _iter_block_items(doc):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            if item.style is not None and item.style.name.startswith("Heading"):
                current_heading = text
            blocks.append({"section": current_heading, "text": text})
        else:
            table_text = _linearize_table(item)
            if table_text:
                blocks.append({"section": current_heading, "text": table_text})

    return [{
        "page_number": None,
        "section": b["section"],
        "text": b["text"],
    } for b in blocks]
```

This recovers the 42.5% / 19.7% and carries a `section` breadcrumb for §6.1 enrichment. It returns one entry per block rather than one giant blob, so chunking can respect real boundaries.

### 8.4 Chunker (`rag/chunking.py`) — full replacement

```python
def create_chunks(pages, chunk_size=1000, overlap=150, min_chunk=50):
    """Character-window chunker with guaranteed forward progress.

    Fixes two reproduced bugs in the previous implementation:
      * start could go negative -> text[-145:855] silently dropped ~700 chars
      * start could stall  -> infinite loop, 100% CPU, request never returns
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks, chunk_index = [], 0

    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Prefer a clean break, but never break so early that we lose progress.
            if end < len(text):
                window_floor = start + max(min_chunk, chunk_size // 2)
                for sep in ("\n\n", "\n", ". ", " "):
                    cut = text.rfind(sep, window_floor, end)
                    if cut != -1:
                        end = cut + (len(sep) if sep != " " else 0)
                        break

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= min_chunk or (chunk_text and end >= len(text)):
                chunks.append({
                    "chunk_index": chunk_index,
                    "page_number": page.get("page_number"),
                    "section": page.get("section"),
                    "text": chunk_text,
                })
                chunk_index += 1

            if end >= len(text):
                break

            next_start = end - overlap
            # THE FIX: forward progress is now structurally guaranteed.
            start = next_start if next_start > start else end

    return chunks
```

Three guarantees the old code lacked: `end` is clamped to `len(text)`; the break-point search has a **floor** (`window_floor`) so a stray early space can't collapse the window; and `start` is **monotonically increasing** (`next_start if next_start > start else end`), which makes both the hang and the negative-index loss structurally impossible rather than merely unlikely.

> Longer term, size by **tokens** (`tiktoken` or the NIM tokenizer), not characters — 1000 chars is ~250 tokens for English but wildly different for code, tables, or CJK.

### 8.5 Query pipeline (`rag/services.py`) — Phase 0/1 shape

```python
import hashlib
import logging

logger = logging.getLogger(__name__)


def _dedup(chunks):
    """Three copies of the Gaia PDF must not consume three context slots."""
    seen, out = set(), []
    for c in chunks:
        h = hashlib.sha256(c["text"].strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(c)
    return out


def answer_question(question, history=None):
    nvidia, qdrant = get_providers()          # lazy — no import-time side effects (L2)

    route = classify_intent(question, history)          # §6.3
    if route.intent == "GREETING":
        return {"answer": nvidia.simple_chat(question), "sources": []}
    if route.intent == "SUMMARIZE":
        return summarize_documents(route.document_ids)  # uses Document.summary, not Qdrant

    query_embedding = nvidia.embed_query(route.rewritten_query)
    hits = qdrant.search_chunks(query_embedding, limit=settings.RAG_TOP_K)

    # H1: filter EVERY chunk, not just hits[0]
    kept = [h for h in hits if h["score"] >= settings.RAG_MIN_SCORE]
    kept = _dedup(kept)
    kept = rerank(route.rewritten_query, kept)          # Phase 2; identity fn until then
    kept = budget(kept, settings.RAG_CONTEXT_TOKEN_BUDGET)[: settings.RAG_CONTEXT_K]

    logger.info(
        "rag.retrieval q=%r intent=%s hits=%d kept=%d top=%.4f",
        question, route.intent, len(hits), len(kept),
        hits[0]["score"] if hits else -1,
    )

    if not kept:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    answer = nvidia.generate_answer(SYSTEM_PROMPT, build_prompt(route.rewritten_query, kept))

    # M2 gone: the model signals not-found structurally, we don't grep its prose.
    if answer.strip() == "NO_ANSWER":
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    return {"answer": answer, "sources": cited_sources(answer, kept)}
```

The `logger.info` line is deliberately load-bearing: **you currently cannot tell a retrieval failure from a generation failure from a threshold rejection**, because all three return the identical canned string with no trace. That one log line would have made C1 obvious on day one.

### 8.6 Prompts

See §5.2 for the full replacement of `SYSTEM_PROMPT`, `build_prompt`, and `build_context`.

---

## 9. Testing and Evaluation Plan

### 9.1 Embedding probe (already run — the basis of §3.1)
For each config (no `input_type` / with / with+instruction), embed a gold chunk, a distractor, and an unrelated sentence; report cosine and the `gold − best_wrong` margin. **Pass criterion: margin > 0.** Current code fails at **−0.0350**; `input_type` alone yields **+0.1613**.

### 9.2 Extraction-coverage test (already run — the basis of §3.2)
Assert extracted characters ≥ 95% of (paragraph chars + table-cell chars) for every file in `media/documents/`. Currently **57.5%** for the Data Capture proposal.

```python
def test_docx_extraction_includes_tables():
    path = "media/documents/Data_Capture_Solution_Design_Proposal_1.docx"
    got = " ".join(b["text"] for b in extract_docx(path))
    assert "verena.wanner@syngenta.com" in got or "Aspect" in got  # table-only content
```

### 9.3 Chunker property tests (already reproduced — the basis of §3.3)

```python
import pytest
from hypothesis import given, strategies as st

@pytest.mark.timeout(5)          # the old code hangs forever here
@given(st.text(min_size=1, max_size=5000))
def test_chunker_terminates_and_loses_nothing(text):
    chunks = create_chunks([{"page_number": 1, "text": text}])
    joined = "".join(c["text"] for c in chunks)
    # every non-whitespace char must survive in at least one chunk
    assert len("".join(text.split())) <= len("".join(joined.split())) + 0

def test_no_data_loss_on_early_space():
    text = "abcde " + "x" * 2000
    chunks = create_chunks([{"page_number": 1, "text": text}])
    assert "".join(c["text"] for c in chunks).count("x") == 2000

@pytest.mark.timeout(5)
def test_no_infinite_loop_on_boundary_space():
    create_chunks([{"page_number": 1, "text": "y" * 150 + " " + "z" * 3000}])
```

### 9.4 Golden set — the thing that makes "better" measurable
Build `task/golden.jsonl`, 30–50 rows, drawn from your real documents and biased toward what actually breaks: table facts, contacts, multi-hop questions, summarize requests, and out-of-scope questions that *must* return not-found.

```json
{"q": "Who is the contact person for ReMo?", "must_include": ["Verena Wanner", "verena.wanner@syngenta.com"], "gold_doc": "RegistrationMonitoring-ReMo_Runbook_Full.docx", "intent": "QA"}
{"q": "give me a brief summary", "intent": "SUMMARIZE"}
{"q": "What is the capital of France?", "expect_not_found": true, "intent": "QA"}
```

### 9.5 Metrics

**Retrieval** (needs only the golden set, no LLM): Recall@k (gold chunk in top-k), MRR, nDCG@10, and **not-found rate** — the last one is your canary; it is currently near 100%.

**Answer quality:** exact-match/containment on `must_include`; **citation validity** (every `[Source N]` resolves to a supplied chunk — cheap hallucination detector); **groundedness** via an LLM-judge scoring each sentence against the cited chunk; **false-refusal rate** (answerable questions that returned `NO_ANSWER`) and **false-answer rate** (out-of-scope questions that got an answer).

### 9.6 Before/after protocol
Freeze the golden set → run the harness on `main` → apply Phase 0 → **re-index** → re-run → diff the table. Report Recall@10, not-found rate, false-refusal rate, and containment. Do not tune `RAG_MIN_SCORE` by feel; sweep it over `[0.05 … 0.40]` on the golden set and pick the knee.

---

## 10. Final Prioritized Action Plan

| Rank | Action | Effort | Expected impact | Evidence |
|---|---|---|---|---|
| 1 | **Add `input_type` query/passage** (§8.2) | 15 min | Ranking margin **−0.035 → +0.161**; correct chunk starts ranking first | Measured §3.1 |
| 2 | **Lower `RAG_MIN_SCORE` 0.70 → ~0.15** (§8.1) | 2 min | Unblocks ~all queries; gold scores 0.22–0.30 vs a 0.70 gate | Measured §3.1 |
| 3 | **Re-index everything** (delete collection + re-upload) | 20 min | **Mandatory** — 1, 4, 5 all change stored vectors; without it they appear to do nothing | — |
| 4 | **Fix DOCX table extraction** (§8.3) | 1 hr | +42.5% / +19.7% of content becomes answerable; recovers all contact data | Measured §3.2 |
| 5 | **Fix the chunker** (§8.4) | 1 hr | Ends silent text loss and the request-hang | Reproduced §3.3 |
| 6 | Filter context per chunk + dedup (§8.5) | 1 hr | Less noise → fewer generic answers | Code H1/H2 |
| 7 | Add retrieval logging (§8.5) | 30 min | Makes the next bug diagnosable in minutes, not days | Code L3 |
| 8 | Prompt rewrite + `NO_ANSWER` (§5.2) | 1 hr | Partial answers, citations, kills brittle string-matching | Code M2 |
| 9 | Golden set + harness (§9.4) | half day | Turns "feels better" into a number; required to tune #2 properly | — |
| 10 | Async ingestion (§7 Phase 1) | half day | No timeouts; hang can't take the server down | Code H4 |
| 11 | **Reranker** (§6.2) | half day | Biggest quality jump after Phase 0; makes the gate robust | Analysis §6.2 |
| 12 | Intent router + `Document.summary` (§6.3) | 1 day | Finally answers "summarize" — your original question | Measured §3.5 |
| 13 | Conversation history + query rewriting | 1 day | Follow-up questions work | Code H5 |
| 14 | Hybrid search + RRF | 1–2 days | Recall on exact IDs/codes/names | Analysis §6.2 |

**Items 1–3 are ~40 minutes total and should be done in one sitting.** They are, by the measurements, the difference between a bot that answers almost nothing and a bot that answers most things.

---

## Appendix A — Recommended end-to-end flow (target state)

```
INGEST
  upload → sha256 dedup → 202 Accepted → async worker
    → extract  (PDF: per-page | DOCX: body-order paragraphs + linearized tables | TXT: sniffed encoding)
    → clean    (dehyphenate, strip repeated headers/footers, normalize whitespace)
    → split    (headings/paragraphs → 512-token windows, 64 overlap, forward-progress-safe)
    → enrich   (prepend "file > section > heading" breadcrumb to embedded text)
    → summarize (map-reduce → Document.summary)
    → embed    (input_type="passage")
    → upsert   (payload: document_id, file_name, page, section, chunk_index, text, content_hash)
                + payload index on document_id

QUERY
  question (+ last N turns)
    → intent route ─── GREETING ──→ direct LLM reply, sources: []
         │             SUMMARIZE ─→ Document.summary map-reduce, sources: [docs]
         │             META ──────→ Postgres query
         └── QA ↓
    → query rewrite (resolve pronouns, 2-3 paraphrases)
    → embed each (input_type="query") → Qdrant top-30  [+ BM25 top-30]
    → RRF merge → dedup (hash + cosine>0.97)
    → cross-encoder rerank → top-N by rerank_score >= gate, within token budget
    → build context (rerank order, breadcrumbs, [Source N] labels)
    → generate (context first, question last, citation contract)
    → parse: NO_ANSWER → not-found | else validate [Source N] → attach sources
    → log: intent, variants, hit count, top scores, kept count, latency, tokens
```

## Appendix B — Reproduction scripts

The three probes behind this analysis (§9.1 embeddings, §9.2 extraction coverage, §9.3 chunker) were run against this repo and the live NVIDIA API on 2026-07-17. They live in the session scratchpad:

- `verify_chunking.py` — reproduces the negative-start data loss and the infinite loop
- `verify_docx.py` — quantifies dropped table content per file in `media/documents/`
- `verify_embed.py` / `verify_fix.py` — probe the live embeddings endpoint and produce the §3.1 table

Recommend moving these into `backend/rag/tests/` (§9.3) so they run in CI rather than living in a temp folder.
