# Mukii RAG Chatbot

Mukii is a Django-based Retrieval-Augmented Generation (RAG) chatbot for answering questions from uploaded documents. The application accepts PDF, DOCX, and TXT files, extracts their text, splits it into chunks, creates embeddings, stores the chunks in Qdrant, and uses those chunks as context when answering user questions.

This repository is a local development project. The current implementation uses Django for the web app, SQLite for the relational database, Qdrant for vector search, and NVIDIA-hosted models for embeddings, reranking, and chat completions.

## 1. Project overview

### What the project does

- Lets a user upload a document through a web or API flow.
- Extracts text from the uploaded file.
- Cleans and chunks the text.
- Generates embeddings for each chunk.
- Stores chunks and their embeddings in Qdrant.
- Answers questions by retrieving relevant chunks and sending them to an NVIDIA chat model.

### Current implementation status

The codebase includes the following working pieces:

- Document upload and processing via Django views and services.
- Basic authentication with Django sessions.
- A simple chat UI served by Django templates.
- REST endpoints for login, document upload/delete, and chat questions.
- Vector indexing and retrieval through Qdrant.
- Reranking and answer generation through NVIDIA APIs.

### Known limitations

- **Synchronous ingestion.** Document upload and indexing happen inside the request. Large documents can make the request slow or time out.
- **The ask endpoint is unauthenticated.** `AskQuestionView` uses `AllowAny`, so anyone who can reach the server can query the indexed documents. Uploading and deleting do require a login.
- **Source citations collapse for DOCX and TXT.** Citations are de-duplicated by file name and page number, and those formats have no page number, so a multi-chunk answer still reports a single source.
- **Conversation memory is session-scoped and limited.** The recent chat window is used to resolve follow-ups, but very long threads may still lose earlier context.
- **Very broad requests still depend on the indexed material.** A generic request such as "summarise this document" can still fall back to the not-found message when the available chunks do not support that answer well.
- **No automated tests.** See section 17.

## 2. Key features

- Upload documents in PDF, DOCX, or TXT format
- Validate file type and file size
- Extract text from supported formats
- Clean text and split it into chunks
- Generate embeddings for chunks
- Store chunks in Qdrant with dense and sparse vectors
- Retrieve relevant chunks for a question
- Rerank candidates before prompt construction
- Generate grounded answers from retrieved context
- Route greetings, overviews, and knowledge questions through the right path
- Use recent conversation turns to resolve follow-up questions
- Provide a simple chat UI and REST APIs

## 3. Tech stack

| Layer | Technology |
|---|---|
| Backend | Django 5.2, Django REST Framework |
| Database | SQLite (default) |
| Vector store | Qdrant |
| AI / embeddings | NVIDIA OpenAI-compatible API |
| Document parsing | pypdf, python-docx |
| Environment handling | python-dotenv |
| Testing | pytest, pytest-django |

## 4. Repository structure

```text
backend/
  accounts/          Django auth-related views and URLs
  chat/              Chat API and UI views
  config/            Django settings, URLs, WSGI/ASGI entrypoints
  documents/         Upload, extraction, storage, and document management
  rag/               Chunking, prompts, retrieval, provider integrations
  static/            Front-end assets (CSS/JS)
  templates/         Django templates for the chat UI
  manage.py          Django management entrypoint
postman/            Postman collection and environment files
project_summary.md  Additional analysis notes
pyproject.toml       Python dependency declarations
```

## 5. Prerequisites

- Python 3.12+
- A working NVIDIA API key
- A running Qdrant instance
- Docker is recommended for running Qdrant locally

The project declares dependencies in [pyproject.toml](pyproject.toml), but the verified local setup in this environment was to install the listed packages directly with pip.

## 6. Local setup guide

### 6.1 Create and activate a virtual environment

```bash
cd /path/to/AI-BOT
python -m venv .venv
source .venv/bin/activate
```

### 6.2 Install Python dependencies

The following command was verified in this workspace:

```bash
python -m pip install --upgrade pip
python -m pip install "django>=5.2,<5.3" "django-environ>=0.14.0" "djangorestframework>=3.17.1" "drf-spectacular>=0.30.0" "httpx>=0.28.1" "openai>=2.45.0" "pypdf>=6.14.2" "python-docx>=1.2.0" "python-dotenv>=1.2.2" "qdrant-client>=1.18.0" "pytest>=9.1.1" "pytest-django>=4.12.0" "ruff>=0.15.21"
```

### 6.3 Create environment file

Copy the example file to the Django project directory:

```bash
cp backend/.env.example backend/.env
```

Then edit [backend/.env](backend/.env) and fill in the values.

## 7. Required environment variables

The settings file loads values from [backend/.env](backend/.env) through Django settings.

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `DEBUG` | No | `True` | Enables Django debug mode |
| `DJANGO_SECRET_KEY` | Recommended | None | Django secret key |
| `NVIDIA_API_KEY` | Yes | None | API key for NVIDIA models |
| `NVIDIA_BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | NVIDIA API endpoint |
| `NVIDIA_CHAT_MODEL` | Yes | None | Model used for chat completion |
| `NVIDIA_EMBEDDING_MODEL` | Yes | None | Model used for embeddings |
| `NVIDIA_RERANK_MODEL` | No | `nvidia/rerank-qa-mistral-4b` | Model used for reranking |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant service URL |
| `QDRANT_API_KEY` | No | None | Qdrant API key if needed |
| `QDRANT_COLLECTION_NAME` | No | `document_chunks_v2` | Qdrant collection name |
| `RAG_ENABLE_LLM_INTENT` | No | `True` | Enable LLM intent classification fallback |
| `RAG_INTENT_MAX_TOKENS` | No | `80` | Max tokens for the intent classifier |
| `RAG_INTENT_TIMEOUT` | No | `10` | Timeout in seconds for intent classification |
| `RAG_OVERVIEW_MAX_DOCUMENTS` | No | `5` | Number of recent ready documents sampled for overview |
| `RAG_OVERVIEW_DOC_CHUNKS` | No | `3` | Number of opening chunks per document for overview queries |

Example:

```env
DEBUG=True
DJANGO_SECRET_KEY=change-this-secret-key

NVIDIA_API_KEY=your-nvidia-key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embed-v1
NVIDIA_RERANK_MODEL=nvidia/rerank-qa-mistral-4b
RAG_ENABLE_LLM_INTENT=True
RAG_INTENT_MAX_TOKENS=80
RAG_INTENT_TIMEOUT=10
RAG_OVERVIEW_MAX_DOCUMENTS=5
RAG_OVERVIEW_DOC_CHUNKS=3

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=document_chunks_v2
```

## 8. Database and migration setup

The project uses SQLite by default. The database file is created at [backend/db.sqlite3](backend/db.sqlite3) after migrations.

Run the migrations:

```bash
cd backend
python manage.py migrate
```

This command was verified in the workspace and completed successfully.

### Create a Django user

To use the login flow, create a Django superuser or a regular user:

```bash
python manage.py createsuperuser
```

The authentication endpoint expects a valid Django user. The Postman collection defaults to the username `mukund` and expects you to set the password in the environment.

### 8.1 Re-indexing documents

`reindex_documents` re-extracts, re-embeds, and re-uploads stored documents into Qdrant. Run it after wiping or recreating the Qdrant storage, after switching `QDRANT_COLLECTION_NAME`, or after any change to extraction or chunking — the vectors already in Qdrant are not updated automatically.

```bash
cd backend
python manage.py reindex_documents                  # every document
python manage.py reindex_documents --ids 1 2        # only these document ids
python manage.py reindex_documents --purge-orphans  # also drop vectors whose document row is gone
```

The command reads the original files from `backend/media/`, so those files must still be present.

## 9. Run the application

### 9.1 Start Qdrant

The application expects Qdrant to be reachable at the configured URL. A local Docker-based start command is documented in the project and was used in the repository notes:

```bash
docker start qdrant
```

If Qdrant is not running, the chat API will return a 503 error with a message about the vector database being unavailable.

### 9.2 Start Django

```bash
cd backend
python manage.py runserver 0.0.0.0:8000
```

The server should be available at:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/api/chat/

The root URL redirects to the chat UI path.

## 10. AI bot architecture and end-to-end flow

### 10.1 Upload and ingestion flow

1. A document is uploaded through the document upload endpoint.
2. The file is saved as a Django `Document` model instance.
3. The upload view calls the document processing service.
4. The text is extracted depending on the file type:
   - PDF: parsed with `pypdf`
   - DOCX: parsed with `python-docx`
   - TXT: read as plain text
5. The extracted text is cleaned.
6. The cleaned text is split into chunks.
7. Each chunk is embedded using the NVIDIA embedding model.
8. The embeddings and payloads are written into Qdrant.
9. The document record is marked as ready.

### 10.2 Retrieval and answer flow

1. The user sends a question through the chat API or the web UI.
2. The message is classified as small talk, overview, or knowledge.
3. Small talk questions are answered directly by the chat model without retrieval.
4. Overview requests sample opening chunks from the most recently ready documents.
5. Knowledge questions are rewritten with the recent conversation in mind and then embedded with the NVIDIA embedding model.
6. Qdrant is queried with both dense vector similarity and sparse keyword-style matching.
7. The results are fused and passed to a reranker.
8. The reranker scores the retrieved passages against the question.
12. The response is returned with a list of sources.

## 11. How the RAG pipeline works

### Document ingestion

The ingestion pipeline is implemented in [backend/documents/services.py](backend/documents/services.py).

- The `DocumentService.upload()` method creates a database record and saves the uploaded file.
- The `DocumentService.process_document()` method handles text extraction, cleaning, chunking, embedding, and Qdrant indexing.
- The current implementation performs these steps synchronously.

### Text extraction

The extractors are in [backend/documents/extractors](backend/documents/extractors):

- [backend/documents/extractors/pdf.py](backend/documents/extractors/pdf.py) extracts per-page text from PDFs.
- [backend/documents/extractors/docx.py](backend/documents/extractors/docx.py) extracts paragraphs and tables from DOCX files.
- [backend/documents/extractors/text.py](backend/documents/extractors/text.py) reads plain text files.

The DOCX extractor walks the document body in order rather than reading `document.paragraphs`, because paragraph iteration alone skips every table. On the reference runbook in this repository that accounted for roughly 20% of the document text, including the contact, environment, and parameter tables. It emits:

- **Headings** prefixed with Markdown hashes (`#`, `##`, `###`) matching the Word heading level, so a chunk taken from the middle of a section still carries its section title. The system prompt tells the model these are document headings, which is what lets it answer questions about document structure.
- **Table rows** as `label | value`, one row per line, which keeps a label next to its value when a table is split across chunks.
- **Wide tables** (three or more columns) with the header cells repeated on every data row as `header: value`. Without this, a row that lands in a different chunk from its header row is unreadable.

Only PDFs carry real page numbers. DOCX and TXT are extracted as a single unnumbered section, so `page_number` is `null` for those formats.

### Text cleaning

The cleaner in [backend/documents/text_cleaner.py](backend/documents/text_cleaner.py) removes extra whitespace and normalizes line breaks before chunking.

### Chunking

Chunking is implemented in [backend/rag/chunking.py](backend/rag/chunking.py).

The current strategy:

- Uses a chunk size of about 1000 characters.
- Uses a 150-character overlap.
- Splits on spaces when possible.
- Merges very short trailing pieces into the previous chunk.

### Embeddings

Embedding generation uses [backend/rag/providers/nvidia.py](backend/rag/providers/nvidia.py).

- The embedding model comes from the `NVIDIA_EMBEDDING_MODEL` setting.
- Embeddings are produced in batches of 50 chunks during document processing.

### Search and vector storage

Vector storage and retrieval are implemented in [backend/rag/providers/qdrant.py](backend/rag/providers/qdrant.py).

- The app creates a Qdrant collection when needed.
- Each chunk is stored as a point with:
  - a named dense vector (`dense`), from the NVIDIA embedding model
  - a named sparse vector (`text`), built from token frequencies. Qdrant is configured with the `IDF` modifier, so inverse document frequency is computed server-side and no extra embedding model or dependency is needed for the keyword side.
  - document metadata such as `document_id`, `file_name`, `page_number`, `chunk_index`, and `text`
- Search runs a dense prefetch and a sparse prefetch, then fuses them with Reciprocal Rank Fusion inside Qdrant. Dense retrieval is weak on exact literals such as account numbers and repository names, which the sparse side ranks first; fusion means neither retriever has to win outright.
- Point ids are derived deterministically from `document_id` and `chunk_index`, so re-indexing a document overwrites its points instead of duplicating them.

**Collection migration.** Named dense and sparse vectors cannot be added to a collection that was created without them. The default collection name is therefore `document_chunks_v2`. If you are coming from an older checkout that used `document_chunks`, point `QDRANT_COLLECTION_NAME` at the new name and re-index (see section 8.1). The old collection is left untouched and can be deleted once you are satisfied with the new one.

### Neighbour expansion

After reranking, [backend/rag/providers/qdrant.py](backend/rag/providers/qdrant.py) pulls in the chunk immediately before and after each surviving hit and orders the whole set by document position.

A section that spans a chunk boundary would otherwise only ever return its first half: the continuation answers no question on its own, so it ranks too low to be retrieved. Neighbours are therefore selected by position rather than by score. They are added to the prompt context only; citations are still built from the reranked hits.

### Reranking

The reranking step is supported in [backend/rag/services.py](backend/rag/services.py).

- The code first searches Qdrant, which returns a wide candidate set (24 by default). No similarity threshold is applied at this stage: retrieval only has to surface candidates, and cosine scores are not comparable between different questions.
- It then runs the NVIDIA reranker over those candidates. The reranker is a cross-encoder, and its logits *are* comparable across questions, so a single constant can act as the relevance gate.
- If reranking is unavailable, the code logs a warning and falls back to the fused retrieval order so the chat keeps working.

The default rerank settings in [backend/config/settings.py](backend/config/settings.py) are:

- `RAG_TOP_N = 6` — chunks handed to the model after reranking
- `RAG_RERANK_FLOOR = -11.0` — logit below which a passage is treated as irrelevant

The floor was calibrated against this repository's reference document over fourteen questions: the lowest-scoring chunk that genuinely contained an answer scored about `-10.2`, while the best chunk for a question the corpus could not answer scored about `-12.1`. The default sits between the two. **Recalibrate it if you change the reranker model or index a materially different corpus**, and do so against a labelled question set rather than by feel.

Note that reranking is served from a different host (`https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking`) than chat and embeddings. `NVIDIA_BASE_URL` does not affect it, which is why the reranker is called over plain HTTP rather than through the OpenAI-compatible client.

### Prompt building and answer generation

Prompt construction is in [backend/rag/prompts.py](backend/rag/prompts.py).

The prompt builder creates a context block from the retrieved chunks and places the question after that context. The system prompt covers:

- **How to read the context.** It explains that `#`/`##`/`###` lines are section headings copied from the document and that passages are supplied in document order, which is what allows structural questions ("what section follows X") to be answered. It also explains that `|` lines are table rows.
- **Grounding.** Answer only from the context, and never speculate — the phrasings the model reached for ("it might be", "typically", "usually") are named explicitly, because a general "do not guess" instruction was not enough to stop it inventing plausible document structure.
- **Partial answers.** Preferred over refusing outright, with a short line naming what is missing.
- **Verbatim reproduction.** Requests for the full text of a section must reproduce every table row rather than summarising.
- **Not-found handling.** Only when the context is entirely unrelated, reply with the exact not-found sentence, which the source builder also keys on to suppress citations.

Answer generation is performed by [backend/rag/providers/nvidia.py](backend/rag/providers/nvidia.py), using the configured chat model.

The system prompt begins with `/no_think`, which suppresses the reasoning trace on the `nemotron` chat models. The directive is honoured inconsistently: when it is ignored, the reasoning trace consumes the token budget, which shows up as occasional slow responses and answers that stop mid-sentence.

### Intent routing and conversation memory

Intent routing is implemented in [backend/rag/intents.py](backend/rag/intents.py) and [backend/rag/services.py](backend/rag/services.py). The classifier routes a turn as small talk, overview, or knowledge before retrieval starts. That prevents greetings and broad context requests from being treated as failed retrievals.

Conversation memory is stored in the Django session by [backend/chat/history.py](backend/chat/history.py). The history window is trimmed in [backend/rag/history.py](backend/rag/history.py) so follow-up questions such as "aur carry forward ka?" or "iska matlab?" can be rewritten into a searchable query with the earlier topic in mind.

## 12. API documentation

The project exposes these main API routes:

### 12.1 Login

**Endpoint**

```http
POST /api/accounts/login/
```

**Request body**

```json
{
  "username": "mukund",
  "password": "your-password"
}
```

**Response**

```json
{
  "message": "Login Successful"
}
```

### 12.2 Upload document

**Endpoint**

```http
POST /api/documents/upload/
```

**Request**

- `multipart/form-data`
- field name: `file`

**Supported extensions**

- `.pdf`
- `.docx`
- `.txt`

**Response**

```json
{
  "id": 1,
  "original_name": "sample.txt",
  "status": "ready",
  "total_pages": 1,
  "total_chunks": 3
}
```

### 12.3 Delete document

**Endpoint**

```http
DELETE /api/documents/<document_id>/
```

**Response**

```json
{
  "detail": "Document successfully deleted."
}
```

### 12.4 Ask a question

**Endpoint**

```http
POST /api/chat/ask/
```

**Request body**

```json
{
  "question": "What is this document about?"
}
```

**Response**

```json
{
  "answer": "The answer generated from the retrieved context.",
  "sources": [
    {
      "document_name": "runbook.docx",
      "page_number": null,
      "score": -4.21
    }
  ]
}
```

`score` is the reranker logit, not a similarity value. It is unbounded and usually negative; higher is more relevant. Anything below `RAG_RERANK_FLOOR` is dropped before the answer is generated. `page_number` is `null` for DOCX and TXT sources.

When no passage clears the floor, the endpoint still returns `200` with the not-found message and an empty `sources` list:

```json
{
  "answer": "I couldn't find information about this in the knowledge base.",
  "sources": []
}
```

This endpoint currently permits unauthenticated requests; see the known limitations in section 1.

### 12.5 Chat UI

**Endpoint**

```http
GET /api/chat/
```

This returns the HTML chat UI.

## 13. Postman collection setup and usage

The repository includes a Postman collection and environment under [postman](postman):

- [postman/Mukii-RAG.postman_collection.json](postman/Mukii-RAG.postman_collection.json)
- [postman/Mukii-RAG.postman_environment.json](postman/Mukii-RAG.postman_environment.json)
- [postman/README.md](postman/README.md)

### Import steps

1. Open Postman.
2. Import both JSON files from [postman](postman).
3. Select the environment named `Mukii RAG - Local`.
4. Set the `username` and `password` variables to a real Django user.

### Recommended request order

1. `Get CSRF Cookie`
2. `Login`
3. `Upload Document`
4. `Ask Question`
5. `Delete Document`

The Postman README documents the CSRF behavior and explains why the chat ask request needs a CSRF token after login.

## 14. User guide

### How to use the application

1. Start Qdrant.
2. Start Django.
3. Create or use a Django user.
4. Open the chat UI at http://127.0.0.1:8000/api/chat/.
5. Upload a supported document.
6. Wait for the document to reach the `ready` state.
7. Ask a question about the uploaded document.

### What to expect

- The bot answers from the uploaded document content, not from general knowledge.
- It returns source references when the answer uses retrieved passages.
- If the retrieval system cannot find relevant context, the response may say that the information was not found in the knowledge base.

## 15. Developer guide

### Main Django apps

- `accounts`: login endpoint and session-based authentication
- `documents`: file upload, storage, extraction, and processing
- `chat`: chat views, serializers, and UI routes
- `rag`: chunking, prompt building, retrieval, and NVIDIA integration

### Important services and modules

- [backend/documents/services.py](backend/documents/services.py): main document ingestion service
- [backend/rag/services.py](backend/rag/services.py): main retrieval and answer generation flow
- [backend/rag/providers/nvidia.py](backend/rag/providers/nvidia.py): NVIDIA embedding, reranking, and chat completion logic
- [backend/rag/providers/qdrant.py](backend/rag/providers/qdrant.py): Qdrant collection management, indexing, and search
- [backend/rag/chunking.py](backend/rag/chunking.py): chunk creation logic
- [backend/rag/prompts.py](backend/rag/prompts.py): system prompt and prompt assembly

### Request flow summary

- Web/API request enters a Django view.
- The view validates input and calls the relevant service.
- The service uses the providers to talk to NVIDIA and Qdrant.
- The final answer is returned as JSON or rendered in the chat UI.

## 16. Common errors and troubleshooting

### `ModuleNotFoundError: No module named 'django'`

Install the Python dependencies first.

### `Missing credentials` from the NVIDIA provider

The app requires `NVIDIA_API_KEY`, `NVIDIA_CHAT_MODEL`, and `NVIDIA_EMBEDDING_MODEL` to be set in [backend/.env](backend/.env).

### `503 Service Unavailable` from the chat API

This usually means Qdrant is not reachable. Make sure the Qdrant service is running and reachable at the configured URL.

### `403` on chat ask requests

The chat ask endpoint does not require a login, but Django's session authentication enforces CSRF once a session exists. So the request succeeds while logged out and starts returning `403` after logging in unless the client sends the current CSRF token.

### The bot answers "I couldn't find information about this in the knowledge base"

Work outwards from the index before touching the prompt:

1. Confirm the document reached `ready` and that the text is actually in Qdrant. If extraction or chunking changed since the upload, re-index (section 8.1).
2. Check whether the passage is reaching the model at all. Retrieval fetches 24 candidates, the reranker keeps at most `RAG_TOP_N`, and anything below `RAG_RERANK_FLOOR` is discarded.
3. A passage that scores just below the floor is the common cause. Recalibrate against a labelled question set rather than lowering the floor until the answer appears — too low a floor lets unanswerable questions through and produces confident nonsense.

Corpus-wide requests such as "summarise this document" are expected to return this message; see the known limitations in section 1.

### Upload fails with a validation error

The backend only accepts `.pdf`, `.docx`, and `.txt` files, and the maximum file size is 10 MB.

### Postman fails to upload a fixture

The collection uses fixtures under [postman/fixtures](postman/fixtures). If Postman cannot resolve them automatically, choose the file manually in the request body.

## 17. Testing instructions

The repository currently contains empty Django test modules and no active test coverage. The verified test command is:

```bash
cd backend
pytest
```

In this workspace it reported:

```text
collected 0 items
no tests ran
```

## 18. Deployment-related notes

The codebase does not include a production deployment configuration such as Docker Compose, Kubernetes manifests, or a production-ready WSGI/ASGI deployment setup. The implemented workflow is intended for local development and testing with Django, SQLite, and a local Qdrant instance.

## 19. Verified commands in this workspace

The following commands were verified while preparing this README:

```bash
cd backend
python manage.py check
python manage.py migrate
pytest
```

The migration command completed successfully. The current test suite reports no executable tests.
