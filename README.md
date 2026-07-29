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

### Important limitation

Document upload and indexing currently happen synchronously inside the request. This means large documents can make the request slow or time out.

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
cd /workspaces/AI-BOT
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

Example:

```env
DEBUG=True
DJANGO_SECRET_KEY=change-this-secret-key

NVIDIA_API_KEY=your-nvidia-key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embed-v1
NVIDIA_RERANK_MODEL=nvidia/rerank-qa-mistral-4b

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
2. The question is embedded with the NVIDIA embedding model.
3. Qdrant is queried with both dense vector similarity and sparse keyword-style matching.
4. The results are fused and passed to a reranker.
5. The reranker scores the retrieved passages against the question.
6. The top results are enriched with neighbouring chunks to preserve context around each hit.
7. The prompt builder assembles the context and the question.
8. The NVIDIA chat model generates the final answer.
9. The response is returned with a list of sources.

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
  - dense vector data
  - sparse vector data
  - document metadata such as `document_id`, `file_name`, `page_number`, `chunk_index`, and `text`
- The search flow uses both dense and sparse retrieval and then fuses the results.

### Reranking

The reranking step is supported in [backend/rag/services.py](backend/rag/services.py).

- The code first searches Qdrant.
- It then runs the NVIDIA reranker over the search results.
- If reranking is unavailable, the code falls back to the already ordered results.

The default rerank settings in [backend/config/settings.py](backend/config/settings.py) are:

- `RAG_TOP_N = 6`
- `RAG_RERANK_FLOOR = -11.0`

### Prompt building and answer generation

Prompt construction is in [backend/rag/prompts.py](backend/rag/prompts.py).

The system prompt instructs the model to answer from the provided context and to avoid unsupported claims. The prompt builder creates a context block from the retrieved chunks and places the question after that context.

Answer generation is performed by [backend/rag/providers/nvidia.py](backend/rag/providers/nvidia.py), using the configured chat model.

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
      "document_name": "sample.txt",
      "page_number": 1,
      "score": 0.123
    }
  ]
}
```

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

The chat ask endpoint uses Django session authentication and CSRF protection once a session exists. After logging in, make sure the client sends the current CSRF token.

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
