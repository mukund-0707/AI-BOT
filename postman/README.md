# Postman collection - Mukii RAG Chatbot

## Import

In Postman: **Import** -> drop both files:

- `Mukii-RAG.postman_collection.json`
- `Mukii-RAG.postman_environment.json`

Then pick **Mukii RAG - Local** in the environment dropdown (top right) and set
`password` to your Django user's password. `username` defaults to `mukund`.

## Before running

```bash
docker start qdrant                  # vector DB on :6333
cd backend && python manage.py runserver
```

If the chat endpoint returns 503, Qdrant is down - the **4. Infra - Qdrant**
folder tells you which of the two is broken.

## Run order

The requests share state through collection variables, so order matters:

| # | Request | What it does |
|---|---------|--------------|
| 1 | Get CSRF Cookie | stores `csrftoken` |
| 2 | Login | sets `sessionid`, re-stores the rotated `csrftoken` |
| 3 | Upload Document | stores `document_id` |
| 4 | Ask Question | the actual RAG call |
| 5 | Delete Document | cleans up the upload |

The Collection Runner works on folders 1 -> 4 in order. Three requests are
marked as manual and will fail in an unattended run - see below.

## The CSRF gotcha

Two different auth classes are in play:

- `AskQuestionView` uses the project default `SessionAuthentication`
  (`config/settings.py`), which **enforces CSRF once a session cookie exists**.
  So `/api/chat/ask/` needs the `X-CSRFToken` header after you log in - but
  works fine without it while logged out.
- `UploadDocumentView` / `DeleteDocumentView` use
  `CsrfExemptSessionAuthentication`, so they never need the header.

Django also **rotates the CSRF token during login**, which is why the Login
request re-captures it. If you ever get a surprise `403` on Ask, re-run Login.

## Manual-only requests

These are deliberate failure cases; they do not pass in a plain top-to-bottom run:

| Request | Setup needed |
|---------|--------------|
| `Upload - not logged in (expect 403)` | clear cookies for the host first, then re-run Login |
| `Ask - missing CSRF token (expect 403)` | must be logged in |
| `Ask - Qdrant down (expect 503)` | `docker stop qdrant`, send, then `docker start qdrant` |

## Fixtures

`fixtures/sample.txt` is a valid upload; `fixtures/unsupported.csv` triggers the
400 file-type rejection. If Postman cannot resolve the relative path, click
**Select Files** on the request body and pick them manually.

Note that `Upload Document` runs the whole ingest pipeline synchronously
(extract -> chunk -> NVIDIA embeddings -> Qdrant upsert), so it is slow for
large files and it does consume NVIDIA API quota.

## Retrieval probe

`Ask - section lookup (retrieval probe)` is not a pass/fail test - it exercises
an exact-section question and logs to the Postman console whether the section
was retrieved and whether the answer still reports the text as truncated. Use it
to check whether chunking or text-cleaning changes actually helped.
