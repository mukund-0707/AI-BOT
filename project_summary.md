# Project Summary & Analysis: Mukii RAG Chatbot

## 1. Overall Project Architecture

This project is a Django-based backend for a Retrieval-Augmented Generation (RAG) chatbot named **Mukii**. It allows users to upload documents, processes them into searchable vectors, and answers user questions based on the document contents.

### Core Modules:
1. **`documents` (Document Ingestion Pipeline)**:
   - Responsible for uploading and extracting text from PDFs, DOCXs, and TXTs.
   - **`services.py -> DocumentService.process_document`**:
     - Extracts text from files using extractors (`extract_pdf`, `extract_docx`, etc.).
     - Cleans the text and chunks it using `rag.chunking.create_chunks`.
     - Generates vector embeddings for each chunk using `NVIDIAProvider`.
     - Upserts these embeddings and chunks into the Qdrant vector database.

2. **`rag` (Retrieval & Generation Pipeline)**:
   - Handles the core logic for answering questions.
   - **`providers/nvidia.py`**: Interacts with the NVIDIA API for embeddings and LLM chat completions.
   - **`providers/qdrant.py`**: Interacts with the Qdrant vector database for storing and searching chunks.
   - **`services.py -> answer_question`**: The main function that receives a query, vectorizes it, searches Qdrant for similar context chunks, and builds a prompt for the LLM to answer.
   - **`prompts.py`**: Contains the system instructions directing the LLM to behave strictly as an AI assistant that answers based on context.

3. **`chat` (API Endpoints)**:
   - Exposes HTTP endpoints.
   - **`views.py -> AskQuestionView`**: Receives user questions, calls `answer_question`, and returns the LLM response with relevant sources.

---

## 2. Why the Current Approach Fails on "Summarize" Queries

In your `chat/views.py`, you pointed out that the chatbot struggles when a user says something like *"give me a brief summary"*. 

**The Root Cause:**
The RAG pipeline strictly relies on **Vector Similarity Search**. When the user asks *"Give me a brief summary"*, the system generates an embedding for this phrase and compares it to the document chunks in Qdrant. Since document chunks contain factual information (not meta-phrases about summaries), the similarity score is very low (usually below your `RAG_MIN_SCORE` of 0.70). Therefore, the system concludes that no relevant chunks were found and returns: *"I couldn't find information about this in the knowledge base."*

---

## 3. Recommended Approaches to Make the Chatbot Smarter

To solve this, we need to bypass or augment the standard semantic search for specific intents.

### Approach 1: Pre-calculate Document Summaries (Recommended)
When a document is uploaded, it is difficult to summarize it on the fly during a chat query because you can't load a 100-page document into the LLM context.
1. Modify `documents/models.py` to add a `summary` field to the `Document` model.
2. In `documents/services.py` -> `process_document`, after extracting the text, make an LLM call to generate a summary of the document (or use a map-reduce summarization strategy for long docs).
3. Save this summary in the database.
4. When a user asks for a summary, you can directly fetch the `summary` fields of the uploaded documents and feed them to the chat model instead of doing a Qdrant search.

### Approach 2: Query Intent Routing (LLM Router)
Before doing the Qdrant vector search in `answer_question`, use a lightweight LLM call (or simple keyword rules) to classify the user's intent.
- **Intent: `Greeting/Chit-chat`** -> Route directly to LLM, skip Qdrant.
- **Intent: `Summarization`** -> Fetch the pre-calculated summaries or pull random representative chunks across all documents, and ask the LLM to summarize them.
- **Intent: `Question Answering`** -> Execute the normal Qdrant vector search pipeline.

**Example Router Logic (`rag/services.py`):**
```python
def answer_question(question):
    # 1. Intent Classification
    intent = classify_intent(question) # Returns "SUMMARY", "GREETING", or "QA"
    
    if intent == "SUMMARY":
        # Fetch summaries instead of chunks
        context = get_all_document_summaries()
        return generate_summary_response(context, question)
        
    elif intent == "GREETING":
        return {"answer": nvidia.simple_chat(question), "sources": []}
        
    # 2. Standard QA Pipeline (Existing code)
    question_embedding = nvidia.generate_embedding(question)
    results = qdrant.search_chunks(question_embedding=question_embedding)
    # ... rest of the code ...
```

### Approach 3: Function Calling / Agentic Approach
Instead of a rigid pipeline, give the NVIDIA LLM access to "Tools".
- `search_documents(query: str)` -> Uses Qdrant.
- `get_document_summary(doc_id: int)` -> Fetches a summary.
Let the LLM decide which tool to call based on the user's input. This makes the bot highly adaptable and "smart".

---

## Summary of the Flow
1. **User uploads file** -> `DocumentService.upload()` -> `DocumentService.process_document()` -> Text is extracted, chunked, embedded, and stored in Qdrant.
2. **User asks question** -> `AskQuestionView` -> `answer_question()` -> Query is embedded -> Similar chunks are retrieved from Qdrant -> LLM gets context + query and generates an answer.
