from django.conf import settings

from rag.prompts import (
    SYSTEM_PROMPT,
    build_prompt,
)

from rag.providers.nvidia import NVIDIAProvider
from rag.providers.qdrant import QdrantProvider

nvidia = NVIDIAProvider()
qdrant = QdrantProvider()


# def answer_question(document_id, question):

#     document = Document.objects.filter(
#         id=document_id,
#         status=Document.Status.READY,
#     ).first()
#     print(f"Document found: {document is not None}")
#     print("DOCUMENNT:", document)
#     if not document:
#         return {
#             "answer": "The document is not ready for question answering.",
#             "sources": [],
#         }

#     question_embedding = nvidia.generate_embedding(question)

#     results = qdrant.search_chunks(
#         question_embedding=question_embedding,
#         document_id=document.id,
#     )
#     print("Rsults:", results)

#     if not results:
#         print("No results found for the question.")
#         return {
#             "answer": "The answer was not found in the uploaded document.",
#             "sources": [],
#         }

#     threshold = getattr(settings, "RAG_MIN_SCORE", 0.40)

#     if results[0]["score"] < threshold:
#         return {
#             "answer": "The answer was not found in the uploaded document.",
#             "sources": [],
#         }

#     prompt = build_prompt(
#         question=question,
#         chunks=results,
#     )

#     answer = nvidia.generate_answer(
#         SYSTEM_PROMPT,
#         prompt,
#     )

#     sources = []

#     for chunk in results:

#         sources.append(
#             {
#                 "document_name": chunk["file_name"],
#                 "page_number": chunk["page_number"],
#                 "score": round(chunk["score"], 3),
#             }
#         )

#     return {
#         "answer": answer,
#         "sources": sources,
#     }


NOT_FOUND = "I couldn't find information about this in the knowledge base."


def answer_question(question):

    question_embedding = nvidia.generate_embedding(question)

    results = qdrant.search_chunks(
        question_embedding=question_embedding,
    )
    print("Results:", results)

    threshold = getattr(settings, "RAG_MIN_SCORE", 0.70)

    # Filter every chunk, not just the best one. Previously a single strong hit
    # dragged the whole low-scoring tail into the prompt, and the model padded
    # its answer with whatever that noise suggested.
    results = [chunk for chunk in results if chunk["score"] >= threshold]

    if not results:
        return {
            "answer": NOT_FOUND,
            "sources": [],
        }

    prompt = build_prompt(
        question=question,
        chunks=results,
    )

    answer = nvidia.generate_answer(
        SYSTEM_PROMPT,
        prompt,
    )

    return {
        "answer": answer,
        "sources": build_sources(answer, results),
    }


def build_sources(answer, results):
    """Citations only make sense when the answer actually used the documents."""

    lowered = answer.lower()

    # `startswith("hi")` would also catch "History...", so match whole words.
    first_word = lowered.split()[0].strip(".,!") if lowered.split() else ""
    is_greeting = first_word in {"hi", "hello", "hey"}

    if (
        is_greeting
        or "couldn't find information" in lowered
        or "which topic" in lowered
    ):
        return []

    sources = []
    seen_pages = set()

    for chunk in results:
        unique_key = f"{chunk['file_name']}_{chunk['page_number']}"

        if unique_key in seen_pages:
            continue

        seen_pages.add(unique_key)
        sources.append(
            {
                "document_name": chunk["file_name"],
                "page_number": chunk["page_number"],
                "score": round(chunk["score"], 3),
            }
        )

    return sources
