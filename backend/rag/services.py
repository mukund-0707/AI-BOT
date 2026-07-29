from django.conf import settings
from docx import document

from documents.models import Document

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


def answer_question(question):

    question_embedding = nvidia.generate_embedding(question)

    results = qdrant.search_chunks(
        question_embedding=question_embedding,
    )
    print("Results:", results)

    threshold = getattr(settings, "RAG_MIN_SCORE", 0.70)

    valid_chunks = [chunk for chunk in results if chunk["score"] >= threshold]

    prompt = build_prompt(
        question=question,
        chunks=valid_chunks,
    )

    answer = nvidia.generate_answer(
        SYSTEM_PROMPT,
        prompt,
    )

    # Smart Source Hiding
    ans_lower = answer.lower()
    is_greeting = ans_lower.startswith("hello") or ans_lower.startswith("hi")
    is_not_found = "couldn't find information" in ans_lower
    is_vague = "specify which topic" in ans_lower

    if is_greeting or is_not_found or is_vague:
        sources = []
    else:
        sources = []
        seen_pages = set()
        for chunk in valid_chunks:
            unique_key = f"{chunk['file_name']}_{chunk['page_number']}"
            if unique_key not in seen_pages:
                seen_pages.add(unique_key)
                sources.append(
                    {
                        "document_name": chunk["file_name"],
                        "page_number": chunk["page_number"],
                        "score": round(chunk["score"], 3),
                    }
                )

    return {
        "answer": answer,
        "sources": sources,
    }
