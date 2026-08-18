import logging

from django.conf import settings

from documents.models import Document

from rag import history as history_window
from rag import intents
from rag.language import localised
from rag.prompts import (
    EMPTY_CORPUS,
    NOT_FOUND,
    NO_ANSWER,
    OVERVIEW_PROMPT,
    SMALL_TALK_FALLBACK,
    SMALL_TALK_PROMPT,
    SYSTEM_PROMPT,
    build_overview_prompt,
    build_prompt,
    build_small_talk_prompt,
)

from rag.providers.nvidia import NVIDIAProvider
from rag.providers.qdrant import QdrantProvider

logger = logging.getLogger(__name__)

nvidia = NVIDIAProvider()
qdrant = QdrantProvider()


def recent_messages(history):
    """The slice of the conversation worth spending prompt on."""

    return history_window.window(
        history,
        limit=settings.RAG_HISTORY_LIMIT,
        char_budget=settings.RAG_HISTORY_CHAR_BUDGET,
        message_chars=settings.RAG_HISTORY_MESSAGE_CHARS,
    )


def deduplicate_chunks(chunks):
    """Collapse duplicate passages before the reranker consumes the budget."""

    deduped = []
    seen = set()

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        key = " ".join(text.split()).lower()
        if key in seen:
            continue

        seen.add(key)
        deduped.append(chunk)

    return deduped


def rerank(question, results):
    """The reranker is the relevance gate; vector order is only a safety net."""

    try:
        return nvidia.rerank(
            question,
            results,
            top_n=settings.RAG_TOP_N,
            floor=settings.RAG_RERANK_FLOOR,
        )
    except Exception as exc:
        logger.warning("Reranking unavailable, keeping fused order: %s", exc)
        return results[: settings.RAG_TOP_N]


def answer_question(question, history=None):
    """Route first, retrieve second.

    Small talk and "give me context" match no passage, so anything that decides
    from the result set alone answers both with "not found". The intent is
    settled before a single vector is fetched.

    `history` is the conversation so far, oldest first. It is what lets a
    follow-up ("aur carry forward ka?") be rewritten into something retrievable.
    """

    messages = recent_messages(history)
    logger.debug("Carrying %d messages of history", len(messages))

    decision = intents.classify(question, nvidia, history=messages)
    logger.debug("Routed %r as %s", question, decision)

    if decision["intent"] == intents.SMALL_TALK:
        return small_talk_answer(question, decision["language"], messages)

    if decision["intent"] == intents.OVERVIEW:
        # An orientation answer has nothing to follow up on.
        return overview_answer(decision["language"])

    return knowledge_answer(question, decision, messages)


def small_talk_answer(question, language, history=None):
    """No retrieval and no citations - there is nothing here to ground."""

    try:
        answer = nvidia.generate_answer(
            SMALL_TALK_PROMPT,
            build_small_talk_prompt(question, language),
            history=history,
        )
    except Exception as exc:
        # A greeting is not worth a 500.
        logger.warning("Small talk generation failed: %s", exc)
        answer = ""

    return {
        "answer": answer or localised(SMALL_TALK_FALLBACK, language),
        "sources": [],
    }


def overview_answer(language):
    """Orientation for a user who does not have a specific question yet."""

    document_ids = list(
        Document.objects.filter(status=Document.Status.READY)
        .order_by("-created_at")
        .values_list("id", flat=True)[: settings.RAG_OVERVIEW_MAX_DOCUMENTS]
    )

    chunks = qdrant.sample_chunks(
        document_ids,
        settings.RAG_OVERVIEW_DOC_CHUNKS,
    )
    logger.debug("Overview sampled %d chunks", len(chunks))

    if not chunks:
        return {
            "answer": localised(EMPTY_CORPUS, language),
            "sources": [],
        }

    answer = nvidia.generate_answer(
        OVERVIEW_PROMPT,
        build_overview_prompt(chunks, language),
    )

    # An overview invites a question, it does not make a cited claim, so the
    # sampled pages would be misleading as citations.
    return {
        "answer": answer or localised(EMPTY_CORPUS, language),
        "sources": [],
    }


def retrieve(question, decision):
    """Everything between a routed question and the prompt that answers it.

    Both answer paths need exactly this, and the two copies had already started
    to drift - the streaming one grew its own not-found handling while the other
    went through `is_no_answer`. One copy means a retrieval change lands on both
    at once.

    Returns `(results, prompt)`. Empty results mean nothing cleared the
    reranker; the caller decides what to say about that, because the streaming
    path has to say it differently.
    """

    search_query = decision["search_query"] or question

    question_embedding = nvidia.generate_embedding(search_query)

    # Dense search uses the rewritten English query; sparse search uses both queries.
    keywords = (
        search_query if search_query == question else f"{search_query} {question}"
    )

    results = qdrant.search_chunks(
        question_embedding=question_embedding,
        question=keywords,
    )
    logger.debug("Retrieved %d candidates", len(results))

    results = deduplicate_chunks(results)
    logger.debug("Kept %d unique chunks after deduplication", len(results))

    results = rerank(search_query, results)
    logger.debug("Kept %d chunks after reranking", len(results))

    if not results:
        return [], None

    chunks = qdrant.with_neighbours(results)
    logger.debug("Expanded to %d chunks with neighbours", len(chunks))

    prompt = build_prompt(
        question=question,
        chunks=chunks,
        language=decision["language"],
    )

    return results, prompt


def knowledge_answer(question, decision, history=None):

    language = decision["language"]

    results, prompt = retrieve(question, decision)

    if not results:
        return not_found(language)

    answer = nvidia.generate_answer(
        SYSTEM_PROMPT,
        prompt,
        history=history_window.user_turns(history),
    )

    if is_no_answer(answer):
        return not_found(language)

    return {
        "answer": answer,
        "sources": build_sources(results),
    }


def stream_answer_question(question, history=None):
    """Same routing and retrieval as answer_question, but streams the answer.

    Yields a sequence of strings:
      - token strings as the LLM produces them
      - a single final dict  {"sources": [...], "answer": "<full assembled text>"}
        so the caller can store the completed answer in history.

    Small talk and overview fall back to the non-streaming path and yield the
    full answer as one token followed by the sources dict.
    """

    messages = recent_messages(history)
    decision = intents.classify(question, nvidia, history=messages)

    # Small talk and overview do not stream — they are short and fast already.
    if decision["intent"] == intents.SMALL_TALK:
        result = small_talk_answer(question, decision["language"], messages)
        yield result["answer"]
        yield {"sources": result["sources"], "answer": result["answer"]}
        return

    if decision["intent"] == intents.OVERVIEW:
        result = overview_answer(decision["language"])
        yield result["answer"]
        yield {"sources": result["sources"], "answer": result["answer"]}
        return

    language = decision["language"]

    results, prompt = retrieve(question, decision)

    if not results:
        not_found_answer = localised(NOT_FOUND, language)
        yield not_found_answer
        yield {"sources": [], "answer": not_found_answer}
        return

    assembled = []

    for token in nvidia.stream_answer(
        SYSTEM_PROMPT,
        prompt,
        history=history_window.user_turns(messages),
    ):
        assembled.append(token)
        yield token

    full_answer = "".join(assembled).strip()

    if is_no_answer(full_answer):
        not_found_answer = localised(NOT_FOUND, language)
        yield {"sources": [], "answer": not_found_answer, "replace": True}
        return

    yield {"sources": build_sources(results), "answer": full_answer}


def is_no_answer(answer):
    """The model refuses with a sentinel, so this check survives translation.

    Matching on the refusal sentence itself only worked while every answer was
    English.
    """

    stripped = (answer or "").strip()

    return not stripped or NO_ANSWER in stripped.upper()


def not_found(language):
    return {
        "answer": localised(NOT_FOUND, language),
        "sources": [],
    }


def build_sources(results):
    """One citation per document page, in reranked order."""

    sources = []
    seen_pages = set()

    for chunk in results:
        unique_key = f"{chunk['file_name']}_{chunk['page_number']}"

        if unique_key in seen_pages:
            continue

        seen_pages.add(unique_key)

        # Chunks pulled in by position rather than by score have no score.
        score = chunk.get("score")

        sources.append(
            {
                "document_name": chunk["file_name"],
                "page_number": chunk["page_number"],
                "score": round(score, 3) if score is not None else None,
            }
        )

    return sources
