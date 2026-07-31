"""Decide what the user wants before any retrieval runs.

Retrieval used to be the only path, so intent was inferred from its results: an
empty result set meant "not found". A greeting matches nothing either, so it got
the same answer. This module makes that decision up front instead, and returns
the two other things retrieval and generation need to get right - the language to
reply in, and an English query to search with.

Rules run first because they are free and cover the common traffic. The model is
asked only about what the rules could not place, and any failure there falls back
to the pre-existing behaviour: treat it as a question.
"""

import json
import logging
import re

from django.conf import settings

from rag.prompts import (
    INTENT_PROMPT,
    build_intent_prompt,
)

logger = logging.getLogger(__name__)

SMALL_TALK = "small_talk"
OVERVIEW = "overview"
KNOWLEDGE = "knowledge"

INTENTS = {SMALL_TALK, OVERVIEW, KNOWLEDGE}

EN = "en"
HINGLISH = "hinglish"
HI = "hi"

LANGUAGES = {EN, HINGLISH, HI}

DEVANAGARI = re.compile(r"[ऀ-ॿ] | in\s+hindi | in\s+devanagari | hindi | hindi\s+me\s+do", re.IGNORECASE | re.VERBOSE)

# Hindi function words (Latin script) used for Hinglish detection.
HINGLISH_MARKERS = re.compile(
    r"""\b(
        mujhe | mera | meri | mere | hamara | aap | aapka | tum
        | kya | kyu | kyun | kyon | kaise | kaisa | kaisi | kitna | kitne
        | kahan | kaun | kab | konsa | kaunsa
        | hai | hain | tha | thi | hoga | hogi | hota | hoti
        | karta | karti | karna | karne | kare | karo
        | nahi | nhi | naa | mat
        | chahiye | chaiye | batao | bataiye | bata | samjhao | samjha
        | dijiye | dedo | thoda | thodi | kuch | koi | sirf | bhi | wala | wali | hinglish\s+me
        | iska | iske | isme | uska | uske | usme | yeh | woh | vah
        | baare | bare | matlab | jankari | jaankari | jaanana | janna
        | acha | accha | theek | thik | bilkul | zaroori | zarurat | hinglish | in\s+hinglish 
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Match the whole message only; content queries still reach retrieval.
SMALL_TALK_PATTERN = re.compile(
    r"""^(
        (hi+|hey+|hello+|heya|yo+|hola|namaste|namaskar|howdy|greetings|sup)
            (\s+(there|again|mukii|bot|buddy|bhai))?
        | good\s+(morning|afternoon|evening|day)
        | how\s+(are|r)\s+(you|u|ya)(\s+doing)?
        | how('?s|\s+is)\s+it\s+going
        | what('?s|\s+is)\s+up
        | (aap\s+|tum\s+)?kaise\s+(ho|hain|hai)
        | kya\s+haal\s+(hai|he|h)
        | (tum|aap)\s+kaun\s+(ho|hain|hai)
        | who\s+(are|r)\s+(you|u)
        | what('?s|\s+is)\s+your\s+name
        | (tumhara|aapka)\s+naam\s+kya\s+(hai|he|h)
        | what\s+can\s+(you|u)\s+do(\s+for\s+me)?
        | (ok(ay)?\s+|great\s+|cool\s+)?(thanks|thank\s+you|thanx|thx|ty
            |shukriya|dhanyavad|dhanyawad)
            (\s+(so\s+much|a\s+lot|mukii|bhai))?
        | (bye|goodbye|good\s*night|see\s+(you|ya)|take\s+care|gn|alvida)
        | (ok(ay)?|hmm+|cool|nice|great|awesome|theek\s+hai|thik\s+hai)
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

# Whole-message only. Topic queries fall through to retrieval.
OVERVIEW_PATTERN = re.compile(
    r"""^(
        (please\s+|kindly\s+|can\s+you\s+|could\s+you\s+|pls\s+)?
        (
            (give|share|show|tell)\s+(me\s+|us\s+)?
                (some\s+|a\s+|the\s+|any\s+)?
                (brief\s+|short\s+|basic\s+|general\s+|quick\s+|overall\s+)?
                (context|overview|summary|brief|information|info|idea|intro
                    |introduction|details|background)
            | (what|whats|what's)\s+(is\s+)?(this|it|all\s+this|everything)\s+about
            | (what|whats|what's)\s+(is\s+)?(in\s+)?(here|this)
            | what\s+(can|do|should)\s+(i|we)\s+ask(\s+about)?
            | what\s+do\s+you\s+know(\s+about\s+(this|it))?
            | what\s+(is|are)\s+(the\s+)?(topics|contents|main\s+points|sections)
            | (brief|short)\s+(me|summary|overview|information|info)
            | summar(y|ise|ize)(\s+(this|it|everything|all))?
            | (context|overview|summary|brief|introduction)
            | (mujhe\s+|humein\s+|hume\s+)?
                (thoda|thodi|kuch|koi|puri|poori)?\s*
                (context|information|info|jankari|jaankari|detail|details
                    |overview|summary|idea)\s*
                (do|de\s*do|dedo|dijiye|chahiye|chaiye|bata\s*do|batao|bataiye)
            | (kuch|thoda|thodi)\s+(batao|bata\s*do|bataiye|samjhao)
            | (mujhe\s+)?(iske|is|iske)\s+baare\s+(mein|me)\s+
                (kuch|thoda|thodi)?\s*(batao|bata\s*do|bataiye|samjhao
                    |jankari\s+do)
            | (isme|is\s*me|isamen)\s+kya\s+kya?\s*(hai|he|h)
            | kya\s+kya\s+(hai|he|h)(\s+isme)?
            | (main|mai|hum)\s+kya\s+(puch|pooch)\s*(sakta|sakti|sakte)\s+
                (hoon|hun|hu|hain)
        )
        (\s+(please|pls|na|zara|thoda))?
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

# Strip decoration, not content: the model's own punctuation habits should not decide the route.
TRIM = " \t\n.!?,;:'\"“”‘’()-–—*"


def normalise(message):
    return re.sub(r"\s+", " ", str(message or "")).strip().strip(TRIM).strip()


def detect_language(message):
    """Script first, then Hindi function words. Content words are never enough."""

    if DEVANAGARI.search(message):
        return HI

    if HINGLISH_MARKERS.search(message):
        return HINGLISH

    return EN


def classify_by_rules(message):
    """Returns an intent, or None when the message needs a closer look."""

    cleaned = normalise(message)

    if not cleaned:
        return SMALL_TALK

    if SMALL_TALK_PATTERN.match(cleaned):
        return SMALL_TALK

    if OVERVIEW_PATTERN.match(cleaned):
        return OVERVIEW

    return None


def decision(intent, language, search_query="", source="rules"):
    return {
        "intent": intent,
        "language": language,
        "search_query": search_query,
        "source": source,
    }


def parse_model_decision(raw, message, language):
    """Read the classifier's JSON, distrusting every field.

    A bad label, a missing key or a stray sentence around the object all mean the
    same thing: fall back to treating the message as a question.
    """

    match = re.search(r"\{.*\}", str(raw or ""), re.DOTALL)

    if not match:
        raise ValueError(f"no JSON object in classifier output: {raw!r}")

    payload = json.loads(match.group(0))

    intent = str(payload.get("intent", "")).strip().lower()

    if intent not in INTENTS:
        raise ValueError(f"unknown intent: {intent!r}")

    model_language = str(payload.get("language", "")).strip().lower()
    # The rule-based detector wins on script, which it cannot get wrong.
    language = model_language if model_language in LANGUAGES else language

    search_query = str(payload.get("search_query") or "").strip()

    if intent != KNOWLEDGE:
        search_query = ""
    elif not search_query:
        search_query = message

    return decision(intent, language, search_query, source="model")


def classify_by_model(message, language, provider, history=None):

    raw = provider.classify(
        INTENT_PROMPT,
        build_intent_prompt(message, history),
        max_tokens=settings.RAG_INTENT_MAX_TOKENS,
    )

    return parse_model_decision(raw, message, language)


def classify(message, provider=None, history=None):
    """The single entry point. `provider=None` means rules only.

    `history` is what turns a follow-up into a searchable query - only the model
    stage can use it, since resolving "aur uske baad?" needs the conversation, not
    a pattern.

    Never raises: an unroutable message is a question, which is what the app did
    before this module existed.
    """

    cleaned = normalise(message)
    language = detect_language(cleaned)

    intent = classify_by_rules(cleaned)

    if intent is not None:
        query = cleaned if intent == KNOWLEDGE else ""
        return decision(intent, language, query)

    if provider is not None and settings.RAG_ENABLE_LLM_INTENT:
        try:
            return classify_by_model(cleaned, language, provider, history)
        except Exception as exc:
            logger.warning("Intent classification unavailable: %s", exc)

    return decision(KNOWLEDGE, language, cleaned, source="fallback")
