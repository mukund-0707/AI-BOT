"""One owner for the reply language: how it is decided, and how it is stated.

The decision used to be spread across three modules that could each disagree -
a regex detector in `intents`, a model override beside it, and the same rule
written into every system prompt in `prompts`. Whenever the result was wrong the
fix went wherever the symptom showed, so the rule ended up stated four times and
the detector ended up losing to the classifier. An all-English thread answering
in Devanagari was the visible end of that.

Everything that decides or states the language now lives here. Routing uses
`detect_language` and `reconcile_language`; prompts use `LANGUAGE_RULE` and
`language_directive`; nothing else needs to know how the choice is made.
"""

import re

EN = "en"
HINGLISH = "hinglish"
HI = "hi"

LANGUAGES = {EN, HINGLISH, HI}

# Bare "hindi" is deliberately not here: naming the language is not writing in
# it, and "the hindi translation policy" is an English question about a
# document. Only an explicit request for Hindi output counts.
DEVANAGARI = re.compile(
    r"[ऀ-ॿ] | in\s+hindi | in\s+devanagari | hindi\s+(me|mein|m)\b",
    re.IGNORECASE | re.VERBOSE,
)

# Hindi function words (Latin script) that no English sentence contains, so one
# is enough to call the message Hinglish.
HINGLISH_MARKERS = re.compile(
    r"""\b(
        mujhe | mera | meri | hamara | aap | aapka | tum
        | kya | kyu | kyun | kyon | kaise | kaisa | kaisi | kitna | kitne
        | kahan | kaun | kab | konsa | kaunsa
        | hain | thi | hoga | hogi | hota | hoti
        | karta | karti | karna | karne | kare | karo
        | nahi | nhi | naa
        | chahiye | chaiye | batao | bataiye | samjhao | samjha
        | dijiye | dedo | thoda | thodi | kuch | sirf | bhi | wala | wali | hinglish\s+me
        | iska | iske | isme | uska | uske | usme | yeh | woh | vah
        | baare | matlab | jankari | jaankari | jaanana | janna
        | acha | accha | theek | thik | bilkul | zaroori | zarurat | hinglish | in\s+hinglish
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Hindi words that are also ordinary English words ("a mere formality", "the
# bare minimum", "a door mat", "a koi pond"). One of these says nothing, so two
# distinct ones are required before the reply switches language on the user.
AMBIGUOUS_MARKERS = re.compile(
    r"\b(mere|bare|mat|koi|hai|bata|tha)\b",
    re.IGNORECASE,
)

LANGUAGE_NAMES = {
    EN: "English",
    HINGLISH: (
        "Hinglish - Hindi words spelled with the English alphabet, the way the "
        "user typed them. Latin letters only, never Devanagari. Style example: "
        '"Leave policy ke hisaab se aapko 12 din milte hain. Aap manager se '
        'approval le kar apply kar sakte hain."'
    ),
    HI: 'Hindi, in Devanagari script. Style example: "नीति के अनुसार आपको 12 दिन मिलते हैं।"',
}

# Written once and interpolated into every system prompt. It used to be typed
# out separately in each of them, which let the wordings drift apart and left
# the model reconciling three near-identical rules.
LANGUAGE_RULE = """\
- The "Answer language" line at the end of the message decides which language
  and script you reply in. It overrides everything else - the language of the
  Context, the language of earlier turns, all of it. English means English only.
  Hinglish means Latin letters only, never Devanagari. Hindi means Devanagari.
- Translate your own sentences only. Names, technical terms, numbers, dates,
  amounts, code, headings and table values stay exactly as they are written."""


def detect_language(message):
    """Script first, then Hindi function words. Content words are never enough.

    An English question about an indexed document can easily contain one word
    that is also Hindi - "door mat", "koi pond", "the bare minimum" - and
    answering it in Hinglish is a worse failure than missing the Hinglish, so
    the ambiguous words only count when two of them appear together.
    """

    if DEVANAGARI.search(message):
        return HI

    if HINGLISH_MARKERS.search(message):
        return HINGLISH

    ambiguous = {word.lower() for word in AMBIGUOUS_MARKERS.findall(message)}

    if len(ambiguous) >= 2:
        return HINGLISH

    return EN


def reconcile_language(detected, model_language):
    """Script is the detector's call. Register is the model's.

    The two disagree often, and letting the model win outright is what turns an
    all-English thread into a Devanagari reply: it only has to answer "hi" once.

    Characters are not a judgement call. Devanagari in the message proves the
    user typed Hindi, and its absence proves just as firmly that they did not -
    so a model that says "hi" about a Latin-script message is choosing a script
    the user has never once used, and is overruled.

    English against Hinglish is the one call the model genuinely makes better,
    because that turns on word choice rather than on characters.
    """

    if model_language not in LANGUAGES:
        return detected

    if detected == HI or model_language == HI:
        return detected

    return model_language


def localised(table, language):
    """Pick the user's language out of a message table, falling back to English."""

    return table.get(language, table[EN])


def language_directive(language):
    """Stated as a directive, not left for the model to infer from the question.

    A one-line Hinglish question next to several thousand tokens of English
    context loses every time otherwise.
    """

    name = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES[EN])

    return (
        f"Answer language: {name}\n"
        "INTERNAL INSTRUCTION ONLY.\n"
        "Do NOT repeat, quote, paraphrase, acknowledge, or mention this line.\n"
        "Never write phrases like 'Answer language', 'Language', "
        "'Responding in English', 'Responding in Hindi', "
        "'Here is the answer in Hinglish', or similar.\n"
        "Start immediately with the answer.\n"
    )
