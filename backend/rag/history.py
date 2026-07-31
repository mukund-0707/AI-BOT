"""The read side of conversation memory: what the model actually sees.

Storage is the caller's problem (chat.history keeps the list in the session).
This module only decides how much of that list is worth spending prompt on, and
it lives in rag because both the classifier and the answer path need the same
window - and because rag must not depend on chat.
"""

USER = "user"
ASSISTANT = "assistant"


def clip(text, limit):
    """Long answers are cut, not dropped: the topic survives, the detail goes."""

    text = str(text or "").strip()

    if limit is None or len(text) <= limit:
        return text

    return text[:limit].rstrip() + " ..."


def total_chars(messages):
    return sum(len(message["text"]) for message in messages)


def window(history, limit, char_budget=None, message_chars=None):
    """The last `limit` messages, trimmed to a prompt budget.

    Three rules, in order:

    1. Take the tail. The list is oldest-first, so nothing has to be deleted for
       an old message to leave the window - it just falls off the slice.
    2. Never open on an assistant message. A slice can cut a pair in half and
       leave a reply whose question is outside the window; the model reads that
       as an answer to nothing and tends to repeat it.
    3. Fit the character budget, dropping whole pairs from the front, so rule 2
       keeps holding as the budget bites.
    """

    recent = [
        {"role": message["role"], "text": clip(message["text"], message_chars)}
        for message in (history or [])[-limit:]
        if message.get("text")
    ]

    if recent and recent[0]["role"] == ASSISTANT:
        recent = recent[1:]

    if char_budget is None:
        return recent

    while recent and total_chars(recent) > char_budget:
        recent = recent[1:]

        if recent and recent[0]["role"] == ASSISTANT:
            recent = recent[1:]

    return recent


def user_turns(history):
    """Only what the user said - the answer path must not see its own replies.

    Tested against the model: with its previous answers in the window it will
    repeat a fact from them even when the current Context does not contain it,
    and no amount of prompt wording stops it ("notice period phir se batao" ->
    the old number, verbatim). The previous questions are enough to resolve
    "aur carry forward ka?", and they carry no facts to copy.
    """

    return [message for message in history or [] if message["role"] == USER]


def as_text(history, separator="\n"):
    """Flattened for prompts that take one string, such as the classifier."""

    return separator.join(
        f"{message['role']}: {message['text']}" for message in history or []
    )
