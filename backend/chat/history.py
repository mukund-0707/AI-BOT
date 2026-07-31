"""Conversation memory, stored in the Django session.

The browser already carries a conversation id: the `sessionid` cookie. A second
cookie would only be a second thing to keep in sync, so the id lives inside the
session, where it is still useful for logging and for "New chat" to rotate.

Messages are plain JSON-serialisable dicts, because that is what the session
backend can store.
"""

import uuid

from django.conf import settings

HISTORY_KEY = "history"
CONVERSATION_KEY = "conversation_id"

USER = "user"
ASSISTANT = "assistant"


def conversation_id(session):
    """Created on first use, kept until "New chat" rotates it."""

    if not session.get(CONVERSATION_KEY):
        session[CONVERSATION_KEY] = uuid.uuid4().hex
        session.modified = True

    return session[CONVERSATION_KEY]


def load(session):
    """The stored messages, oldest first."""

    history = session.get(HISTORY_KEY)

    return history if isinstance(history, list) else []


def append(session, question, answer):
    """Add one exchange and keep only the tail.

    The list is replaced rather than appended to: mutating the stored list in
    place leaves `session.modified` False, and Django then never saves it.
    """

    history = load(session) + [
        {"role": USER, "text": question},
        {"role": ASSISTANT, "text": answer},
    ]

    session[HISTORY_KEY] = history[-settings.RAG_HISTORY_STORE_LIMIT :]
    session.modified = True

    return session[HISTORY_KEY]


def reset(session):
    """Start a fresh conversation without touching the rest of the session.

    Deliberately not `session.flush()`: that would also sign out a logged-in
    admin who happened to click "New chat".
    """

    session[HISTORY_KEY] = []
    session[CONVERSATION_KEY] = uuid.uuid4().hex
    session.modified = True

    return session[CONVERSATION_KEY]
