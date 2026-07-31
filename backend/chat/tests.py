from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse

from chat import history
from rag.providers.qdrant import VectorStoreUnavailable


class AskEndpointTest(TestCase):
    """The session is the conversation, so these go through the real client."""

    def setUp(self):
        answer_patch = patch("chat.views.answer_question")
        self.answer_question = answer_patch.start()
        self.addCleanup(answer_patch.stop)

        self.answer_question.return_value = {"answer": "**Answer**", "sources": []}

    def ask(self, question):
        return self.client.post(
            reverse("ask-question"),
            {"question": question},
            content_type="application/json",
        )

    def test_first_question_carries_no_history(self):
        self.ask("what is the leave policy")

        self.assertEqual(self.answer_question.call_args.kwargs["history"], [])

    def test_second_question_sees_the_first_exchange(self):
        self.ask("what is the leave policy")
        self.ask("aur carry forward ka?")

        self.assertEqual(
            self.answer_question.call_args.kwargs["history"],
            [
                {"role": "user", "text": "what is the leave policy"},
                {"role": "assistant", "text": "**Answer**"},
            ],
        )

    def test_history_grows_in_order(self):
        for question in ("first", "second", "third"):
            self.ask(question)

        stored = self.client.session[history.HISTORY_KEY]

        self.assertEqual(
            [entry["text"] for entry in stored],
            [
                "first",
                "**Answer**",
                "second",
                "**Answer**",
                "third",
                "**Answer**",
            ],
        )

    @override_settings(RAG_HISTORY_STORE_LIMIT=4)
    def test_oldest_messages_fall_out_of_the_store(self):
        for question in ("first", "second", "third"):
            self.ask(question)

        stored = self.client.session[history.HISTORY_KEY]

        self.assertEqual(len(stored), 4)
        self.assertEqual(stored[0]["text"], "second")

    def test_a_failed_answer_stores_nothing(self):
        self.answer_question.side_effect = VectorStoreUnavailable("qdrant is down")

        response = self.ask("what is the leave policy")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.session.get(history.HISTORY_KEY, []), [])

    def test_conversation_id_is_stable_across_questions(self):
        self.ask("first")
        first_id = self.client.session[history.CONVERSATION_KEY]

        self.ask("second")

        self.assertEqual(self.client.session[history.CONVERSATION_KEY], first_id)

    def test_empty_question_is_rejected(self):
        response = self.ask("   ")

        self.assertEqual(response.status_code, 400)
        self.answer_question.assert_not_called()


class ResetEndpointTest(TestCase):

    def setUp(self):
        answer_patch = patch("chat.views.answer_question")
        self.answer_question = answer_patch.start()
        self.addCleanup(answer_patch.stop)

        self.answer_question.return_value = {"answer": "**Answer**", "sources": []}

    def test_reset_clears_history_and_rotates_the_id(self):
        self.client.post(
            reverse("ask-question"),
            {"question": "what is the leave policy"},
            content_type="application/json",
        )
        old_id = self.client.session[history.CONVERSATION_KEY]

        response = self.client.post(reverse("new-chat"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session[history.HISTORY_KEY], [])
        self.assertNotEqual(self.client.session[history.CONVERSATION_KEY], old_id)
        self.assertEqual(response.json()["conversation_id"], self.client.session[history.CONVERSATION_KEY])

    def test_the_next_question_starts_clean(self):
        self.client.post(
            reverse("ask-question"),
            {"question": "what is the leave policy"},
            content_type="application/json",
        )
        self.client.post(reverse("new-chat"))
        self.client.post(
            reverse("ask-question"),
            {"question": "aur carry forward ka?"},
            content_type="application/json",
        )

        self.assertEqual(self.answer_question.call_args.kwargs["history"], [])


class SessionHistoryTest(TestCase):
    """The store helpers on their own, without a request."""

    def session(self):
        from django.contrib.sessions.backends.db import SessionStore

        return SessionStore()

    def test_append_marks_the_session_modified(self):
        session = self.session()
        session.modified = False

        history.append(session, "q", "a")

        self.assertTrue(session.modified)

    def test_load_survives_a_corrupt_value(self):
        session = self.session()
        session[history.HISTORY_KEY] = "not a list"

        self.assertEqual(history.load(session), [])

    def test_conversation_id_is_created_once(self):
        session = self.session()

        first = history.conversation_id(session)

        self.assertEqual(history.conversation_id(session), first)
