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


class StreamAskEndpointTest(TestCase):
    """The path the browser actually uses, so it needs the same guarantees.

    Everything here answers one question: does the turn survive the response?
    The answer is produced inside a generator, which the WSGI server drains
    after the middleware chain has finished, so nothing the generator writes to
    the session is saved unless the view saves it itself.
    """

    def setUp(self):
        self.seen_history = []

        def fake_stream(question, history=None):
            self.seen_history.append(list(history or []))
            yield "**Answer**"
            yield {"sources": [], "answer": "**Answer**"}

        stream_patch = patch(
            "chat.views.stream_answer_question",
            side_effect=fake_stream,
        )
        self.stream_answer_question = stream_patch.start()
        self.addCleanup(stream_patch.stop)

    def ask(self, question):
        response = self.client.post(
            reverse("ask-question-stream"),
            {"question": question},
            content_type="application/json",
        )

        # The session is only written while the body is produced, so a test
        # that never reads the body is testing a request that never finished.
        if response.status_code == 200:
            response.body = b"".join(response.streaming_content)

        return response

    def test_first_question_carries_no_history(self):
        self.ask("what is the leave policy")

        self.assertEqual(self.seen_history[0], [])

    def test_second_question_sees_the_first_exchange(self):
        self.ask("what is the leave policy")
        self.ask("aur carry forward ka?")

        self.assertEqual(
            self.seen_history[1],
            [
                {"role": "user", "text": "what is the leave policy"},
                {"role": "assistant", "text": "**Answer**"},
            ],
        )

    def test_the_exchange_is_persisted_not_just_passed_along(self):
        self.ask("what is the leave policy")

        stored = self.client.session[history.HISTORY_KEY]

        self.assertEqual(
            [entry["text"] for entry in stored],
            ["what is the leave policy", "**Answer**"],
        )

    def test_tokens_and_sources_reach_the_client(self):
        response = self.ask("what is the leave policy")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"token": "**Answer**"', response.body)
        self.assertIn(b'"done": true', response.body)

    def test_a_failed_answer_stores_nothing(self):
        self.stream_answer_question.side_effect = VectorStoreUnavailable(
            "qdrant is down"
        )

        response = self.ask("what is the leave policy")

        self.assertIn(b"qdrant is down", response.body)
        self.assertEqual(self.client.session.get(history.HISTORY_KEY, []), [])

    def test_empty_question_is_rejected(self):
        response = self.ask("   ")

        self.assertEqual(response.status_code, 400)
        self.stream_answer_question.assert_not_called()

    def test_an_unexpected_failure_still_ends_the_stream(self):
        """The status line is already sent, so this cannot become a 500.

        Without a terminal event the browser sits with its composer disabled,
        waiting for one that never arrives, and only a reload frees it.
        """

        def explode(question, history=None):
            yield "partial "
            raise RuntimeError("nvidia rate limited")

        self.stream_answer_question.side_effect = explode

        response = self.ask("what is the leave policy")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"error"', response.body)
        self.assertNotIn(b"nvidia rate limited", response.body)

    def test_a_crashed_answer_stores_nothing(self):
        def explode(question, history=None):
            yield "partial "
            raise RuntimeError("nvidia rate limited")

        self.stream_answer_question.side_effect = explode

        self.ask("what is the leave policy")

        self.assertEqual(self.client.session.get(history.HISTORY_KEY, []), [])


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
        self.assertEqual(
            response.json()["conversation_id"],
            self.client.session[history.CONVERSATION_KEY],
        )

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
