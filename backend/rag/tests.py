from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.test import TestCase, override_settings

from documents.models import Document

from rag import history, intents, services
from rag.prompts import EMPTY_CORPUS, NOT_FOUND
from rag.providers.nvidia import NVIDIAProvider, ReasoningFilter


def chunk(text="Leave is accrued monthly.", page=2, index=4, score=-2.5):
    return {
        "score": score,
        "document_id": 1,
        "file_name": "hr-policy.pdf",
        "page_number": page,
        "chunk_index": index,
        "text": text,
    }


class NVIDIAProviderEmbeddingTest(TestCase):
    def test_queries_use_query_input_type(self):
        provider = NVIDIAProvider()
        provider.client = MagicMock()
        provider.client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )

        provider.generate_embedding("who is the owner")

        provider.client.embeddings.create.assert_called_once_with(
            model=settings.NVIDIA_EMBEDDING_MODEL,
            input="who is the owner",
            extra_body={"input_type": "query"},
        )

    def test_passages_use_passage_input_type(self):
        provider = NVIDIAProvider()
        provider.client = MagicMock()
        provider.client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.3, 0.4])]
        )

        provider.generate_embeddings_batch(["policy text", "contact details"])

        provider.client.embeddings.create.assert_called_once_with(
            model=settings.NVIDIA_EMBEDDING_MODEL,
            input=["policy text", "contact details"],
            extra_body={"input_type": "passage"},
        )


class ReasoningFilterTest(TestCase):
    """A streamed tag arrives in pieces, so per-delta matching is not enough.

    The model emits "<th" and then "ink>". Searching each delta on its own
    never finds the tag, and the whole reasoning trace reaches both the screen
    and - because the caller assembles the answer from what is yielded - the
    stored conversation.
    """

    def filtered(self, deltas):
        reasoning = ReasoningFilter()

        return "".join(reasoning.feed(delta) for delta in deltas) + reasoning.flush()

    def test_plain_text_passes_through(self):
        self.assertEqual(self.filtered(["Leave is ", "12 days."]), "Leave is 12 days.")

    def test_a_whole_tag_in_one_delta(self):
        self.assertEqual(
            self.filtered(["<think>weighing it up</think>", "Real answer."]),
            "Real answer.",
        )

    def test_a_tag_split_across_deltas(self):
        self.assertEqual(
            self.filtered(
                ["<th", "ink>", "weighing ", "it up", "</th", "ink>", "Real answer."]
            ),
            "Real answer.",
        )

    def test_one_character_at_a_time(self):
        source = "before<think>hidden</think>after"

        self.assertEqual(self.filtered(list(source)), "beforeafter")

    def test_text_around_the_tag_survives(self):
        self.assertEqual(
            self.filtered(["Answer: <think>hm</think> 12 days."]),
            "Answer:  12 days.",
        )

    def test_an_unclosed_tag_drops_the_trace(self):
        """A trace the model never came out of is not an answer."""

        self.assertEqual(
            self.filtered(["Partial ", "<think>", "still going"]), "Partial "
        )

    def test_a_lone_angle_bracket_is_not_swallowed(self):
        self.assertEqual(self.filtered(["a < b"]), "a < b")

    def test_a_trailing_partial_tag_is_released_on_flush(self):
        self.assertEqual(self.filtered(["done <th"]), "done <th")


class ChunkDeduplicationTest(TestCase):
    def test_duplicate_chunks_are_collapsed(self):
        chunks = [
            {"document_id": 1, "page_number": 2, "chunk_index": 0, "text": "alpha"},
            {"document_id": 1, "page_number": 2, "chunk_index": 0, "text": "alpha"},
            {"document_id": 1, "page_number": 3, "chunk_index": 1, "text": "beta"},
        ]

        deduped = services.deduplicate_chunks(chunks)

        self.assertEqual(
            deduped,
            [
                {"document_id": 1, "page_number": 2, "chunk_index": 0, "text": "alpha"},
                {"document_id": 1, "page_number": 3, "chunk_index": 1, "text": "beta"},
            ],
        )


class ClassifyRulesTest(TestCase):
    """The rules run before any model call, so they are checked on their own."""

    def classify(self, message):
        # No provider: rules only, no network.
        return intents.classify(message, provider=None)

    def test_small_talk(self):
        messages = [
            "hi",
            "Hi!",
            "hii",
            "Hello there",
            "Heyyy",
            "good morning",
            "how are you",
            "how r u",
            "what's up",
            "who are you",
            "What is your name",
            "what can you do",
            "thanks",
            "Thank you so much",
            "thx",
            "bye",
            "Good night",
            "take care",
            "namaste",
            "aap kaise ho",
            "kya haal hai",
            "shukriya",
            "theek hai",
            "ok",
        ]

        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(
                    self.classify(message)["intent"],
                    intents.SMALL_TALK,
                )

    def test_overview(self):
        messages = [
            "give me context",
            "Give me some brief information",
            "give me a summary",
            "mujhe context do",
            "mujhe thodi information do",
            "kuch batao",
            "iske baare mein kuch batao",
            "isme kya kya hai",
            "what is this about",
            "what can i ask",
            "summarise this",
            "overview",
            "context",
        ]

        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(
                    self.classify(message)["intent"],
                    intents.OVERVIEW,
                )

    def test_questions_are_never_short_circuited(self):
        """The regression that would hurt most: a real question routed away."""

        messages = [
            "hi, what is the leave policy",
            "hello what does section 3 say",
            "give me the details of the leave policy",
            "brief information about the notice period",
            "mujhe leave policy ki information chahiye",
            "how are you handling refunds",
            "thank you note format",
            "what can you do to reset my password",
            "history of the company",
            "who won the 2019 world cup",
            "yeh policy kaise kaam karti hai",
            "यह कैसे काम करता है",
        ]

        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(
                    self.classify(message)["intent"],
                    intents.KNOWLEDGE,
                )

    def test_language_detection(self):
        cases = {
            "what is the leave policy": intents.EN,
            "How does the refund process work?": intents.EN,
            "mujhe leave policy chahiye": intents.HINGLISH,
            "yeh kaise kaam karta hai": intents.HINGLISH,
            "छुट्टी की नीति क्या है": intents.HI,
            "यह कैसे काम करता है": intents.HI,
        }

        for message, language in cases.items():
            with self.subTest(message=message):
                self.assertEqual(self.classify(message)["language"], language)

    def test_what_the_user_told_you_is_not_a_retrieval_question(self):
        """The documents cannot answer these, so search only says "not found".

        Both spellings appear because the same user switches between them
        mid-thread - "mera naam" one turn, "mera name" the next.
        """

        messages = [
            "my name is Riya",
            "I am Riya",
            "mera naam Riya hai",
            "mujhe Riya bolte hain",
            "what is my name",
            "What is My name?",
            "whats my name",
            "my name?",
            "tell me my name",
            "do you remember my name",
            "Mera name kya he",
            "mera naam kya hai",
            "mujhe mera naam batao",
            "tumhe mera name yaad hai",
        ]

        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(
                    self.classify(message)["intent"],
                    intents.SMALL_TALK,
                )

    def test_a_document_question_that_mentions_a_name_still_retrieves(self):
        """The guard on the rule above: "name" is also a document word."""

        messages = [
            "my name in the employee register",
            "what is the naming convention",
            "I am looking for the notice period",
            "tell me my leave balance",
            "mera leave kitna hai",
        ]

        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(
                    self.classify(message)["intent"],
                    intents.KNOWLEDGE,
                )

    def test_english_questions_are_not_flipped_by_one_ambiguous_word(self):
        """Every one of these is a plain English question about a document.

        Each contains a word that is also Hindi, which used to be enough on its
        own - and replying to an English question in Hinglish is a louder
        failure than missing a Hinglish one.
        """

        messages = (
            "what is a door mat",
            "the koi pond maintenance schedule",
            "Hai Corporation revenue",
            "show me the bare minimum requirements",
            "this is a mere formality",
        )

        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(self.classify(message)["language"], intents.EN)

    def test_two_ambiguous_words_together_are_still_hinglish(self):
        self.assertEqual(
            self.classify("mere paas koi jankari nahi")["language"],
            intents.HINGLISH,
        )

    def test_naming_hindi_is_not_writing_in_it(self):
        self.assertEqual(
            self.classify("what is the hindi translation policy")["language"],
            intents.EN,
        )

    def test_asking_for_hindi_output_still_switches(self):
        for message in ("answer in hindi", "hindi me batao"):
            with self.subTest(message=message):
                self.assertEqual(self.classify(message)["language"], intents.HI)

    def test_knowledge_search_query_defaults_to_the_question(self):
        decision = self.classify("what is the notice period")

        self.assertEqual(decision["search_query"], "what is the notice period")

    def test_no_search_query_for_routed_intents(self):
        for message in ("hi", "give me context"):
            with self.subTest(message=message):
                self.assertEqual(self.classify(message)["search_query"], "")

    def test_empty_message_is_not_a_question(self):
        self.assertEqual(self.classify("   ")["intent"], intents.SMALL_TALK)


class ClassifyModelTest(TestCase):
    """Everything the classifier returns is distrusted."""

    def provider(self, raw):
        provider = MagicMock()
        provider.classify.return_value = raw

        return provider

    def test_parses_a_clean_decision(self):
        provider = self.provider(
            '{"intent": "knowledge", "language": "hinglish",'
            ' "search_query": "leave policy process"}'
        )

        decision = intents.classify(
            "Mujhe leave policy ka process chahiye",
            provider,
        )

        self.assertEqual(decision["intent"], intents.KNOWLEDGE)
        self.assertEqual(decision["language"], intents.HINGLISH)
        self.assertEqual(decision["search_query"], "leave policy process")
        self.assertEqual(decision["source"], "model")

    def test_tolerates_prose_around_the_json(self):
        provider = self.provider(
            'Sure! {"intent": "overview", "language": "en", "search_query": ""}'
        )

        decision = intents.classify("kuch to batao yaar", provider)

        self.assertEqual(decision["intent"], intents.OVERVIEW)

    def test_bad_json_falls_back_to_knowledge(self):
        provider = self.provider("small_talk, probably")

        decision = intents.classify("kuch to batao yaar", provider)

        self.assertEqual(decision["intent"], intents.KNOWLEDGE)
        self.assertEqual(decision["source"], "fallback")
        self.assertEqual(decision["search_query"], "kuch to batao yaar")

    def test_unknown_intent_falls_back_to_knowledge(self):
        provider = self.provider(
            '{"intent": "chitchat", "language": "en", "search_query": ""}'
        )

        self.assertEqual(
            intents.classify("kuch to batao yaar", provider)["intent"],
            intents.KNOWLEDGE,
        )

    def test_provider_failure_falls_back_to_knowledge(self):
        provider = MagicMock()
        provider.classify.side_effect = RuntimeError("timeout")

        decision = intents.classify("kuch to batao yaar", provider)

        self.assertEqual(decision["intent"], intents.KNOWLEDGE)
        self.assertEqual(decision["source"], "fallback")

    def test_the_model_cannot_switch_a_latin_message_to_devanagari(self):
        """The bug this guards: one "hi" from the classifier, and an English
        thread answers in Devanagari.

        No Devanagari in the message is proof the user did not type Hindi, so
        the model does not get to overrule it.
        """

        for detected in (intents.EN, intents.HINGLISH):
            with self.subTest(detected=detected):
                self.assertEqual(
                    intents.reconcile_language(detected, intents.HI),
                    detected,
                )

    def test_devanagari_in_the_message_settles_it(self):
        for claimed in (intents.EN, intents.HINGLISH):
            with self.subTest(claimed=claimed):
                self.assertEqual(
                    intents.reconcile_language(intents.HI, claimed),
                    intents.HI,
                )

    def test_the_model_still_decides_english_against_hinglish(self):
        """The one call it makes better - word choice, not characters."""

        self.assertEqual(
            intents.reconcile_language(intents.EN, intents.HINGLISH),
            intents.HINGLISH,
        )
        self.assertEqual(
            intents.reconcile_language(intents.HINGLISH, intents.EN),
            intents.EN,
        )

    def test_unknown_language_keeps_the_detected_one(self):
        provider = self.provider(
            '{"intent": "knowledge", "language": "martian", "search_query": "x"}'
        )

        decision = intents.classify("yeh kaise kaam karta hai", provider)

        self.assertEqual(decision["language"], intents.HINGLISH)

    def test_search_query_is_dropped_for_routed_intents(self):
        provider = self.provider(
            '{"intent": "small_talk", "language": "en", "search_query": "hello"}'
        )

        self.assertEqual(intents.classify("yo yo", provider)["search_query"], "")

    def test_rules_win_before_the_model_is_asked(self):
        provider = self.provider(
            '{"intent": "knowledge", "language": "en", "search_query": "hi"}'
        )

        decision = intents.classify("hi", provider)

        self.assertEqual(decision["intent"], intents.SMALL_TALK)
        provider.classify.assert_not_called()

    @override_settings(RAG_ENABLE_LLM_INTENT=False)
    def test_kill_switch_skips_the_model(self):
        provider = self.provider(
            '{"intent": "overview", "language": "en", "search_query": ""}'
        )

        decision = intents.classify("kuch to batao yaar", provider)

        self.assertEqual(decision["intent"], intents.KNOWLEDGE)
        provider.classify.assert_not_called()


@override_settings(RAG_ENABLE_LLM_INTENT=False)
class RoutingTest(TestCase):
    """Rules-only routing, so every case here is deterministic."""

    def setUp(self):
        nvidia_patch = patch.object(services, "nvidia")
        qdrant_patch = patch.object(services, "qdrant")

        self.nvidia = nvidia_patch.start()
        self.qdrant = qdrant_patch.start()

        self.addCleanup(nvidia_patch.stop)
        self.addCleanup(qdrant_patch.stop)

        self.nvidia.generate_embedding.return_value = [0.1, 0.2]
        self.nvidia.generate_answer.return_value = "**Answer**"
        self.nvidia.rerank.return_value = [chunk()]
        self.qdrant.search_chunks.return_value = [chunk()]
        self.qdrant.with_neighbours.return_value = [chunk()]

    def test_small_talk_never_touches_retrieval(self):
        result = services.answer_question("hi")

        self.assertEqual(result["answer"], "**Answer**")
        self.assertEqual(result["sources"], [])
        self.nvidia.generate_embedding.assert_not_called()
        self.qdrant.search_chunks.assert_not_called()
        self.nvidia.rerank.assert_not_called()

    def test_small_talk_survives_a_provider_outage(self):
        self.nvidia.generate_answer.side_effect = RuntimeError("502")

        result = services.answer_question("hello")

        self.assertIn("Hi!", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_small_talk_answers_in_the_users_language(self):
        self.nvidia.generate_answer.side_effect = RuntimeError("502")

        result = services.answer_question("aap kaise ho")

        self.assertIn("help kar sakta hoon", result["answer"])

    def test_overview_samples_documents_instead_of_searching(self):
        Document.objects.create(
            file="documents/hr-policy.pdf",
            original_name="hr-policy.pdf",
            file_type="pdf",
            file_size=1024,
            status=Document.Status.READY,
        )
        self.qdrant.sample_chunks.return_value = [chunk(score=None)]

        result = services.answer_question("give me context")

        self.assertEqual(result["answer"], "**Answer**")
        # An overview invites a question; it makes no cited claim.
        self.assertEqual(result["sources"], [])
        self.qdrant.search_chunks.assert_not_called()
        self.qdrant.sample_chunks.assert_called_once()

    def test_overview_only_samples_ready_documents(self):
        Document.objects.create(
            file="documents/pending.pdf",
            original_name="pending.pdf",
            file_type="pdf",
            file_size=1024,
            status=Document.Status.PROCESSING,
        )
        self.qdrant.sample_chunks.return_value = []

        services.answer_question("give me context")

        document_ids, _ = self.qdrant.sample_chunks.call_args[0]
        self.assertEqual(list(document_ids), [])

    def test_overview_with_an_empty_index(self):
        self.qdrant.sample_chunks.return_value = []

        result = services.answer_question("mujhe context do")

        self.assertEqual(result["answer"], EMPTY_CORPUS[intents.HINGLISH])
        self.nvidia.generate_answer.assert_not_called()

    def test_knowledge_answer_carries_sources(self):
        result = services.answer_question("what is the leave policy")

        self.assertEqual(result["answer"], "**Answer**")
        self.assertEqual(
            result["sources"],
            [
                {
                    "document_name": "hr-policy.pdf",
                    "page_number": 2,
                    "score": -2.5,
                }
            ],
        )

    def test_deduplication_happens_before_reranking(self):
        self.qdrant.search_chunks.return_value = [
            chunk(text="alpha", index=0),
            chunk(text="alpha", index=1),
            chunk(text="beta", index=2),
        ]
        self.nvidia.rerank.side_effect = lambda question, chunks, **kwargs: chunks[:1]

        services.answer_question("what is the leave policy")

        _, rerank_chunks, *_ = self.nvidia.rerank.call_args.args
        self.assertEqual([item["text"] for item in rerank_chunks], ["alpha", "beta"])

    def test_empty_rerank_reports_not_found_in_the_users_language(self):
        self.nvidia.rerank.return_value = []

        result = services.answer_question("company ka refund process kya hai")

        self.assertEqual(result["answer"], NOT_FOUND[intents.HINGLISH])
        self.assertEqual(result["sources"], [])

    def test_sentinel_answer_reports_not_found(self):
        self.nvidia.generate_answer.return_value = "NO_ANSWER"

        result = services.answer_question("who won the 2019 world cup")

        self.assertEqual(result["answer"], NOT_FOUND[intents.EN])
        self.assertEqual(result["sources"], [])

    def test_empty_answer_reports_not_found(self):
        self.nvidia.generate_answer.return_value = "   "

        result = services.answer_question("what is the leave policy")

        self.assertEqual(result["answer"], NOT_FOUND[intents.EN])

    def test_language_directive_reaches_the_prompt(self):
        services.answer_question("leave policy kitne din ki hai")

        _, user_prompt = self.nvidia.generate_answer.call_args[0]
        self.assertIn("Answer language: Hinglish", user_prompt)


class RewrittenQueryTest(TestCase):
    """A Hinglish question must retrieve as well as its English form."""

    def setUp(self):
        nvidia_patch = patch.object(services, "nvidia")
        qdrant_patch = patch.object(services, "qdrant")

        self.nvidia = nvidia_patch.start()
        self.qdrant = qdrant_patch.start()

        self.addCleanup(nvidia_patch.stop)
        self.addCleanup(qdrant_patch.stop)

        self.nvidia.classify.return_value = (
            '{"intent": "knowledge", "language": "hinglish",'
            ' "search_query": "leave policy process"}'
        )
        self.nvidia.generate_embedding.return_value = [0.1]
        self.nvidia.generate_answer.return_value = "**Answer**"
        self.nvidia.rerank.return_value = [chunk()]
        self.qdrant.search_chunks.return_value = [chunk()]
        self.qdrant.with_neighbours.return_value = [chunk()]

    def test_dense_search_uses_the_english_rewrite(self):
        question = "Mujhe leave policy ka poora process samjha do"

        services.answer_question(question)

        self.nvidia.generate_embedding.assert_called_once_with("leave policy process")
        # The keyword side keeps the original wording too, so a literal the
        # rewrite dropped can still match.
        keywords = self.qdrant.search_chunks.call_args.kwargs["question"]
        self.assertIn("leave policy process", keywords)
        self.assertIn(question, keywords)

    def test_the_model_still_sees_the_original_question(self):
        question = "Mujhe leave policy ka poora process samjha do"

        services.answer_question(question)

        _, user_prompt = self.nvidia.generate_answer.call_args[0]
        self.assertIn(question, user_prompt)
        self.assertIn("Answer language: Hinglish", user_prompt)


def exchange(index, answer_chars=10):
    return [
        {"role": "user", "text": f"question {index}"},
        {"role": "assistant", "text": "a" * answer_chars},
    ]


class HistoryWindowTest(TestCase):
    """The rolling window: tail slice, no orphan reply, then a char budget."""

    def conversation(self, pairs, answer_chars=10):
        messages = []

        for index in range(pairs):
            messages.extend(exchange(index, answer_chars))

        return messages

    def test_short_history_passes_through(self):
        messages = self.conversation(2)

        self.assertEqual(history.window(messages, limit=20), messages)

    def test_only_the_tail_is_kept(self):
        messages = self.conversation(6)  # 12 messages

        kept = history.window(messages, limit=4)

        self.assertEqual([entry["text"] for entry in kept][0], "question 4")
        self.assertEqual(len(kept), 4)

    def test_window_never_opens_on_an_assistant_message(self):
        messages = self.conversation(6)

        kept = history.window(messages, limit=5)

        # The slice would have started on an answer, so that answer is dropped.
        self.assertEqual(kept[0]["role"], "user")
        self.assertEqual(len(kept), 4)

    def test_long_messages_are_clipped(self):
        messages = self.conversation(1, answer_chars=500)

        kept = history.window(messages, limit=20, message_chars=100)

        self.assertTrue(kept[1]["text"].endswith("..."))
        self.assertLessEqual(len(kept[1]["text"]), 104)

    def test_budget_drops_whole_pairs_from_the_front(self):
        messages = self.conversation(4, answer_chars=100)

        kept = history.window(messages, limit=20, char_budget=250)

        self.assertLessEqual(history.total_chars(kept), 250)
        self.assertEqual(kept[0]["role"], "user")

    def test_empty_history(self):
        self.assertEqual(history.window(None, limit=20), [])
        self.assertEqual(history.window([], limit=20), [])

    def test_blank_messages_are_skipped(self):
        messages = [
            {"role": "user", "text": "question"},
            {"role": "assistant", "text": ""},
        ]

        self.assertEqual(len(history.window(messages, limit=20)), 1)


@override_settings(RAG_ENABLE_LLM_INTENT=False)
class HistoryRoutingTest(TestCase):
    """History has to reach the classifier and the answer call, and only those."""

    def setUp(self):
        nvidia_patch = patch.object(services, "nvidia")
        qdrant_patch = patch.object(services, "qdrant")

        self.nvidia = nvidia_patch.start()
        self.qdrant = qdrant_patch.start()

        self.addCleanup(nvidia_patch.stop)
        self.addCleanup(qdrant_patch.stop)

        self.nvidia.generate_embedding.return_value = [0.1]
        self.nvidia.generate_answer.return_value = "**Answer**"
        self.nvidia.rerank.return_value = [chunk()]
        self.qdrant.search_chunks.return_value = [chunk()]
        self.qdrant.with_neighbours.return_value = [chunk()]

        self.history = [
            {"role": "user", "text": "what is the leave policy"},
            {"role": "assistant", "text": "Earned leave is 12 days a year."},
        ]

    def test_knowledge_answer_forwards_only_the_user_turns(self):
        """Its own past answers are a fact source the model cannot resist."""

        services.answer_question("and the notice period?", history=self.history)

        self.assertEqual(
            self.nvidia.generate_answer.call_args.kwargs["history"],
            [{"role": "user", "text": "what is the leave policy"}],
        )

    def test_no_assistant_text_reaches_the_answer_call(self):
        services.answer_question("and the notice period?", history=self.history)

        forwarded = self.nvidia.generate_answer.call_args.kwargs["history"]

        self.assertTrue(all(entry["role"] == "user" for entry in forwarded))

    def test_the_classifier_still_sees_both_roles(self):
        """It only writes a search query, so prior answers are safe there."""

        with patch.object(services.intents, "classify") as classify:
            classify.return_value = {
                "intent": intents.KNOWLEDGE,
                "language": intents.EN,
                "search_query": "notice period",
                "source": "model",
            }

            services.answer_question("and the notice period?", history=self.history)

        self.assertEqual(classify.call_args.kwargs["history"], self.history)

    def test_small_talk_forwards_history(self):
        services.answer_question("thanks", history=self.history)

        self.assertEqual(
            self.nvidia.generate_answer.call_args.kwargs["history"],
            self.history,
        )

    def test_overview_ignores_history(self):
        self.qdrant.sample_chunks.return_value = [chunk(score=None)]

        services.answer_question("give me context", history=self.history)

        self.assertNotIn("history", self.nvidia.generate_answer.call_args.kwargs)

    def test_history_is_optional(self):
        """Every existing caller passes one argument and must keep working."""

        result = services.answer_question("what is the leave policy")

        self.assertEqual(result["answer"], "**Answer**")
        self.assertEqual(
            self.nvidia.generate_answer.call_args.kwargs["history"],
            [],
        )

    @override_settings(RAG_HISTORY_LIMIT=2)
    def test_only_the_window_is_forwarded(self):
        long_history = self.history + [
            {"role": "user", "text": "and sick leave?"},
            {"role": "assistant", "text": "8 days a year."},
        ]

        services.answer_question("and the notice period?", history=long_history)

        # Window is the last 2 messages; only the user one survives the filter.
        self.assertEqual(
            self.nvidia.generate_answer.call_args.kwargs["history"],
            [{"role": "user", "text": "and sick leave?"}],
        )


class FollowUpClassificationTest(TestCase):
    """The rewrite is the whole reason history reaches the classifier."""

    def test_history_reaches_the_classifier_prompt(self):
        provider = MagicMock()
        provider.classify.return_value = (
            '{"intent": "knowledge", "language": "hinglish",'
            ' "search_query": "leave carry forward rules"}'
        )
        past = [
            {"role": "user", "text": "leave policy kya hai"},
            {"role": "assistant", "text": "Earned leave 12 din."},
        ]

        decision = intents.classify("aur carry forward ka?", provider, history=past)

        _, user_prompt = provider.classify.call_args[0]
        self.assertIn("leave policy kya hai", user_prompt)
        # normalise() strips the trailing "?" - routing only, the answer prompt
        # still gets the question as the user typed it.
        self.assertIn("aur carry forward ka", user_prompt)
        self.assertEqual(decision["search_query"], "leave carry forward rules")

    def test_rules_still_win_without_asking_the_model(self):
        provider = MagicMock()
        past = [{"role": "user", "text": "leave policy kya hai"}]

        decision = intents.classify("hi", provider, history=past)

        self.assertEqual(decision["intent"], intents.SMALL_TALK)
        provider.classify.assert_not_called()

    def test_no_history_keeps_the_old_prompt(self):
        provider = MagicMock()
        provider.classify.return_value = (
            '{"intent": "knowledge", "language": "en", "search_query": "notice period"}'
        )

        intents.classify("what is the notice period", provider)

        _, user_prompt = provider.classify.call_args[0]
        self.assertNotIn("Recent conversation", user_prompt)


class BuildSourcesTest(TestCase):
    def test_deduplicates_pages(self):
        results = [
            chunk(page=2, index=4),
            chunk(page=2, index=5),
            chunk(page=3, index=6),
        ]

        pages = [source["page_number"] for source in services.build_sources(results)]

        self.assertEqual(pages, [2, 3])

    def test_tolerates_a_missing_score(self):
        sources = services.build_sources([chunk(score=None)])

        self.assertIsNone(sources[0]["score"])

    def test_does_not_depend_on_the_answer_wording(self):
        """Sniffing the answer text broke as soon as answers were not English."""

        self.assertEqual(len(services.build_sources([chunk()])), 1)
