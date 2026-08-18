from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import AskQuestionSerializer

from chat import history

from rag.services import answer_question, stream_answer_question
from rag.providers.qdrant import VectorStoreUnavailable
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt

import json
import logging

logger = logging.getLogger(__name__)
STREAM_FAILED = "The answer stopped unexpectedly. Please try again."


@ensure_csrf_cookie
def chatbot(request):
    return render(
        request,
        "chatbot/index.html",
    )


class AskQuestionView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):

        serializer = AskQuestionSerializer(data=request.data)

        serializer.is_valid(raise_exception=True)

        question = serializer.validated_data["question"]

        history.conversation_id(request.session)

        try:
            result = answer_question(
                question=question,
                history=history.load(request.session),
            )
        except VectorStoreUnavailable as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # Only a completed exchange is stored: a question whose answer failed
        # would otherwise sit in the history with no reply under it.
        history.append(request.session, question, result["answer"])

        return Response(
            result,
            status=status.HTTP_200_OK,
        )


class ResetConversationView(APIView):
    """ "New chat": forget the thread, keep the session (and any login)."""

    permission_classes = [AllowAny]

    def post(self, request):

        conversation_id = history.reset(request.session)

        return Response(
            {"conversation_id": conversation_id},
            status=status.HTTP_200_OK,
        )


@method_decorator(csrf_exempt, name="dispatch")
class StreamAskView(View):
    """Streaming version of AskQuestionView using Server-Sent Events.

    Plain Django View — not DRF APIView — so DRF content negotiation never
    runs and StreamingHttpResponse passes through untouched.

    Each token is sent as:
        data: {"token": "..."}\n\n

    When the stream ends, a final event carries the sources and full answer:
        data: {"done": true, "sources": [...], "answer": "..."}\n\n
    """

    def post(self, request):
        try:
            body = json.loads(request.body)
        except (json.JSONDecodeError, ValueError):
            return JsonResponse({"detail": "Invalid JSON."}, status=400)

        question = (body.get("question") or "").strip()

        if not question:
            return JsonResponse({"detail": "question is required."}, status=400)

        history.conversation_id(request.session)
        current_history = history.load(request.session)
        session = request.session

        def event_stream():
            final_meta = None

            try:
                for chunk in stream_answer_question(question, current_history):
                    if isinstance(chunk, dict):
                        final_meta = chunk
                    else:
                        yield f"data: {json.dumps({'token': chunk})}\n\n"
            except VectorStoreUnavailable as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return
            except Exception:
                logger.exception("Streaming answer failed")
                yield f"data: {json.dumps({'error': STREAM_FAILED})}\n\n"
                return

            if final_meta:
                history.append(session, question, final_meta["answer"])
                session.save()
                yield f"data: {json.dumps({'done': True, **final_meta})}\n\n"

        response = StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response
