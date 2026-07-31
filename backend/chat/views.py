from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import AskQuestionSerializer

from chat import history

from rag.services import answer_question
from rag.providers.qdrant import VectorStoreUnavailable
from django.shortcuts import render

from django.views.decorators.csrf import ensure_csrf_cookie


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
    """"New chat": forget the thread, keep the session (and any login)."""

    permission_classes = [AllowAny]

    def post(self, request):

        conversation_id = history.reset(request.session)

        return Response(
            {"conversation_id": conversation_id},
            status=status.HTTP_200_OK,
        )
