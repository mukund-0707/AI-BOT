from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import AskQuestionSerializer

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

        try:
            result = answer_question(
                question=serializer.validated_data["question"],
            )
        except VectorStoreUnavailable as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response(
            result,
            status=status.HTTP_200_OK,
        )
