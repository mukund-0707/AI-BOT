from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import AskQuestionSerializer

from rag.services import answer_question
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

        result = answer_question(
            question=serializer.validated_data["question"],
        )

        return Response(
            result,
            status=status.HTTP_200_OK,
        )



# hi mujeh teri ek helpp chahiye mene ek chatbot create kia hua its okay chal raha he no doubt but vo smart nhi h itna 

# means agr tu usko bolega ki mereko summary de brief to vo sare chunks ko padh k summaries nhi kr skta ya respond nhi kr skta 

# i mean need a better solution kyuki mere idea kthm ho gaye he isko leke kya krna chahiye 

# kyuki query k bases pr chunks aaate he agr tune suppose k brief information de likha kuch nhi aaega vese mene thoda boht optimize kiya hua he ab aaega bt bs handle krvaya he vo sahi me nhi krega 

# mereko bta ki kya krna chahiye 


# chatbot kese optimise karu mujeh better approach de phri me use reiview kruga...