from django.urls import path

from .views import AskQuestionView
from .views import chatbot

urlpatterns = [
    path(
        "ask/",
        AskQuestionView.as_view(),
        name="ask-question",
    ),
    path(
        "",
        chatbot,
        name="chatbot",
    ),
]
