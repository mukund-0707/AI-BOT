from django.urls import path

from .views import AskQuestionView
from .views import ResetConversationView
from .views import chatbot

urlpatterns = [
    path(
        "ask/",
        AskQuestionView.as_view(),
        name="ask-question",
    ),
    path(
        "new/",
        ResetConversationView.as_view(),
        name="new-chat",
    ),
    path(
        "",
        chatbot,
        name="chatbot",
    ),
]
