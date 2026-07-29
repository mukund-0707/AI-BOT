from django.urls import path
from .views import *

urlpatterns = [
    path(
        "upload/",
        UploadDocumentView.as_view(),
        name="document-upload",
    ),
    path(
        "<int:document_id>/",
        DeleteDocumentView.as_view(),
        name="document-delete",
    ),
]
