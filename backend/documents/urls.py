from django.urls import path
from .views import *

urlpatterns = [
    path(
        "upload/",
        UploadDocumentView.as_view(),
        name="document-upload",
    ),
    # path(
    #     "<int:document_id>/process/",
    #     ProcessDocumentView.as_view(),
    #     name="document-process",
    # ),
    path(
        "<int:document_id>/",
        DeleteDocumentView.as_view(),
        name="document-delete",
    ),
]


# POST /api/documents/upload/
# GET  /api/documents/
# GET  /api/documents/{id}/
# DELETE /api/documents/{id}/
