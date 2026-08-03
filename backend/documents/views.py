from django.shortcuts import render
from .models import Document
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import DocumentUploadSerializer
from .services import DocumentService
from rest_framework.permissions import IsAuthenticated
from accounts.authenticate import CsrfExemptSessionAuthentication
from rag.providers.qdrant import VectorStoreUnavailable


class UploadDocumentView(APIView):
    authentication_classes = [CsrfExemptSessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        print(request.data)
        serializer = DocumentUploadSerializer(data=request.data)
        print(serializer.is_valid())
        print(serializer)

        serializer.is_valid(raise_exception=True)

        document = DocumentService.upload(serializer.validated_data["file"])
        print("Document:", document)

        try:
            DocumentService.process_document(document)
        except VectorStoreUnavailable as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        document.refresh_from_db()

        return Response(
            {
                "id": document.id,
                "original_name": document.original_name,
                "status": document.status,
                "total_pages": document.total_pages,
                "total_chunks": document.total_chunks,
            },
            status=status.HTTP_201_CREATED,
        )


class DeleteDocumentView(APIView):
    authentication_classes = [CsrfExemptSessionAuthentication]
    permission_classes = [IsAuthenticated]

    def delete(self, request, document_id):
        try:
            document = Document.objects.get(id=document_id)
        except Document.DoesNotExist:
            return Response(
                {"detail": "Document not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        DocumentService.delete_document(document)

        return Response(
            {"detail": "Document successfully deleted."},
            status=status.HTTP_200_OK,
        )
