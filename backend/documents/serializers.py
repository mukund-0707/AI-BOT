from pathlib import Path

from rest_framework import serializers

from .models import Document


class DocumentUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ["id", "file"]

    def validate_file(self, file):
        allowed_extensions = [".pdf", ".docx", ".txt"]

        extension = Path(file.name).suffix.lower()

        if extension not in allowed_extensions:
            raise serializers.ValidationError(
                "Only PDF, DOCX and TXT files are allowed."
            )

        max_size = 10 * 1024 * 1024  # 10 MB

        if file.size > max_size:
            raise serializers.ValidationError("Maximum file size is 10 MB.")

        if file.size == 0:
            raise serializers.ValidationError("Uploaded file is empty.")

        return file
