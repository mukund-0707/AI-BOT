from rest_framework import serializers

from documents.models import Document


class AskQuestionSerializer(serializers.Serializer):

    question = serializers.CharField()

    def validate_question(self, value):

        value = value.strip()

        if not value:
            raise serializers.ValidationError("Question cannot be empty.")

        return value
