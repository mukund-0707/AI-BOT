from django.db import models

# Create your models here.


class Document(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    file = models.FileField(upload_to="documents/")
    original_name = models.CharField(max_length=255)
    file_type = models.CharField(max_length=20)
    file_size = models.PositiveBigIntegerField()

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.UPLOADED,
    )

    total_pages = models.PositiveIntegerField(default=0)
    total_chunks = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.original_name
