from pathlib import Path
from django.utils import timezone

from .models import Document
from .extractors.pdf import extract_pdf
from .extractors.docx import extract_docx
from .extractors.text import extract_text
from rag.chunking import create_chunks
from documents.text_cleaner import clean_text


from rag.providers.nvidia import NVIDIAProvider
from rag.providers.qdrant import QdrantProvider


class DocumentService:
    @staticmethod
    def upload(file):
        extension = Path(file.name).suffix.lower().replace(".", "")

        document = Document.objects.create(
            file=file,
            original_name=file.name,
            file_type=extension,
            file_size=file.size,
        )

        return document

    @staticmethod
    def extract_document(document):
        extension = Path(document.file.name).suffix.lower()

        if extension == ".pdf":
            return extract_pdf(document.file.path)
        elif extension == ".docx":
            return extract_docx(document.file.path)
        elif extension == ".txt":
            return extract_text(document.file.path)
        else:
            raise ValueError(f"Unsupported document type: {extension}")

    @staticmethod
    def process_document(document):

        try:
            document.status = Document.Status.PROCESSING
            document.save(update_fields=["status"])

            pages = DocumentService.extract_document(document)
            print("PAGES \n", pages)

            cleaned_pages = []

            for page in pages:
                text = clean_text(page["text"])

                if text:
                    cleaned_pages.append(
                        {
                            "page_number": page["page_number"],
                            "text": text,
                        }
                    )
                    print("CLEANED_PAGES: \n", cleaned_pages)

            chunks = create_chunks(cleaned_pages)
            # print("CHUNKS: \n", chunks)

            if not chunks:
                document.status = Document.Status.FAILED
                document.error_message = (
                    "No text chunks were created from the document."
                )
                document.save(update_fields=["status", "error_message"])
                return

            provider = NVIDIAProvider()

            embeddings = []
            texts = [chunk["text"] for chunk in chunks]
            batch_size = 50

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = provider.generate_embeddings_batch(batch_texts)
                embeddings.extend(batch_embeddings)

            if not embeddings or not embeddings[0]:
                document.status = Document.Status.FAILED
                document.error_message = (
                    "Unable to generate embeddings for document chunks."
                )
                document.save(update_fields=["status", "error_message"])
                return

            qdrant = QdrantProvider()

            qdrant.ensure_collection(len(embeddings[0]))

            # Remove old chunks before re-indexing.
            deel = qdrant.delete_chunks(document.id)
            print("DEEL \n", deel)

            qdrant.upsert_chunks(
                document,
                chunks,
                embeddings,
            )

            document.status = Document.Status.READY
            document.total_pages = len(cleaned_pages)
            document.total_chunks = len(chunks)
            document.processed_at = timezone.now()

            document.save(
                update_fields=[
                    "status",
                    "total_pages",
                    "total_chunks",
                    "processed_at",
                ]
            )

        except Exception as exc:
            document.status = Document.Status.FAILED
            document.error_message = str(exc)

            document.save(
                update_fields=[
                    "status",
                    "error_message",
                ]
            )

            raise

    @staticmethod
    def delete_document(document):
        try:
            qdrant = QdrantProvider()
            qdrant.delete_chunks(document.id)

            if document.file:
                document.file.delete(save=False)

            document.delete()
        except Exception as exc:
            print(f"Error deleting document: {exc}")
            raise
