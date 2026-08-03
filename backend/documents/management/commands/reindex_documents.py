from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from documents.models import Document
from documents.services import DocumentService
from rag.providers.qdrant import QdrantProvider, VectorStoreUnavailable


class Command(BaseCommand):
    help = (
        "Re-extract, re-embed and re-upload every stored document into Qdrant. "
        "Use this after the Qdrant storage has been wiped or recreated."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--ids",
            nargs="+",
            type=int,
            help="Only reindex these document ids.",
        )
        parser.add_argument(
            "--purge-orphans",
            action="store_true",
            help=(
                "Also delete vectors whose document no longer exists in the "
                "database. Ignored when --ids is used."
            ),
        )

    def handle(self, *args, **options):

        qdrant = QdrantProvider()

        try:
            qdrant.health_check()
        except VectorStoreUnavailable as exc:
            self.stderr.write(self.style.ERROR(str(exc)))
            return

        if options["purge_orphans"] and not options["ids"]:
            self.purge_orphans(qdrant)

        documents = Document.objects.order_by("id")

        if options["ids"]:
            documents = documents.filter(id__in=options["ids"])

        if not documents.exists():
            self.stdout.write("No documents to reindex.")
            return

        succeeded = 0
        failed = 0

        for document in documents:
            if not Path(document.file.path).exists():
                failed += 1
                self.stderr.write(
                    self.style.WARNING(
                        f"[{document.id}] {document.original_name} - "
                        "file is missing on disk, skipped."
                    )
                )
                continue

            self.stdout.write(f"[{document.id}] {document.original_name} ...")

            try:
                DocumentService.process_document(document)
            except Exception as exc:
                failed += 1
                self.stderr.write(self.style.ERROR(f"[{document.id}] failed: {exc}"))
                continue

            document.refresh_from_db()

            succeeded += 1
            self.stdout.write(
                self.style.SUCCESS(
                    f"[{document.id}] indexed {document.total_chunks} chunks."
                )
            )

        self.stdout.write(
            self.style.SUCCESS(f"Done. {succeeded} indexed, {failed} failed.")
        )

    def purge_orphans(self, qdrant):

        known_ids = set(Document.objects.values_list("id", flat=True))

        offset = None
        orphan_ids = set()

        while True:
            points, offset = qdrant.client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=["document_id"],
                with_vectors=False,
            )

            for point in points:
                document_id = point.payload.get("document_id")
                if document_id not in known_ids:
                    orphan_ids.add(document_id)

            if offset is None:
                break

        if not orphan_ids:
            self.stdout.write("No orphaned vectors found.")
            return

        for document_id in sorted(orphan_ids, key=lambda value: (value is None, value)):
            qdrant.delete_chunks(document_id)

        self.stdout.write(
            self.style.SUCCESS(
                f"Purged vectors for {len(orphan_ids)} deleted document(s): "
                f"{sorted(orphan_ids, key=str)}"
            )
        )
