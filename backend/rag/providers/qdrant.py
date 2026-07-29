from django.conf import settings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

import uuid

# Namespace for deterministic point ids. Re-indexing a document then overwrites
# its existing points instead of inserting a second copy of every chunk.
POINT_NAMESPACE = uuid.UUID("6f2d1c4e-9a3b-4f5e-8c7d-1b2a3c4d5e6f")


def point_id(document_id, chunk_index):
    return str(uuid.uuid5(POINT_NAMESPACE, f"{document_id}:{chunk_index}"))


class VectorStoreUnavailable(Exception):
    """The Qdrant server could not be reached."""


def connection_error():
    return VectorStoreUnavailable(
        f"Could not connect to the vector database at {settings.QDRANT_URL}. "
        "Start Qdrant and try again "
        "(docker start qdrant)."
    )


class QdrantProvider:

    def __init__(self):

        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )

    def health_check(self):
        try:
            return self.client.get_collections()
        except ResponseHandlingException as exc:
            raise connection_error() from exc

    def ensure_collection(
        self,
        vector_size,
    ):
        try:
            collections = self.client.get_collections()

            existing = [c.name for c in collections.collections]

            if settings.QDRANT_COLLECTION_NAME in existing:
                return

            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
        except ResponseHandlingException as exc:
            raise connection_error() from exc

    def upsert_chunks(
        self,
        document,
        chunks,
        embeddings,
    ):
        points = []
        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            points.append(
                PointStruct(
                    id=point_id(document.id, chunk["chunk_index"]),
                    vector=embedding,
                    payload={
                        "document_id": document.id,
                        "file_name": document.original_name,
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                    },
                )
            )

        try:
            self.client.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=points,
            )
        except ResponseHandlingException as exc:
            raise connection_error() from exc

    def search_chunks(
        self,
        question_embedding,
        limit=8,
    ):

        try:
            # Over-fetch so that dropping duplicates still leaves `limit`
            # distinct chunks to answer from.
            results = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=question_embedding,
                limit=limit * 3,
            )
        except ResponseHandlingException as exc:
            raise connection_error() from exc
        except UnexpectedResponse as exc:
            # Nothing has been indexed yet, so the collection does not exist.
            if exc.status_code == 404:
                return []
            raise

        output = []
        seen_text = set()

        for point in results.points:

            # The same passage is often indexed more than once - the file was
            # uploaded twice, or an older index run left its points behind.
            fingerprint = point.payload["text"].strip()

            if fingerprint in seen_text:
                continue

            seen_text.add(fingerprint)

            output.append(
                {
                    "score": point.score,
                    "file_name": point.payload["file_name"],
                    "page_number": point.payload["page_number"],
                    "chunk_index": point.payload["chunk_index"],
                    "text": point.payload["text"],
                }
            )

            if len(output) >= limit:
                break
        print("OUTPUT:", output)

        return output

    def delete_chunks(self, document_id):
        try:
            self.client.delete(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
        except ResponseHandlingException as exc:
            raise connection_error() from exc
        except UnexpectedResponse as exc:
            # No collection means there is nothing to delete.
            if exc.status_code == 404:
                return
            raise
