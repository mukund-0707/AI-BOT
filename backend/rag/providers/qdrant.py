# pyrefly: ignore [missing-import]
from django.conf import settings

# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

import uuid


class QdrantProvider:

    def __init__(self):

        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )

    def health_check(self):
        return self.client.get_collections()

    def ensure_collection(
        self,
        vector_size,
    ):
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
                    id=str(uuid.uuid4()),
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

        self.client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
        )

    def search_chunks(
        self,
        question_embedding,
        limit=10,
    ):

        from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

        try:
            results = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=question_embedding,
                limit=limit,
            )
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return []
            raise
        except ResponseHandlingException:
            return []
        except Exception as e:
            if "Not found: Collection" in str(e):
                return []
            raise

        output = []

        for point in results.points:

            output.append(
                {
                    "score": point.score,
                    "file_name": point.payload["file_name"],
                    "page_number": point.payload["page_number"],
                    "chunk_index": point.payload["chunk_index"],
                    "text": point.payload["text"],
                }
            )
        print("OUTPUT:", output)

        return output

    def delete_chunks(self, document_id):
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
