from django.conf import settings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVector,
    SparseVectorParams,
    Modifier,
    Prefetch,
    FusionQuery,
    Fusion,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

import logging
import re
import uuid
import zlib
from collections import Counter

logger = logging.getLogger(__name__)

POINT_NAMESPACE = uuid.UUID("6f2d1c4e-9a3b-4f5e-8c7d-1b2a3c4d5e6f")

DENSE = "dense"
SPARSE = "text"

# Keeps ids, emails, versions and error codes as single tokens.
TOKEN = re.compile(r"[a-z0-9_.@+-]{2,}")


def point_id(document_id, chunk_index):
    return str(uuid.uuid5(POINT_NAMESPACE, f"{document_id}:{chunk_index}"))


def sparse_vector(text):
    """Term frequencies keyed by a hash of each token.

    Qdrant applies IDF server-side, so rare literals - an AWS account number,
    a repo name - dominate the match. That is exactly what dense embeddings
    are worst at, which is why the two retrievers are fused rather than ranked.
    """

    counts = Counter(TOKEN.findall(text.lower()))

    return SparseVector(
        indices=[zlib.crc32(token.encode()) for token in counts],
        values=[float(count) for count in counts.values()],
    )


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
                vectors_config={
                    DENSE: VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    SPARSE: SparseVectorParams(modifier=Modifier.IDF),
                },
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
                    vector={
                        DENSE: embedding,
                        SPARSE: sparse_vector(chunk["text"]),
                    },
                    payload={
                        "document_id": document.id,
                        "file_name": document.original_name,
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                    },
                )
            )
        # print("POINTS \n", points)
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
        question,
        limit=24,
    ):
        """Fetch candidates for the reranker - recall matters here, not order.

        Dense retrieval alone ranks an exact identifier like an AWS account
        number around 7th; the keyword side puts it 1st. Reciprocal Rank Fusion
        runs server-side, so neither retriever has to win outright.
        """

        try:
            results = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                prefetch=[
                    Prefetch(
                        query=question_embedding,
                        using=DENSE,
                        limit=limit,
                    ),
                    Prefetch(
                        query=sparse_vector(question),
                        using=SPARSE,
                        limit=limit,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
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
                    "document_id": point.payload["document_id"],
                    "file_name": point.payload["file_name"],
                    "page_number": point.payload["page_number"],
                    "chunk_index": point.payload["chunk_index"],
                    "text": point.payload["text"],
                }
            )
        logger.debug("Fused %d unique candidates", len(output))

        return output

    def sample_chunks(self, document_ids, per_document):
        """The opening chunks of each document, for questions with no topic.

        "Give me context" cannot be searched for: there is nothing to match, so
        fusion returns arbitrary chunks and the reranker drops them all. Point
        ids are deterministic, so the start of every document can be read
        directly instead - no query, no extra index.
        """

        ids = [
            point_id(document_id, chunk_index)
            for document_id in document_ids
            for chunk_index in range(per_document)
        ]

        if not ids:
            return []

        try:
            points = self.client.retrieve(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                ids=ids,
                with_payload=True,
            )
        except ResponseHandlingException as exc:
            raise connection_error() from exc
        except UnexpectedResponse as exc:
            # Nothing has been indexed yet, so the collection does not exist.
            if exc.status_code == 404:
                return []
            raise

        chunks = [
            {
                "score": None,
                "document_id": point.payload["document_id"],
                "file_name": point.payload["file_name"],
                "page_number": point.payload["page_number"],
                "chunk_index": point.payload["chunk_index"],
                "text": point.payload["text"],
            }
            for point in points
        ]

        return sorted(
            chunks,
            key=lambda chunk: (chunk["file_name"], chunk["chunk_index"]),
        )

    def with_neighbours(self, chunks):
        """Add the chunk before and after each hit, in document order.

        A section that spans a chunk boundary only ever retrieves its first
        half: the continuation answers no question on its own, so it ranks far
        too low to be fetched. Pulling it in by position rather than by score
        is what keeps a table or a procedure whole.
        """

        selected = {(chunk["document_id"], chunk["chunk_index"]) for chunk in chunks}

        wanted = {
            (chunk["document_id"], chunk["chunk_index"] + offset)
            for chunk in chunks
            for offset in (-1, 1)
        } - selected

        try:
            # Point ids are deterministic, so neighbours can be addressed
            # directly. Ids past either end of a document simply return nothing.
            points = self.client.retrieve(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                ids=[point_id(*key) for key in wanted],
                with_payload=True,
            )
        except ResponseHandlingException as exc:
            raise connection_error() from exc

        neighbours = [
            {
                "score": None,
                "document_id": point.payload["document_id"],
                "file_name": point.payload["file_name"],
                "page_number": point.payload["page_number"],
                "chunk_index": point.payload["chunk_index"],
                "text": point.payload["text"],
            }
            for point in points
        ]

        return sorted(
            chunks + neighbours,
            key=lambda chunk: (chunk["file_name"], chunk["chunk_index"]),
        )

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
