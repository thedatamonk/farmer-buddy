"""Qdrant vector database service."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchText,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

from kisan.core.config import Settings
from kisan.core.exceptions import IndexingError, RetrievalError, VectorDBError
from kisan.core.logging import logger


class VectorDBService:
    """Service for interacting with Qdrant vector database."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self.collection_name = settings.qdrant_collection
        self._embedding_dimension = 1536  # text-embedding-3-small dimension

    def ensure_collection(self) -> None:
        """Ensure the collection exists, create if not."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self._embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {self.collection_name}")

            # Ensure payload indexes exist for fields used in filters
            collection_info = self.client.get_collection(self.collection_name)
            indexed_fields = set(collection_info.payload_schema.keys()) if collection_info.payload_schema else set()

            # source uses keyword index (exact match for delete_by_source / count_by_source)
            if "source" not in indexed_fields:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created keyword index for 'source'")

            # scheme_name and section_header use full-text indexes (substring match via MatchText)
            for field in ("scheme_name", "section_header"):
                if field not in indexed_fields:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=TextIndexParams(
                            type="text",
                            tokenizer=TokenizerType.WORD,
                            lowercase=True,
                        ),
                    )
                    logger.info(f"Created full-text index for '{field}'")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise VectorDBError(f"Failed to ensure collection: {e}") from e

    def index_documents(
        self,
        embeddings: list[list[float]],
        documents: list[dict],
        ids: list[int | str] | None = None,
    ) -> int:
        """Index documents with their embeddings."""
        try:
            self.ensure_collection()

            if ids is None:
                # Generate deterministic UUID5 IDs from source + chunk_index
                # to avoid collisions across PDFs when delete_by_source resets counts
                ids = []
                for i, doc in enumerate(documents):
                    source = doc.get("source", "unknown")
                    chunk_index = doc.get("chunk_index", i)
                    ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}:{chunk_index}")))

            points = [
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload=doc,
                )
                for idx, embedding, doc in zip(ids, embeddings, documents)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(f"Indexed {len(points)} documents to {self.collection_name}")
            return len(points)

        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise IndexingError(f"Failed to index documents: {e}") from e

    def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        score_threshold: float = 0.0,
        filter_by: dict[str, str] | None = None,
    ) -> list[dict]:
        """Search for similar documents.

        Args:
            filter_by: Optional dict of payload field -> value for metadata filtering.
                       e.g. {"section_header": "Eligibility", "scheme_name": "PM-KISAN"}
        """
        try:
            top_k = top_k or self.settings.top_k_results

            query_filter = None
            if filter_by:
                conditions = [
                    FieldCondition(key=key, match=MatchText(text=value))
                    for key, value in filter_by.items()
                ]
                query_filter = Filter(must=conditions)

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
            ).points

            return [
                {
                    "id": str(hit.id),
                    "score": hit.score,
                    **hit.payload,
                }
                for hit in results
            ]

        except UnexpectedResponse as e:
            if "doesn't exist" in str(e):
                logger.warning(f"Collection {self.collection_name} does not exist")
                return []
            raise RetrievalError(f"Failed to search: {e}") from e
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            raise RetrievalError(f"Failed to search: {e}") from e

    def delete_by_source(self, source: str) -> None:
        """Delete all points where payload source matches the given filename."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="source", match=MatchValue(value=source))]
                    )
                ),
            )
            logger.info(f"Deleted points with source={source}")
        except UnexpectedResponse as e:
            if "doesn't exist" in str(e):
                logger.debug(f"Collection {self.collection_name} does not exist, nothing to delete")
                return
            raise VectorDBError(f"Failed to delete by source: {e}") from e
        except Exception as e:
            logger.error(f"Failed to delete by source: {e}")
            raise VectorDBError(f"Failed to delete by source: {e}") from e

    def count_by_source(self, source: str) -> int:
        """Count points with the given source filename."""
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                ),
            )
            return result.count
        except UnexpectedResponse as e:
            if "doesn't exist" in str(e):
                return 0
            raise VectorDBError(f"Failed to count by source: {e}") from e
        except Exception as e:
            logger.error(f"Failed to count by source: {e}")
            raise VectorDBError(f"Failed to count by source: {e}") from e

    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise VectorDBError(f"Failed to delete collection: {e}") from e

    def get_collection_info(self) -> dict | None:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
            }
        except UnexpectedResponse:
            return None
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
