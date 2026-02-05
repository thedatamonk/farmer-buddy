"""Qdrant vector database service."""

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from kisan.core.config import Settings
from kisan.core.exceptions import IndexingError, RetrievalError, VectorDBError
from kisan.core.logging import logger


class VectorDBService:
    """Service for interacting with Qdrant vector database."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url)
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
            else:
                logger.debug(f"Collection already exists: {self.collection_name}")

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
                # Generate integer IDs based on current count
                current_count = self.client.count(self.collection_name).count
                ids = [current_count + i for i in range(len(embeddings))]

            # Convert string IDs to integers if they are numeric strings
            processed_ids = []
            for idx in ids:
                if isinstance(idx, str) and idx.isdigit():
                    processed_ids.append(int(idx))
                elif isinstance(idx, int):
                    processed_ids.append(idx)
                else:
                    # Use UUID for non-numeric string IDs
                    import uuid
                    processed_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, str(idx))))

            points = [
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload=doc,
                )
                for idx, embedding, doc in zip(processed_ids, embeddings, documents)
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
    ) -> list[dict]:
        """Search for similar documents."""
        try:
            top_k = top_k or self.settings.top_k_results

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
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
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except UnexpectedResponse:
            return None
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
