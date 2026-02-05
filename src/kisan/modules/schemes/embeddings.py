"""Embedding generation for scheme documents."""

from kisan.core.logging import logger
from kisan.services.llm import LLMService


class SchemeEmbeddings:
    """Generate embeddings for scheme documents."""

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self._batch_size = 100  # OpenAI embedding batch limit

    async def embed_documents(self, documents: list[dict]) -> list[list[float]]:
        """Generate embeddings for a list of documents.

        Args:
            documents: List of dicts with 'content' key

        Returns:
            List of embedding vectors
        """
        texts = [doc["content"] for doc in documents]
        return await self.embed_texts(texts)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug(f"Embedding batch {i // self._batch_size + 1}")
            embeddings = await self.llm.embed(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        return await self.llm.embed_single(query)
