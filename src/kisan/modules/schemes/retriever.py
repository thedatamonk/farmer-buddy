"""RAG retriever for government schemes."""

from kisan.agent.prompts import SCHEME_QUERY_PROMPT
from kisan.core.config import Settings
from kisan.core.exceptions import RetrievalError
from kisan.core.logging import logger
from kisan.modules.schemes.embeddings import SchemeEmbeddings
from kisan.schemas.scheme import SchemeDocument, SchemeResult
from kisan.services.llm import LLMService
from kisan.services.vectordb import VectorDBService


class SchemeRetriever:
    """Retrieve and answer questions about government schemes."""

    def __init__(
        self,
        llm_service: LLMService,
        vectordb_service: VectorDBService,
        settings: Settings,
    ):
        self.llm = llm_service
        self.embeddings = SchemeEmbeddings(llm_service)
        self.vectordb = vectordb_service
        self.settings = settings

    async def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SchemeDocument]:
        """Search for relevant scheme documents."""
        top_k = top_k or self.settings.top_k_results

        try:
            # Generate query embedding
            query_embedding = await self.embeddings.embed_query(query)

            # Search vector database
            results = self.vectordb.search(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=0.3,
            )

            documents = []
            for result in results:
                documents.append(
                    SchemeDocument(
                        content=result.get("content", ""),
                        source=result.get("source", "Unknown"),
                        page_number=result.get("page_number"),
                        score=result.get("score", 0.0),
                    )
                )

            logger.debug(f"Found {len(documents)} documents for query: {query[:50]}")
            return documents

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RetrievalError(f"Failed to search schemes: {e}") from e

    async def query(self, question: str) -> SchemeResult:
        """Answer a question about government schemes using RAG."""
        # Retrieve relevant documents
        documents = await self.search(question)

        if not documents:
            return SchemeResult(
                query=question,
                documents=[],
                answer="I couldn't find specific information about this in my knowledge base. "
                "Please try rephrasing your question or ask about a specific scheme name.",
                schemes_mentioned=[],
            )

        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Source {i}: {doc.source}, Page {doc.page_number}]\n{doc.content}"
            )
        context = "\n\n".join(context_parts)

        # Generate answer using LLM
        prompt = SCHEME_QUERY_PROMPT.format(context=context, question=question)
        messages = [{"role": "user", "content": prompt}]

        response = await self.llm.chat(messages, temperature=0.3)
        answer = response.get("content", "")

        # Extract mentioned scheme names
        schemes_mentioned = self._extract_scheme_names(answer)

        return SchemeResult(
            query=question,
            documents=documents,
            answer=answer,
            schemes_mentioned=schemes_mentioned,
        )

    def _extract_scheme_names(self, text: str) -> list[str]:
        """Extract scheme names mentioned in the text."""
        # Common scheme name patterns
        common_schemes = [
            "PM-KISAN", "PM Kisan", "Pradhan Mantri Kisan",
            "PMFBY", "Pradhan Mantri Fasal Bima",
            "KCC", "Kisan Credit Card",
            "PMKSY", "Pradhan Mantri Krishi Sinchai",
            "SMAM", "Sub-Mission on Agricultural Mechanization",
            "PKVY", "Paramparagat Krishi Vikas",
            "RKVY", "Rashtriya Krishi Vikas",
            "NFSM", "National Food Security Mission",
            "NMOOP", "National Mission on Oilseeds",
            "e-NAM", "National Agriculture Market",
        ]

        mentioned = []
        text_upper = text.upper()

        for scheme in common_schemes:
            if scheme.upper() in text_upper:
                # Prefer the abbreviated form if both exist
                if scheme in mentioned:
                    continue
                mentioned.append(scheme)

        return list(set(mentioned))[:10]  # Deduplicate and limit
