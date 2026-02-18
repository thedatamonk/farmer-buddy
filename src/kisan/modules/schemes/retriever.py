"""RAG retriever for government schemes."""

import asyncio
import hashlib
import json

from kisan.agent.prompts import QUERY_DECOMPOSITION_PROMPT, SCHEME_QUERY_PROMPT
from kisan.core.config import Settings
from kisan.core.exceptions import RetrievalError
from kisan.core.logging import logger
from kisan.modules.schemes.embeddings import SchemeEmbeddings
from kisan.schemas.scheme import QueryAnalysis, SchemeDocument, SchemeResult
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
        section_header: str | None = None,
        scheme_name: str | None = None,
    ) -> list[SchemeDocument]:
        """Search for relevant scheme documents.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            section_header: Optional filter by section (e.g. "Eligibility Criteria").
            scheme_name: Optional filter by scheme name.
        """
        top_k = top_k or self.settings.top_k_results

        try:
            query_embedding = await self.embeddings.embed_query(query)

            filter_by: dict[str, str] | None = None
            if section_header or scheme_name:
                filter_by = {}
                if section_header:
                    filter_by["section_header"] = section_header
                if scheme_name:
                    filter_by["scheme_name"] = scheme_name

            results = self.vectordb.search(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=0.3,
                filter_by=filter_by,
            )

            documents = []
            for result in results:
                documents.append(
                    SchemeDocument(
                        content=result.get("content", ""),
                        source=result.get("source", "Unknown"),
                        page_number=result.get("page_number"),
                        page_numbers=result.get("page_numbers", []),
                        score=result.get("score", 0.0),
                        scheme_name=result.get("scheme_name", ""),
                        section_header=result.get("section_header", ""),
                        section_hierarchy=result.get("section_hierarchy", ""),
                        content_type=result.get("content_type", "paragraph"),
                    )
                )

            logger.debug(f"Found {len(documents)} documents for query: {query[:50]}")
            return documents

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RetrievalError(f"Failed to search schemes: {e}") from e

    async def _analyze_query(self, question: str) -> QueryAnalysis:
        """Use a lightweight LLM to decompose the query into sub-queries."""
        prompt = QUERY_DECOMPOSITION_PROMPT.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        response = await self.llm.chat(
            messages,
            temperature=0.0,
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
        )

        raw = json.loads(response["content"])
        return QueryAnalysis.model_validate(raw)

    async def _search_with_analysis(
        self, analysis: QueryAnalysis, top_k: int | None = None,
    ) -> list[SchemeDocument]:
        """Run parallel searches using decomposed sub-queries (no filters)."""
        top_k = top_k or self.settings.top_k_results

        # Search each sub-query with no filters
        tasks = [
            self.search(sub_q, top_k=top_k)
            for sub_q in analysis.sub_queries[:3]
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge and deduplicate
        seen: set[str] = set()
        merged: list[SchemeDocument] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning(f"Sub-query search failed: {result}")
                continue
            for doc in result:
                key = doc.source + hashlib.md5(doc.content.encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        # Sort by score descending and return top_k
        merged.sort(key=lambda d: d.score, reverse=True)
        return merged[:top_k]

    async def query(self, question: str) -> SchemeResult:
        """Answer a question about government schemes using RAG."""
        # Analyze query for rewriting, decomposition, and filter extraction
        try:
            analysis = await self._analyze_query(question)
            logger.debug(f"Query analysis: {analysis.model_dump()}")
            documents = await self._search_with_analysis(analysis)
        except Exception as e:
            logger.warning(f"Query analysis failed, falling back to direct search: {e}")
            documents = await self.search(question)

        if not documents:
            return SchemeResult(
                query=question,
                documents=[],
                answer=None,
                schemes_mentioned=[],
            )

        # Build context from documents with section metadata
        context_parts = []
        for i, doc in enumerate(documents, 1):
            header = f"[Source {i}: {doc.source}"
            if doc.section_hierarchy:
                header += f" | Section: {doc.section_hierarchy}"
            elif doc.section_header:
                header += f" | Section: {doc.section_header}"
            if doc.page_numbers:
                header += f", Pages {doc.page_numbers}"
            elif doc.page_number:
                header += f", Page {doc.page_number}"
            header += f" | Type: {doc.content_type}]"
            context_parts.append(f"{header}\n{doc.content}")
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
