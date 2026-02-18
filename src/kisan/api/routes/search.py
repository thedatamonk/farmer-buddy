"""Search API endpoints."""

from fastapi import APIRouter, Depends

from kisan.api.dependencies import get_llm_service, get_vectordb_service
from kisan.core.config import get_settings
from kisan.core.logging import logger
from kisan.modules.schemes.retriever import SchemeRetriever
from kisan.schemas.scheme import ChunkSearchRequest, ChunkSearchResponse
from kisan.services.llm import LLMService
from kisan.services.vectordb import VectorDBService

router = APIRouter()


@router.post("/search/chunks", response_model=ChunkSearchResponse)
async def search_chunks(
    request: ChunkSearchRequest,
    llm_service: LLMService = Depends(get_llm_service),
    vectordb_service: VectorDBService = Depends(get_vectordb_service),
) -> ChunkSearchResponse:
    """Search indexed scheme chunks directly.

    Returns raw retrieved chunks with metadata and similarity scores,
    without LLM answer generation.
    """
    logger.info(f"Chunk search: query='{request.query[:50]}', top_k={request.top_k}")

    settings = get_settings()
    retriever = SchemeRetriever(llm_service, vectordb_service, settings)

    chunks = await retriever.search(
        query=request.query,
        top_k=request.top_k,
        section_header=request.section_header,
        scheme_name=request.scheme_name,
    )

    return ChunkSearchResponse(
        query=request.query,
        total_results=len(chunks),
        chunks=chunks,
    )
