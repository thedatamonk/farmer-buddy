"""Chat API endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from kisan.agent.orchestrator import AgentOrchestrator
from kisan.api.dependencies import (
    get_llm_service,
    get_session_manager,
    get_vectordb_service,
)
from kisan.core.exceptions import SessionNotFoundError
from kisan.core.logging import logger
from kisan.schemas.chat import ChatRequest, ChatResponse, ConversationHistory
from kisan.services.llm import LLMService
from kisan.services.session import SessionManager
from kisan.services.vectordb import VectorDBService

router = APIRouter()


def get_orchestrator(
    llm_service: LLMService = Depends(get_llm_service),
    session_manager: SessionManager = Depends(get_session_manager),
    vectordb_service: VectorDBService = Depends(get_vectordb_service),
) -> AgentOrchestrator:
    """Get agent orchestrator instance."""
    return AgentOrchestrator(llm_service, session_manager, vectordb_service)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """Process a chat message and return a response.

    Optionally include a base64-encoded image for disease detection.
    """
    has_image = request.image is not None
    logger.info(f"Chat request: session={request.session_id}, has_image={has_image}")

    response = await orchestrator.process_message(
        message=request.message,
        session_id=request.session_id,
        image_base64=request.image,
    )

    return response


@router.get("/sessions/{session_id}", response_model=ConversationHistory)
async def get_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> ConversationHistory:
    """Get conversation history for a session."""
    try:
        return session_manager.get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Clear or delete a session."""
    try:
        session_manager.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
