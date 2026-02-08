"""ReAct-style agent orchestrator."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from kisan.agent.prompts import FALLBACK_RESPONSE, SYSTEM_PROMPT
from kisan.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from kisan.core.config import Settings, get_settings
from kisan.core.exceptions import KisanError, LLMError
from kisan.core.logging import logger
from kisan.schemas.chat import ChatResponse, MessageRole, ToolCall
from kisan.services.llm import LLMService
from kisan.services.session import SessionManager
from kisan.services.vectordb import VectorDBService

if TYPE_CHECKING:
    from kisan.modules.mandi.repository import MandiPriceRepository


class AgentOrchestrator:
    """ReAct-style agent that routes queries to appropriate tools."""

    def __init__(
        self,
        llm_service: LLMService,
        session_manager: SessionManager,
        vectordb_service: VectorDBService,
        settings: Settings | None = None,
        mandi_repository: MandiPriceRepository | None = None,
    ):
        self.settings = settings or get_settings()
        self.llm = llm_service
        self.sessions = session_manager
        self.tool_executor = ToolExecutor(
            llm_service, vectordb_service, self.settings, mandi_repository
        )
        self.max_tool_iterations = 3

    async def process_message(
        self,
        message: str,
        session_id: str | None = None,
        image_base64: str | None = None,
    ) -> ChatResponse:
        """Process a user message and return a response."""
        start_time = time.time()
        tools_used: list[ToolCall] = []

        try:
            # Get or create session
            session_id, session = self.sessions.get_or_create_session(session_id)

            # Add user message to session
            self.sessions.add_message(
                session_id, MessageRole.USER, message, image_base64
            )

            # Set image for tool executor if provided
            self.tool_executor.set_current_image(image_base64)

            # Build messages for LLM
            messages = self._build_messages(session_id, image_base64)

            # Run ReAct loop
            response_content = await self._react_loop(
                messages, image_base64, tools_used
            )

            # Add assistant response to session
            self.sessions.add_message(
                session_id, MessageRole.ASSISTANT, response_content
            )

            processing_time = (time.time() - start_time) * 1000

            return ChatResponse(
                response=response_content,
                session_id=session_id,
                tools_used=tools_used,
                processing_time_ms=processing_time,
            )

        except LLMError as e:
            logger.error(f"LLM error: {e}")
            return ChatResponse(
                response=FALLBACK_RESPONSE,
                session_id=session_id or "error",
                tools_used=tools_used,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except KisanError as e:
            logger.error(f"Kisan error: {e}")
            return ChatResponse(
                response=f"I encountered an issue: {e.message}\n\n{FALLBACK_RESPONSE}",
                session_id=session_id or "error",
                tools_used=tools_used,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return ChatResponse(
                response=FALLBACK_RESPONSE,
                session_id=session_id or "error",
                tools_used=tools_used,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def _build_messages(
        self,
        session_id: str,
        image_base64: str | None,
    ) -> list[dict[str, Any]]:
        """Build messages list for LLM API call."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history
        history = self.sessions.get_messages_for_llm(session_id)

        # Handle the last message specially if it has an image
        if history and image_base64:
            for msg in history[:-1]:
                messages.append(msg)

            # Add the last user message with image
            last_msg = history[-1]
            if last_msg["role"] == "user":
                # Determine image type
                if image_base64.startswith("/9j/"):
                    media_type = "image/jpeg"
                elif image_base64.startswith("iVBOR"):
                    media_type = "image/png"
                else:
                    media_type = "image/jpeg"

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": last_msg["content"]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                })
            else:
                messages.append(last_msg)
        else:
            messages.extend(history)

        return messages

    async def _react_loop(
        self,
        messages: list[dict],
        image_base64: str | None,
        tools_used: list[ToolCall],
    ) -> str:
        """Run the ReAct loop until completion or max iterations."""
        iteration = 0

        while iteration < self.max_tool_iterations:
            iteration += 1
            logger.debug(f"ReAct iteration {iteration}")

            # Call LLM with tools
            response = await self.llm.chat(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.7,
            )

            content = response.get("content")
            tool_calls = response.get("tool_calls")

            # If no tool calls, return the response
            if not tool_calls:
                fallback = "I'm not sure how to help with that. Could you rephrase?"
                return content or fallback

            # Process tool calls
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })

            for tool_call in tool_calls:
                function = tool_call["function"]
                tool_name = function["name"]

                try:
                    arguments = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                logger.info(f"Calling tool: {tool_name}")

                # Execute the tool
                try:
                    result = await self.tool_executor.execute(tool_name, arguments)
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    result = f"Tool execution failed: {str(e)}"

                # Track tool usage
                tools_used.append(
                    ToolCall(
                        name=tool_name,
                        input=arguments,
                        output=result[:500] if result else None,
                    )
                )

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })

        # If we hit max iterations, get a final response
        logger.warning("Max tool iterations reached, getting final response")
        response = await self.llm.chat(messages=messages, temperature=0.7)
        return response.get("content") or "I apologize, I had trouble processing that request."
