"""OpenAI LLM service wrapper."""

from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from kisan.core.config import Settings
from kisan.core.exceptions import EmbeddingError, LLMError, VisionError
from kisan.core.logging import logger


class LLMService:
    """Service for interacting with OpenAI LLMs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.chat_model = settings.chat_model
        self.embedding_model = settings.embedding_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        model: str | None = None,
        response_format: dict | None = None,
    ) -> dict:
        """Send a chat completion request."""
        try:
            kwargs = {
                "model": model or self.chat_model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

            response = await self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            result = {
                "content": message.content,
                "role": message.role,
                "tool_calls": None,
            }

            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]

            return result

        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            raise LLMError(f"Failed to get chat response: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def chat_with_vision(
        self,
        text_prompt: str,
        image_base64: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> str:
        """Send a chat completion request with an image."""
        try:
            # Determine image type from base64 header or default to jpeg
            if image_base64.startswith("/9j/"):
                media_type = "image/jpeg"
            elif image_base64.startswith("iVBOR"):
                media_type = "image/png"
            else:
                media_type = "image/jpeg"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            })

            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Vision API error: {e}")
            raise VisionError(f"Failed to analyze image: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0]
