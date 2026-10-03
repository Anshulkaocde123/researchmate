"""Google Gemini LLM generator using official google-genai SDK."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from researchmate.config import settings

log = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    """Structured response from LLM generation."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw_response: Any | None = Field(default=None, exclude=True)


class GeminiGenerator:
    """Wrapper around Google Gemini client with retry logic and token tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name or settings.llm_model
        self._client = client

    @property
    def client(self) -> Any:
        """Lazy load Google Gemini client."""
        if self._client is None:
            key = self.api_key or settings.get_gemini_api_key()
            from google import genai

            self._client = genai.Client(api_key=key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(
        self,
        user_prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = 1024,
    ) -> GenerationResult:
        """Call Gemini to generate content.

        Args:
            user_prompt: Main query with context.
            system_instruction: System prompt enforcing behavior and citations.
            temperature: Sampling temperature (0.0 for deterministic answers).
            max_output_tokens: Maximum tokens in response.

        Returns:
            GenerationResult containing generated text and token usage.
        """
        from google.genai import types

        config_args: dict[str, Any] = {
            "temperature": temperature,
        }
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if max_output_tokens is not None:
            config_args["max_output_tokens"] = max_output_tokens

        config = types.GenerateContentConfig(**config_args)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=config,
        )

        text = response.text or ""
        prompt_tokens = None
        completion_tokens = None

        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

        return GenerationResult(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_response=response,
        )
