"""Anthropic Claude client."""

import os
import time

import anthropic

from .base import BaseLLMClient, LLMResponse


class AnthropicClient(BaseLLMClient):
    name = "Claude-Sonnet"
    model_id = "claude-sonnet-4-6"

    def __init__(self, model: str = None):
        if model:
            self.model_id = model
            self.name = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def query(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=response.content[0].text,
            model=self.model_id,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            latency_ms=elapsed,
        )
