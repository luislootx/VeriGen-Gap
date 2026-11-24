"""OpenAI GPT-4 client."""

import os
import time

from openai import OpenAI

from .base import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    name = "GPT-4o"
    model_id = "gpt-4o"

    def __init__(self, model: str = None):
        if model:
            self.model_id = model
            self.name = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def query(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model_id,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=elapsed,
        )
