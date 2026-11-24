"""Ollama (local Llama 3) client."""

import json
import time
import urllib.request

from .base import BaseLLMClient, LLMResponse


class OllamaClient(BaseLLMClient):
    name = "Llama-3"
    model_id = "llama3.1:8b"

    def __init__(self, model: str = None, base_url: str = "http://localhost:11434"):
        if model:
            self.model_id = model
            self.name = model
        self.base_url = base_url

    def query(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        start = time.perf_counter()

        payload = json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        elapsed = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=self.model_id,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=elapsed,
        )

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return any(self.model_id in m for m in models)
        except Exception:
            return False
