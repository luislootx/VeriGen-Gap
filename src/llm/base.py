"""Base class for LLM client interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class BaseLLMClient(ABC):
    """Abstract base for LLM API clients."""

    name: str = ""
    model_id: str = ""

    @abstractmethod
    def query(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        """Send a prompt and return the response."""
        ...

    def is_available(self) -> bool:
        """Check if this client is configured and reachable."""
        try:
            resp = self.query("Reply with only the word 'OK'.", max_tokens=10)
            return "OK" in resp.text.upper()
        except Exception:
            return False
