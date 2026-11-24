"""Base class for all problem types in the VeriGen-Gap study."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProblemInstance:
    """A single problem instance with its known solution."""
    problem_type: str
    complexity_class: str  # "P" or "NP-complete"
    size: int  # n parameter
    instance_data: dict  # problem-specific data
    solution: Any  # known correct solution
    generation_prompt: str  # prompt asking LLM to solve
    verification_prompt: str  # prompt asking LLM to verify a candidate


class BaseProblem(ABC):
    """Abstract base for problem generators."""

    name: str = ""
    complexity_class: str = ""  # "P" or "NP-complete"

    @abstractmethod
    def generate_instance(self, size: int, seed: int = None) -> ProblemInstance:
        """Generate a random instance of the given size with a known solution."""
        ...

    @abstractmethod
    def verify_solution(self, instance: ProblemInstance, candidate: Any) -> bool:
        """Programmatically verify if a candidate solution is correct."""
        ...

    @abstractmethod
    def parse_generation_response(self, response: str, instance: ProblemInstance) -> Any:
        """Parse LLM's generation response into a candidate solution."""
        ...

    @abstractmethod
    def parse_verification_response(self, response: str) -> bool:
        """Parse LLM's verification response into True/False."""
        ...

    @staticmethod
    def robust_parse_yes_no(response: str) -> bool | None:
        """Robustly parse a YES/NO response from verbose LLM output.
        Checks last line first, then scans for conclusive keywords."""
        resp = response.strip()
        # Check first line (our prompts ask for YES/NO on line 1)
        first_line = resp.split("\n")[0].strip().upper().rstrip(".,!*")
        if first_line in ("YES", "**YES**"):
            return True
        if first_line in ("NO", "**NO**"):
            return False

        # Check last line (LLMs sometimes conclude with the answer)
        last_line = resp.split("\n")[-1].strip().upper().rstrip(".,!*")
        if last_line in ("YES", "**YES**"):
            return True
        if last_line in ("NO", "**NO**"):
            return False

        resp_upper = resp.upper()
        # Look for strong conclusive phrases
        positive = ["THE ANSWER IS YES", "IS VALID", "IS CORRECT", "SATISFIES ALL",
                     "ALL CLAUSES ARE SATISFIED", "YES, THE", "YES,"]
        negative = ["THE ANSWER IS NO", "IS NOT VALID", "IS INCORRECT", "IS INVALID",
                     "DOES NOT SATISFY", "NOT ALL CLAUSES", "NO, THE", "NO,"]

        pos_found = any(p in resp_upper for p in positive)
        neg_found = any(n in resp_upper for n in negative)

        if pos_found and not neg_found:
            return True
        if neg_found and not pos_found:
            return False

        # Fallback: count YES vs NO occurrences
        yes_count = resp_upper.count("YES")
        no_count = resp_upper.count("NO")
        if yes_count > 0 and no_count == 0:
            return True
        if no_count > 0 and yes_count == 0:
            return False

        return None  # Truly ambiguous

    def make_wrong_solution(self, instance: ProblemInstance, seed: int = None) -> Any:
        """Generate an incorrect solution for verification experiments.
        Half of verification prompts should have wrong solutions."""
        ...
