"""Sorting problem — a P problem for baseline comparison."""

import random
from typing import Any

from .base import BaseProblem, ProblemInstance


class SortingProblem(BaseProblem):
    """Sorting: Given a list of integers, produce the sorted list.
    In P (O(n log n)). Used as baseline to contrast with NP-complete problems."""

    name = "Sorting"
    complexity_class = "P"

    def generate_instance(self, size: int, seed: int = None) -> ProblemInstance:
        rng = random.Random(seed)
        arr = [rng.randint(-999, 999) for _ in range(size)]
        solution = sorted(arr)

        arr_str = ", ".join(str(x) for x in arr)
        gen_prompt = (
            f"Sort this list of {size} integers in ascending order.\n\n"
            f"List: [{arr_str}]\n\n"
            f"Reply with ONLY the sorted list in this format: [a, b, c, ...]"
        )

        return ProblemInstance(
            problem_type=self.name,
            complexity_class=self.complexity_class,
            size=size,
            instance_data={"array": arr},
            solution=solution,
            generation_prompt=gen_prompt,
            verification_prompt="",
        )

    def make_verification_prompt(self, instance: ProblemInstance, candidate: list[int]) -> str:
        arr = instance.instance_data["array"]
        arr_str = ", ".join(str(x) for x in arr)
        cand_str = ", ".join(str(x) for x in candidate)
        return (
            f"Verify if this is the correct ascending sort of the given list.\n\n"
            f"Original list: [{arr_str}]\n"
            f"Proposed sorted list: [{cand_str}]\n\n"
            f"Check: (1) same elements? (2) ascending order?\n\n"
            f"Your response must start with exactly 'YES' or 'NO' on the first line, "
            f"then optionally explain."
        )

    def verify_solution(self, instance: ProblemInstance, candidate: Any) -> bool:
        if not isinstance(candidate, list):
            return False
        return candidate == sorted(instance.instance_data["array"])

    def parse_generation_response(self, response: str, instance: ProblemInstance) -> Any:
        try:
            import re
            numbers = re.findall(r'-?\d+', response)
            return [int(x) for x in numbers]
        except Exception:
            return None

    def parse_verification_response(self, response: str) -> bool | None:
        return self.robust_parse_yes_no(response)

    def make_wrong_solution(self, instance: ProblemInstance, seed: int = None) -> list[int]:
        rng = random.Random(seed)
        wrong = sorted(instance.instance_data["array"])
        # Swap two adjacent elements
        if len(wrong) >= 2:
            i = rng.randint(0, len(wrong) - 2)
            wrong[i], wrong[i + 1] = wrong[i + 1], wrong[i]
        return wrong
