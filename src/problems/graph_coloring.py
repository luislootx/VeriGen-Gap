"""Graph 3-Coloring problem generator and verifier."""

import random
from typing import Any

from .base import BaseProblem, ProblemInstance


class GraphColoringProblem(BaseProblem):
    """Graph 3-Coloring: Given an undirected graph, assign one of 3 colors to each
    vertex such that no two adjacent vertices share the same color.

    NP-complete. We generate 3-colorable graphs by planting a coloring."""

    name = "Graph-3-Coloring"
    complexity_class = "NP-complete"

    def generate_instance(self, size: int, seed: int = None) -> ProblemInstance:
        """Generate a 3-colorable graph with `size` vertices."""
        rng = random.Random(seed)
        n = size
        colors = ["R", "G", "B"]

        # Plant a valid 3-coloring
        planted = [rng.choice(colors) for _ in range(n)]

        # Generate edges between differently-colored vertices
        # Edge density ~ 2*n edges (sparse enough to be interesting)
        max_edges = min(2 * n, n * (n - 1) // 2)
        edges = set()
        attempts = 0
        while len(edges) < max_edges and attempts < max_edges * 10:
            u, v = rng.sample(range(n), 2)
            if u > v:
                u, v = v, u
            if planted[u] != planted[v] and (u, v) not in edges:
                edges.add((u, v))
            attempts += 1

        edges = sorted(edges)

        edges_str = "\n".join(f"  {u} -- {v}" for u, v in edges)
        gen_prompt = (
            f"Solve this Graph 3-Coloring problem.\n"
            f"Graph has {n} vertices (0 to {n-1}) and {len(edges)} edges:\n{edges_str}\n\n"
            f"Assign one of three colors (R, G, B) to each vertex such that "
            f"no two adjacent vertices share the same color.\n\n"
            f"Reply with ONLY the coloring in this format:\n"
            f"0=R, 1=G, 2=B, ..."
        )

        return ProblemInstance(
            problem_type=self.name,
            complexity_class=self.complexity_class,
            size=n,
            instance_data={"edges": edges, "n_vertices": n},
            solution=planted,
            generation_prompt=gen_prompt,
            verification_prompt="",
        )

    def make_verification_prompt(self, instance: ProblemInstance, candidate: list[str]) -> str:
        n = instance.instance_data["n_vertices"]
        edges = instance.instance_data["edges"]
        edges_str = "\n".join(f"  {u} -- {v}" for u, v in edges)
        coloring_str = ", ".join(f"{i}={c}" for i, c in enumerate(candidate))

        return (
            f"Verify this graph 3-coloring.\n\n"
            f"Graph: {n} vertices, {len(edges)} edges:\n{edges_str}\n\n"
            f"Proposed coloring: {coloring_str}\n\n"
            f"Check: does any edge connect two vertices of the same color?\n\n"
            f"Your response must start with exactly 'YES' or 'NO' on the first line, "
            f"then optionally explain. YES if valid (no conflicts), NO if invalid."
        )

    def verify_solution(self, instance: ProblemInstance, candidate: Any) -> bool:
        if not isinstance(candidate, list):
            return False
        n = instance.instance_data["n_vertices"]
        if len(candidate) != n:
            return False
        valid_colors = {"R", "G", "B"}
        if not all(c in valid_colors for c in candidate):
            return False
        for u, v in instance.instance_data["edges"]:
            if candidate[u] == candidate[v]:
                return False
        return True

    def parse_generation_response(self, response: str, instance: ProblemInstance) -> Any:
        try:
            import re
            n = instance.instance_data["n_vertices"]
            coloring = [None] * n
            matches = re.findall(r'(\d+)\s*=\s*([RGB])', response, re.IGNORECASE)
            for idx_str, color in matches:
                idx = int(idx_str)
                if 0 <= idx < n:
                    coloring[idx] = color.upper()
            if None in coloring:
                return None
            return coloring
        except Exception:
            return None

    def parse_verification_response(self, response: str) -> bool | None:
        return self.robust_parse_yes_no(response)

    def make_wrong_solution(self, instance: ProblemInstance, seed: int = None) -> list[str]:
        rng = random.Random(seed)
        n = instance.instance_data["n_vertices"]
        colors = ["R", "G", "B"]
        wrong = [rng.choice(colors) for _ in range(n)]
        # Likely invalid for a graph with enough edges
        if self.verify_solution(instance, wrong):
            # Force a conflict on the first edge
            u, v = instance.instance_data["edges"][0]
            wrong[v] = wrong[u]
        return wrong
