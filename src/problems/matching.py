"""Bipartite Maximum Matching — a P problem (Hopcroft-Karp) for baseline."""

import random
from typing import Any

from .base import BaseProblem, ProblemInstance


class BipartiteMatchingProblem(BaseProblem):
    """Bipartite Maximum Matching: Given a bipartite graph, find a maximum matching.
    In P (Hopcroft-Karp, O(E*sqrt(V)))."""

    name = "Bipartite-Matching"
    complexity_class = "P"

    def generate_instance(self, size: int, seed: int = None) -> ProblemInstance:
        """Generate a bipartite graph with `size` vertices on each side."""
        rng = random.Random(seed)
        n = size  # vertices per side

        # Generate random edges with ~2n density
        edges = set()
        for _ in range(2 * n):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            edges.add((u, v))
        edges = sorted(edges)

        # Find maximum matching via augmenting paths
        match_l = [-1] * n
        match_r = [-1] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)

        def augment(u, visited):
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    if match_r[v] == -1 or augment(match_r[v], visited):
                        match_l[u] = v
                        match_r[v] = u
                        return True
            return False

        matching_size = 0
        for u in range(n):
            visited = set()
            if augment(u, visited):
                matching_size += 1

        matching = [(u, match_l[u]) for u in range(n) if match_l[u] != -1]

        edges_str = "\n".join(f"  L{u} -- R{v}" for u, v in edges)
        gen_prompt = (
            f"Find a maximum matching in this bipartite graph.\n\n"
            f"Left vertices: L0 to L{n-1}\n"
            f"Right vertices: R0 to R{n-1}\n"
            f"Edges:\n{edges_str}\n\n"
            f"Find the maximum matching (largest set of edges with no shared vertices).\n\n"
            f"Reply with ONLY the matching in this format:\n"
            f"L0-R2, L1-R0, L3-R1\n"
            f"Size: {matching_size}"
        )

        return ProblemInstance(
            problem_type=self.name,
            complexity_class=self.complexity_class,
            size=n,
            instance_data={"edges": edges, "n_per_side": n, "max_matching_size": matching_size},
            solution=matching,
            generation_prompt=gen_prompt,
            verification_prompt="",
        )

    def make_verification_prompt(self, instance: ProblemInstance, candidate: list[tuple]) -> str:
        edges = instance.instance_data["edges"]
        n = instance.instance_data["n_per_side"]
        max_size = instance.instance_data["max_matching_size"]
        edges_str = "\n".join(f"  L{u} -- R{v}" for u, v in edges)
        match_str = ", ".join(f"L{u}-R{v}" for u, v in candidate)

        return (
            f"Verify this bipartite matching.\n\n"
            f"Graph (L0..L{n-1}, R0..R{n-1}):\n{edges_str}\n\n"
            f"Proposed matching: {match_str}\n"
            f"Claimed size: {len(candidate)}\n"
            f"Expected maximum matching size: {max_size}\n\n"
            f"Check: (1) All edges in matching exist in graph? (2) No vertex appears twice? "
            f"(3) Is it maximum (size = {max_size})?\n\n"
            f"Your response must start with exactly 'YES' or 'NO' on the first line, "
            f"then optionally explain."
        )

    def verify_solution(self, instance: ProblemInstance, candidate: Any) -> bool:
        if not isinstance(candidate, list):
            return False
        edges_set = set(map(tuple, instance.instance_data["edges"]))
        n = instance.instance_data["n_per_side"]
        max_size = instance.instance_data["max_matching_size"]

        used_l = set()
        used_r = set()
        for pair in candidate:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return False
            u, v = pair
            if (u, v) not in edges_set:
                return False
            if u in used_l or v in used_r:
                return False
            used_l.add(u)
            used_r.add(v)

        return len(candidate) == max_size

    def parse_generation_response(self, response: str, instance: ProblemInstance) -> Any:
        try:
            import re
            matches = re.findall(r'L(\d+)\s*-\s*R(\d+)', response, re.IGNORECASE)
            return [(int(u), int(v)) for u, v in matches]
        except Exception:
            return None

    def parse_verification_response(self, response: str) -> bool | None:
        return self.robust_parse_yes_no(response)

    def make_wrong_solution(self, instance: ProblemInstance, seed: int = None) -> list[tuple]:
        """Return a sub-optimal matching (fewer edges than maximum)."""
        rng = random.Random(seed)
        matching = list(instance.solution)
        if len(matching) > 1:
            matching.pop(rng.randint(0, len(matching) - 1))
        return matching
