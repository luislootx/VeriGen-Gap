"""Shortest Path problem — a P problem (Dijkstra) for baseline."""

import heapq
import random
from typing import Any

from .base import BaseProblem, ProblemInstance


class ShortestPathProblem(BaseProblem):
    """Shortest Path: Given a weighted directed graph, find the shortest path
    between two vertices. In P (Dijkstra's algorithm, O(E log V))."""

    name = "Shortest-Path"
    complexity_class = "P"

    def generate_instance(self, size: int, seed: int = None) -> ProblemInstance:
        """Generate a connected weighted graph with `size` vertices."""
        rng = random.Random(seed)
        n = size

        # Build a connected graph: first a spanning path, then random edges
        edges = {}
        for i in range(n - 1):
            w = rng.randint(1, 20)
            edges[(i, i + 1)] = w

        # Add ~n extra random edges
        for _ in range(n):
            u, v = rng.sample(range(n), 2)
            if (u, v) not in edges:
                edges[(u, v)] = rng.randint(1, 20)

        source, target = 0, n - 1

        # Dijkstra
        adj = [[] for _ in range(n)]
        for (u, v), w in edges.items():
            adj[u].append((v, w))

        dist = [float('inf')] * n
        prev = [-1] * n
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))

        # Reconstruct path
        path = []
        node = target
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()

        edges_str = "\n".join(f"  {u} -> {v} (weight {w})" for (u, v), w in sorted(edges.items()))
        gen_prompt = (
            f"Find the shortest path in this weighted directed graph.\n\n"
            f"Graph ({n} vertices, {len(edges)} edges):\n{edges_str}\n\n"
            f"Find the shortest path from vertex {source} to vertex {target}.\n\n"
            f"Reply with ONLY the path and total distance in this format:\n"
            f"Path: 0, 2, 4, 5\nDistance: 42"
        )

        return ProblemInstance(
            problem_type=self.name,
            complexity_class=self.complexity_class,
            size=n,
            instance_data={
                "edges": {f"{u},{v}": w for (u, v), w in edges.items()},
                "source": source,
                "target": target,
                "optimal_distance": dist[target],
                "n_vertices": n,
            },
            solution={"path": path, "distance": dist[target]},
            generation_prompt=gen_prompt,
            verification_prompt="",
        )

    def make_verification_prompt(self, instance: ProblemInstance, candidate: dict) -> str:
        edges = instance.instance_data["edges"]
        n = instance.instance_data["n_vertices"]
        source = instance.instance_data["source"]
        target = instance.instance_data["target"]
        opt_dist = instance.instance_data["optimal_distance"]

        edges_str = "\n".join(
            f"  {k.replace(',', ' ->')} (weight {v})" for k, v in sorted(edges.items())
        )
        path_str = " -> ".join(str(x) for x in candidate.get("path", []))
        claimed_dist = candidate.get("distance", "?")

        return (
            f"Verify this shortest path solution.\n\n"
            f"Graph ({n} vertices):\n{edges_str}\n\n"
            f"Source: {source}, Target: {target}\n"
            f"Proposed path: {path_str}\n"
            f"Claimed distance: {claimed_dist}\n\n"
            f"Check: (1) Is the path valid (all edges exist)? (2) Does the distance match? "
            f"(3) Is it the shortest possible path?\n\n"
            f"Your response must start with exactly 'YES' or 'NO' on the first line, "
            f"then optionally explain."
        )

    def verify_solution(self, instance: ProblemInstance, candidate: Any) -> bool:
        if not isinstance(candidate, dict):
            return False
        path = candidate.get("path", [])
        edges = instance.instance_data["edges"]
        source = instance.instance_data["source"]
        target = instance.instance_data["target"]

        if not path or path[0] != source or path[-1] != target:
            return False

        total = 0
        for i in range(len(path) - 1):
            key = f"{path[i]},{path[i+1]}"
            if key not in edges:
                return False
            total += edges[key]

        return total <= instance.instance_data["optimal_distance"] + 0.001

    def parse_generation_response(self, response: str, instance: ProblemInstance) -> Any:
        try:
            import re
            path_match = re.search(r'[Pp]ath[:\s]*([\d\s,->]+)', response)
            dist_match = re.search(r'[Dd]istance[:\s]*(\d+(?:\.\d+)?)', response)

            if path_match:
                nums = re.findall(r'\d+', path_match.group(1))
                path = [int(x) for x in nums]
            else:
                nums = re.findall(r'\d+', response)
                path = [int(x) for x in nums[:-1]] if dist_match else [int(x) for x in nums]

            distance = float(dist_match.group(1)) if dist_match else None
            return {"path": path, "distance": distance}
        except Exception:
            return None

    def parse_verification_response(self, response: str) -> bool | None:
        return self.robust_parse_yes_no(response)

    def make_wrong_solution(self, instance: ProblemInstance, seed: int = None) -> dict:
        rng = random.Random(seed)
        path = list(instance.solution["path"])
        # Add a detour
        if len(path) >= 3:
            i = rng.randint(1, len(path) - 2)
            path.insert(i, rng.randint(0, instance.instance_data["n_vertices"] - 1))
        wrong_dist = instance.solution["distance"] + rng.randint(5, 20)
        return {"path": path, "distance": wrong_dist}
