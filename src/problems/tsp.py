"""TSP Decision Problem generator and verifier."""

import itertools
import math
import random
from typing import Any

from .base import BaseProblem, ProblemInstance


class TSPDecisionProblem(BaseProblem):
    """TSP Decision: Given cities with coordinates and a distance threshold k,
    is there a tour visiting all cities with total distance <= k?

    NP-complete. We generate instances with a known optimal tour."""

    name = "TSP-Decision"
    complexity_class = "NP-complete"

    def generate_instance(self, size: int, seed: int = None) -> ProblemInstance:
        """Generate TSP instance with `size` cities in 2D.
        For small n, compute optimal tour. For larger n, use planted tour."""
        rng = random.Random(seed)
        n = size
        cities = [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]

        def tour_distance(tour, coords):
            d = 0
            for i in range(len(tour)):
                x1, y1 = coords[tour[i]]
                x2, y2 = coords[tour[(i + 1) % len(tour)]]
                d += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            return d

        if n <= 10:
            # Brute force optimal
            best_tour = None
            best_dist = float('inf')
            for perm in itertools.permutations(range(1, n)):
                tour = [0] + list(perm)
                d = tour_distance(tour, cities)
                if d < best_dist:
                    best_dist = d
                    best_tour = tour
        else:
            # Nearest-neighbor heuristic as baseline
            visited = [False] * n
            tour = [0]
            visited[0] = True
            for _ in range(n - 1):
                last = tour[-1]
                nearest = -1
                nearest_d = float('inf')
                for j in range(n):
                    if not visited[j]:
                        d = math.sqrt(
                            (cities[last][0] - cities[j][0]) ** 2
                            + (cities[last][1] - cities[j][1]) ** 2
                        )
                        if d < nearest_d:
                            nearest_d = d
                            nearest = j
                tour.append(nearest)
                visited[nearest] = True
            best_tour = tour
            best_dist = tour_distance(tour, cities)

        # Set threshold slightly above optimal so answer is YES
        threshold = round(best_dist * 1.05, 2)

        cities_str = "\n".join(f"  City {i}: ({x}, {y})" for i, (x, y) in enumerate(cities))

        gen_prompt = (
            f"Solve this Traveling Salesman Problem.\n"
            f"There are {n} cities with these coordinates:\n{cities_str}\n\n"
            f"Distance threshold: {threshold}\n\n"
            f"Find a tour (visiting each city exactly once and returning to start) "
            f"with total Euclidean distance <= {threshold}.\n\n"
            f"Reply with ONLY the tour as a comma-separated list of city indices, e.g.: 0, 3, 1, 2"
        )

        return ProblemInstance(
            problem_type=self.name,
            complexity_class=self.complexity_class,
            size=n,
            instance_data={
                "cities": cities,
                "threshold": threshold,
                "optimal_distance": best_dist,
            },
            solution=best_tour,
            generation_prompt=gen_prompt,
            verification_prompt="",
        )

    def make_verification_prompt(self, instance: ProblemInstance, candidate: list[int]) -> str:
        cities = instance.instance_data["cities"]
        threshold = instance.instance_data["threshold"]
        n = len(cities)
        cities_str = "\n".join(f"  City {i}: ({x}, {y})" for i, (x, y) in enumerate(cities))
        tour_str = ", ".join(str(c) for c in candidate)

        return (
            f"Verify this TSP solution.\n\n"
            f"{n} cities:\n{cities_str}\n\n"
            f"Proposed tour: {tour_str}\n"
            f"Distance threshold: {threshold}\n\n"
            f"Check: (1) Does the tour visit each city exactly once? "
            f"(2) Is the total Euclidean distance <= {threshold}?\n\n"
            f"Your response must start with exactly 'YES' or 'NO' on the first line, "
            f"then optionally explain."
        )

    def verify_solution(self, instance: ProblemInstance, candidate: Any) -> bool:
        if not isinstance(candidate, list):
            return False
        cities = instance.instance_data["cities"]
        n = len(cities)
        if len(candidate) != n or set(candidate) != set(range(n)):
            return False
        d = 0
        for i in range(n):
            x1, y1 = cities[candidate[i]]
            x2, y2 = cities[candidate[(i + 1) % n]]
            d += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return d <= instance.instance_data["threshold"]

    def parse_generation_response(self, response: str, instance: ProblemInstance) -> Any:
        try:
            import re
            numbers = re.findall(r'\d+', response)
            tour = [int(x) for x in numbers]
            n = len(instance.instance_data["cities"])
            if len(tour) >= n:
                tour = tour[:n]
            return tour
        except Exception:
            return None

    def parse_verification_response(self, response: str) -> bool | None:
        return self.robust_parse_yes_no(response)

    def make_wrong_solution(self, instance: ProblemInstance, seed: int = None) -> list[int]:
        """Shuffle the tour to make it likely exceed threshold."""
        rng = random.Random(seed)
        n = len(instance.instance_data["cities"])
        wrong = list(range(n))
        rng.shuffle(wrong)
        # If accidentally correct, swap two cities
        if self.verify_solution(instance, wrong):
            wrong[1], wrong[2] = wrong[2], wrong[1]
        return wrong
