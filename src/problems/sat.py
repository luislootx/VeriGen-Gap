"""3-SAT problem generator and verifier."""

import random
from typing import Any

from .base import BaseProblem, ProblemInstance


class SAT3Problem(BaseProblem):
    """3-SAT: Given a boolean formula in CNF with exactly 3 literals per clause,
    find a satisfying assignment (generation) or verify one (verification).

    NP-complete. We generate satisfiable instances by planting a solution."""

    name = "3-SAT"
    complexity_class = "NP-complete"

    def generate_instance(self, size: int, seed: int = None) -> ProblemInstance:
        """Generate a satisfiable 3-SAT instance with `size` variables.
        Uses clause-to-variable ratio ~4.26 (near phase transition)."""
        rng = random.Random(seed)
        n_vars = size
        n_clauses = int(4.26 * n_vars)

        # Plant a satisfying assignment
        planted = [rng.choice([True, False]) for _ in range(n_vars)]

        clauses = []
        for _ in range(n_clauses):
            # Pick 3 distinct variables
            vars_idx = rng.sample(range(n_vars), 3)
            # Generate clause that is satisfied by planted assignment
            clause = []
            for v in vars_idx:
                if rng.random() < 0.5:
                    clause.append(v + 1 if planted[v] else -(v + 1))
                else:
                    clause.append(-(v + 1) if planted[v] else v + 1)

            # Ensure at least one literal is satisfied by planted
            satisfied = False
            for lit in clause:
                var = abs(lit) - 1
                val = planted[var]
                if (lit > 0 and val) or (lit < 0 and not val):
                    satisfied = True
                    break
            if not satisfied:
                # Flip one literal to make it satisfied
                idx = rng.randint(0, 2)
                clause[idx] = -clause[idx]

            clauses.append(clause)

        # Format for display
        clause_strs = []
        for c in clauses:
            lits = []
            for lit in c:
                if lit > 0:
                    lits.append(f"x{lit}")
                else:
                    lits.append(f"~x{-lit}")
            clause_strs.append(f"({' OR '.join(lits)})")
        formula_str = " AND ".join(clause_strs)

        gen_prompt = (
            f"Solve this 3-SAT instance. There are {n_vars} boolean variables (x1 to x{n_vars}) "
            f"and {n_clauses} clauses.\n\n"
            f"Formula: {formula_str}\n\n"
            f"Find a satisfying assignment. Reply with ONLY the assignment in this exact format:\n"
            f"x1=T, x2=F, x3=T, ... (T for True, F for False)"
        )

        # Format planted solution for verification prompt
        planted_str = ", ".join(
            f"x{i+1}={'T' if v else 'F'}" for i, v in enumerate(planted)
        )

        return ProblemInstance(
            problem_type=self.name,
            complexity_class=self.complexity_class,
            size=n_vars,
            instance_data={"clauses": clauses, "n_vars": n_vars, "formula_str": formula_str},
            solution=planted,
            generation_prompt=gen_prompt,
            verification_prompt="",  # Set dynamically in experiment
        )

    def make_verification_prompt(self, instance: ProblemInstance, candidate: list[bool]) -> str:
        """Create a verification prompt for a given candidate solution."""
        n_vars = instance.instance_data["n_vars"]
        formula_str = instance.instance_data["formula_str"]
        cand_str = ", ".join(
            f"x{i+1}={'T' if v else 'F'}" for i, v in enumerate(candidate)
        )
        return (
            f"Verify if this assignment satisfies the 3-SAT formula.\n\n"
            f"Formula ({n_vars} variables, {len(instance.instance_data['clauses'])} clauses):\n"
            f"{formula_str}\n\n"
            f"Assignment: {cand_str}\n\n"
            f"Your response must start with exactly 'YES' or 'NO' on the first line, "
            f"then optionally explain. YES if ALL clauses are satisfied, NO otherwise."
        )

    def verify_solution(self, instance: ProblemInstance, candidate: Any) -> bool:
        """Programmatically verify a candidate assignment."""
        if not isinstance(candidate, list) or len(candidate) != instance.instance_data["n_vars"]:
            return False
        for clause in instance.instance_data["clauses"]:
            satisfied = False
            for lit in clause:
                var = abs(lit) - 1
                val = candidate[var]
                if (lit > 0 and val) or (lit < 0 and not val):
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True

    def parse_generation_response(self, response: str, instance: ProblemInstance) -> Any:
        """Parse 'x1=T, x2=F, ...' into list of bools."""
        try:
            n = instance.instance_data["n_vars"]
            assignment = [None] * n
            # Find all xi=T/F patterns
            import re
            matches = re.findall(r'x(\d+)\s*=\s*([TF])', response, re.IGNORECASE)
            for var_str, val_str in matches:
                idx = int(var_str) - 1
                if 0 <= idx < n:
                    assignment[idx] = val_str.upper() == 'T'
            if None in assignment:
                return None
            return assignment
        except Exception:
            return None

    def parse_verification_response(self, response: str) -> bool | None:
        return self.robust_parse_yes_no(response)

    def make_wrong_solution(self, instance: ProblemInstance, seed: int = None) -> list[bool]:
        """Flip some variables from the planted solution to create a wrong one."""
        rng = random.Random(seed)
        wrong = list(instance.solution)
        # Flip 1-3 variables
        n_flips = rng.randint(1, min(3, len(wrong)))
        for idx in rng.sample(range(len(wrong)), n_flips):
            wrong[idx] = not wrong[idx]
        # Make sure it's actually wrong
        if self.verify_solution(instance, wrong):
            wrong[0] = not wrong[0]
        return wrong
