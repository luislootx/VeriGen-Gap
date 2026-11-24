"""Run only Claude Haiku on TSP-Decision and Graph-3-Coloring (missing data)."""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from src.problems.tsp import TSPDecisionProblem
from src.problems.graph_coloring import GraphColoringProblem
from src.llm.anthropic_client import AnthropicClient
from src.experiments.runner import run_generation_experiment, run_verification_experiment

config = {
    "max_tokens": 1024,
    "temperature": 0.0,
}

sizes = [5, 8, 10, 15, 20, 25, 30, 40, 50]
n_instances = 10
n_trials = 3

client = AnthropicClient(model="claude-haiku-4-5-20251001")
print(f"Testing Anthropic... ", end="")
try:
    r = client.query("Reply OK", max_tokens=10)
    print(f"OK! ({r.latency_ms:.0f}ms)")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

problems = [TSPDecisionProblem(), GraphColoringProblem()]
output_file = "results/haiku_missing_20260323.jsonl"

total = len(problems) * len(sizes) * n_instances * n_trials * 3
print(f"\nRunning {total} queries for Haiku on TSP + Graph-Coloring")
print(f"Output: {output_file}\n")

count = 0
with open(output_file, "w") as f:
    for problem in problems:
        print(f"--- {problem.name} ({problem.complexity_class}) ---")
        for size in sizes:
            for inst_idx in range(n_instances):
                seed = size * 1000 + inst_idx
                instance = problem.generate_instance(size, seed=seed)

                for trial in range(n_trials):
                    # Generation
                    gen_result = run_generation_experiment(problem, instance, client, config)
                    record = {
                        "problem": problem.name,
                        "complexity_class": problem.complexity_class,
                        "size": size,
                        "instance_idx": inst_idx,
                        "trial": trial,
                        "model": client.name,
                        **gen_result,
                    }
                    f.write(json.dumps(record) + "\n")
                    count += 1

                    # Verification correct
                    ver_correct = run_verification_experiment(
                        problem, instance, client, config,
                        use_correct=True, seed=seed + trial,
                    )
                    record = {
                        "problem": problem.name,
                        "complexity_class": problem.complexity_class,
                        "size": size,
                        "instance_idx": inst_idx,
                        "trial": trial,
                        "model": client.name,
                        **ver_correct,
                    }
                    f.write(json.dumps(record) + "\n")
                    count += 1

                    # Verification wrong
                    ver_wrong = run_verification_experiment(
                        problem, instance, client, config,
                        use_correct=False, seed=seed + trial + 10000,
                    )
                    record = {
                        "problem": problem.name,
                        "complexity_class": problem.complexity_class,
                        "size": size,
                        "instance_idx": inst_idx,
                        "trial": trial,
                        "model": client.name,
                        **ver_wrong,
                    }
                    f.write(json.dumps(record) + "\n")
                    count += 1
                    f.flush()

            print(f"  size={size} done ({count}/{total})")

print(f"\nDone! {count} records saved to {output_file}")
