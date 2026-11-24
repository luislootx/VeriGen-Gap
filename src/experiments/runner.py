"""Main experiment runner for the VeriGen-Gap study."""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from ..problems import ALL_PROBLEMS
from ..llm import OpenAIClient, AnthropicClient, OllamaClient


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_all_sizes(config: dict) -> list[int]:
    sizes = []
    for group in config["sizes"].values():
        sizes.extend(group)
    return sorted(set(sizes))


def init_clients(config: dict) -> list:
    """Initialize available LLM clients."""
    load_dotenv()
    clients = []

    if "openai" in config.get("models", {}) and os.environ.get("OPENAI_API_KEY"):
        cfg = config["models"]["openai"]
        clients.append(OpenAIClient(model=cfg["model_id"]))
        print(f"  [OK] OpenAI: {cfg['model_id']}")

    if "anthropic" in config.get("models", {}) and os.environ.get("ANTHROPIC_API_KEY"):
        cfg = config["models"]["anthropic"]
        clients.append(AnthropicClient(model=cfg["model_id"]))
        print(f"  [OK] Anthropic: {cfg['model_id']}")

    if "ollama" in config.get("models", {}):
        try:
            cfg = config["models"]["ollama"]
            client = OllamaClient(model=cfg["model_id"])
            if client.is_available():
                clients.append(client)
                print(f"  [OK] Ollama: {cfg['model_id']}")
            else:
                print(f"  [SKIP] Ollama: {cfg['model_id']} not found")
        except Exception:
            print("  [SKIP] Ollama: not reachable")

    return clients


def run_generation_experiment(problem, instance, client, config):
    """Ask the LLM to generate a solution for the given instance."""
    try:
        response = client.query(
            instance.generation_prompt,
            max_tokens=config.get("max_tokens", 1024),
            temperature=config.get("temperature", 0.0),
        )
        candidate = problem.parse_generation_response(response.text, instance)
        is_correct = problem.verify_solution(instance, candidate) if candidate is not None else False

        return {
            "task": "generation",
            "correct": is_correct,
            "parsed_successfully": candidate is not None,
            "raw_response": response.text,
            "latency_ms": response.latency_ms,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }
    except Exception as e:
        return {
            "task": "generation",
            "correct": False,
            "parsed_successfully": False,
            "raw_response": "",
            "error": str(e),
            "latency_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


def run_verification_experiment(problem, instance, client, config, use_correct: bool, seed: int):
    """Ask the LLM to verify a (correct or incorrect) solution."""
    if use_correct:
        candidate = instance.solution
    else:
        candidate = problem.make_wrong_solution(instance, seed=seed)

    prompt = problem.make_verification_prompt(instance, candidate)

    try:
        response = client.query(
            prompt,
            max_tokens=config.get("max_tokens", 256),
            temperature=config.get("temperature", 0.0),
        )
        llm_says = problem.parse_verification_response(response.text)

        # Correct answer: YES if solution is correct, NO if incorrect
        expected = use_correct
        is_correct = (llm_says == expected) if llm_says is not None else False

        return {
            "task": "verification",
            "solution_is_correct": use_correct,
            "llm_answer": llm_says,
            "expected_answer": expected,
            "correct": is_correct,
            "parsed_successfully": llm_says is not None,
            "raw_response": response.text,
            "latency_ms": response.latency_ms,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }
    except Exception as e:
        return {
            "task": "verification",
            "solution_is_correct": use_correct,
            "correct": False,
            "parsed_successfully": False,
            "error": str(e),
            "latency_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


def run_experiments(config_path: str = "configs/default.yaml"):
    """Run the full VeriGen-Gap experiment suite."""
    config = load_config(config_path)
    sizes = get_all_sizes(config)
    n_instances = config.get("instances_per_size", 10)
    n_trials = config.get("trials_per_instance", 3)

    print("=" * 60)
    print("VeriGen-Gap Experiment Runner")
    print("=" * 60)

    print("\nInitializing LLM clients...")
    clients = init_clients(config)
    if not clients:
        print("ERROR: No LLM clients available. Check API keys and Ollama.")
        return

    print(f"\nClients: {[c.name for c in clients]}")
    print(f"Problems: {[p.name for p in ALL_PROBLEMS]}")
    print(f"Sizes: {sizes}")
    print(f"Instances per size: {n_instances}, Trials: {n_trials}")

    # Setup results directory
    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"experiment_{timestamp}.jsonl"

    total_queries = (
        len(ALL_PROBLEMS) * len(sizes) * n_instances * len(clients) * n_trials
        * 3  # generation + verification_correct + verification_wrong
    )
    print(f"\nTotal estimated API queries: {total_queries}")
    print(f"Results file: {results_file}\n")

    results = []
    with open(results_file, "w") as f:
        for problem_cls in ALL_PROBLEMS:
            problem = problem_cls()
            print(f"\n{'-' * 40}")
            print(f"Problem: {problem.name} ({problem.complexity_class})")

            for size in tqdm(sizes, desc=f"  {problem.name}"):
                for inst_idx in range(n_instances):
                    seed = size * 1000 + inst_idx
                    instance = problem.generate_instance(size, seed=seed)

                    for client in clients:
                        for trial in range(n_trials):
                            # 1. Generation experiment
                            gen_result = run_generation_experiment(
                                problem, instance, client, config
                            )
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
                            results.append(record)

                            # 2. Verification with correct solution
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
                            results.append(record)

                            # 3. Verification with wrong solution
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
                            results.append(record)

                            f.flush()

    print(f"\n{'=' * 60}")
    print(f"Done! {len(results)} records saved to {results_file}")
    return results_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_experiments(args.config)
