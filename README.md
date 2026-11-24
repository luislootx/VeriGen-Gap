# The VeriGen Gap

Data, prompts and code for the paper:

> L. Loo, "The VeriGen Gap: Chance-Corrected Evidence of Verification-Generation Asymmetry in LLMs Across Computational Complexity Classes," accepted at the IEEE Symposium Series on Computational Intelligence (SSCI 2027), IEEE Symposia on CI in Natural Language Processing and Social Media.

Three LLMs (GPT-4o-mini, Claude Haiku 4.5, Llama 3.1 8B) generate and verify solutions of the same instances of six combinatorial problems: Sorting, Shortest Path and Bipartite Matching (in P), and 3-SAT, TSP Decision and Graph 3-Coloring (NP-complete), for nine sizes between 5 and 50.

## Contents

| Path | What it is |
|---|---|
| `data/responses.jsonl.gz` | The 14,580 logged responses (4,860 generation, 9,720 verification) |
| `data/prompts.jsonl.gz` | The 2,700 prompts (540 generation, 540 with the reference solution, 1,620 with an incorrect certificate) and their certificates, regenerated from the seeds |
| `data/excluded_certificates.csv` | The seven "incorrect" certificates that turned out to be valid |
| `src/` | Instance generators, checkers, parsers, API clients and the runner that collected the data |
| `run_haiku_missing.py` | Second run, for Claude Haiku on TSP and 3-Coloring |
| `analysis/rescore.py` | The two scoring rules (as-prompted, final statement) and the certificate audit |
| `analysis/camera_ready_analysis.py` | Every number, table and figure of the paper |
| `analysis/export_prompts.py` | Regenerates `data/prompts.jsonl.gz` and `data/excluded_certificates.csv` |
| `results/results.json`, `figures/` | Output of the analysis |

## Reproducing the paper

```bash
pip install numpy matplotlib
python analysis/camera_ready_analysis.py data/responses.jsonl.gz out
```

This needs no API key. It regenerates all instances and certificates from their seeds (TSP with ten cities is solved by brute force, so it takes a few minutes), re-scores every response and writes `results.json` and the three figures to `out/`. The bootstrap uses a fixed seed.

To collect new data, install `requirements.txt`, set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in a `.env` file, pull `llama3.1:8b` with Ollama, and run `python -m src.experiments.runner --config configs/default.yaml`.

## Record format (`responses.jsonl.gz`)

`problem`, `complexity_class`, `size`, `instance_idx` (0 to 9), `trial` (0 to 2), `model`, `task` (`generation` or `verification`), `correct` and `parsed_successfully` (as-prompted rule, as logged), `raw_response`, `latency_ms`, `prompt_tokens`, `completion_tokens`. Verification records add `solution_is_correct` (label of the certificate), `llm_answer` (parsed verdict) and `expected_answer`.

Models: `gpt-4o-mini` (API alias; the snapshot that answered was not logged), `claude-haiku-4-5-20251001` and `llama3.1:8b` (Ollama). Queries were made on 22 and 23 March 2026 at temperature 0 with at most 1,024 output tokens.

## Known issues of the data

* Seven of the 1,620 incorrect certificates (four for 3-SAT, three for TSP with five cities) satisfy the instance. The analysis excludes the 21 responses that judged them.
* Both matching prompts state the size of the maximum matching: the verification prompt in its checks and the generation prompt in the last line of its answer format.
* Incorrect certificates for TSP and 3-Coloring are random permutations and random colorings, which are easier to reject than small edits.
* The `correct` field of generation records follows the as-prompted rule, which misses answers that come after reasoning, and for Shortest-Path ignores the stated length. Use `analysis/rescore.py` for the final-statement rule.
* Responses with `completion_tokens` equal to 1,024 were cut off.
* The records of Claude Haiku on TSP and 3-Coloring come from a second run on 23 March; 45 TSP records (n = 5) present in both runs were deduplicated.

## License

MIT. See `LICENSE`.
