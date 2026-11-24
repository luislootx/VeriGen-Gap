"""Regenerates every prompt and certificate from the seeds of the runner.

Usage (from the repository root):  python analysis/export_prompts.py data
Writes prompts.jsonl.gz (2,700 prompts) and excluded_certificates.csv.
"""
import csv
import gzip
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rescore as R  # noqa: E402


def main(out):
    os.makedirs(out, exist_ok=True)
    n_rec = 0
    with gzip.open(os.path.join(out, "prompts.jsonl.gz"), "wt", encoding="utf-8") as f:
        for name, prob in R.PROB.items():
            for n in R.SIZES:
                for i in range(10):
                    inst = R.instance(name, n, i)
                    base = {"problem": name, "complexity_class": prob.complexity_class, "size": n, "instance_idx": i}
                    f.write(json.dumps({**base, "task": "generation", "trial": "all", "prompt": inst.generation_prompt}) + "\n")
                    f.write(json.dumps({**base, "task": "verification", "solution_is_correct": True, "trial": "all",
                                        "certificate": inst.solution, "prompt": prob.make_verification_prompt(inst, inst.solution)}) + "\n")
                    n_rec += 2
                    for t in range(3):
                        w = R.wrong_certificate(name, n, i, t)
                        f.write(json.dumps({**base, "task": "verification", "solution_is_correct": False, "trial": t, "certificate": w,
                                            "certificate_is_in_fact_valid": R.certificate_is_valid(name, n, i, w),
                                            "prompt": prob.make_verification_prompt(inst, w)}) + "\n")
                        n_rec += 1
    with open(os.path.join(out, "excluded_certificates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem", "size", "instance_idx", "trial"])
        w.writerows(sorted(R.mislabelled()))
    print(n_rec, "prompts written")


if __name__ == "__main__":
    main(sys.argv[1])
