"""Reproduces every number, table and figure of the camera-ready paper from the raw log.

Usage (from the repository root):
    python analysis/camera_ready_analysis.py <log.jsonl[.gz]> <output_dir>
"""
import collections
import gzip
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rescore as R  # noqa: E402

P = ["Sorting", "Shortest-Path", "Bipartite-Matching"]
NP_ = ["3-SAT", "TSP-Decision", "Graph-3-Coloring"]
ALLP = P + NP_
GPT, HAIKU, LLAMA = "gpt-4o-mini", "claude-haiku-4-5-20251001", "llama3.1:8b"
M = [GPT, HAIKU, LLAMA]
MN = {GPT: "GPT-4o-mini", HAIKU: "Claude Haiku 4.5", LLAMA: "Llama 3.1 8B"}
S = R.SIZES
CAP = R.MAX_TOKENS
B = 5000


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main(log, out):
    os.makedirs(out, exist_ok=True)
    rows = load(log)
    res = {"records_in_log": len(rows)}
    bad = R.mislabelled()
    res["mislabelled_certificates"] = sorted(map(list, bad))
    gen, ver = [], []
    strict_mismatch = 0
    for r in rows:
        r["cut"] = r["completion_tokens"] >= CAP
        if r["task"] == "generation":
            inst = R.instance(r["problem"], r["size"], r["instance_idx"])
            c = R.PROB[r["problem"]].parse_generation_response(r["raw_response"], inst)
            ok = R.PROB[r["problem"]].verify_solution(inst, c) if c is not None else False
            strict_mismatch += (ok != r["correct"])
            r["g_first"] = bool(r["correct"])
            r["cand"], r["g_final"] = R.final_generation(r["problem"], r["size"], r["instance_idx"], r["raw_response"])
            r["g_pathonly"] = R.final_generation(r["problem"], r["size"], r["instance_idx"], r["raw_response"], check_distance=False)[1]
            r["g_committed"] = r["g_final"] and not r["cut"]
            r["g_upper"] = r["g_final"] or r["cut"]
            gen.append(r)
        else:
            if not r["solution_is_correct"] and (r["problem"], r["size"], r["instance_idx"], r["trial"]) in bad:
                continue
            t = r["solution_is_correct"]
            r["v_first"], r["v_final"] = R.first_verdict(r["raw_response"]), R.final_verdict(r["raw_response"])
            r["v_logged"] = bool(r["correct"])
            vf = r["v_first"] if r["v_first"] is not None else r["v_final"]
            r["ok_first"] = vf is not None and vf == t
            r["ok_final"] = r["v_final"] is not None and r["v_final"] == t
            r["unres"] = r["v_final"] is None
            # acceptance indicators: only an explicit YES accepts the certificate
            r["acc_final"] = r["v_final"] is True
            r["acc_first"] = r.get("llm_answer") is True
            # ranges: a response cut at the limit could have ended with either verdict
            r["acc_low"] = (not t) if r["cut"] else r["acc_final"]     # worst case for J
            r["acc_high"] = t if r["cut"] else r["acc_final"]          # best case for J
            ver.append(r)
    res["first_match_rescoring_mismatches"] = strict_mismatch
    res["records_analysed"] = len(gen) + len(ver)
    res["excluded_records"] = len(rows) - len(gen) - len(ver)

    G = collections.defaultdict(list)
    V = collections.defaultdict(list)
    for r in gen:
        G[(r["problem"], r["model"], r["size"])].append(r)
    for r in ver:
        V[(r["problem"], r["model"], r["size"], r["solution_is_correct"])].append(r)

    # J is the difference between the acceptance rates of correct and incorrect certificates, so any policy that
    # ignores the certificate (always YES, always NO, guessing, or never answering) has J = 0. Accuracy counts a
    # response without a verdict as incorrect.
    OK = {"acc_first": "v_logged", "acc_final": "ok_final", "acc_low": "ok_final", "acc_high": "ok_final"}

    def cell(p, m, n, g, v):
        """generation accuracy, acceptance rate of correct and of incorrect certificates, verification accuracy"""
        acc = (np.mean([r[OK[v]] for r in V[(p, m, n, True)]]) + np.mean([r[OK[v]] for r in V[(p, m, n, False)]])) / 2
        return (np.mean([r[g] for r in G[(p, m, n)]]), np.mean([r[v] for r in V[(p, m, n, True)]]),
                np.mean([r[v] for r in V[(p, m, n, False)]]), acc)

    def agg(p, m, g, v):
        a, T, F, acc = np.array([cell(p, m, n, g, v) for n in S]).mean(0)
        return dict(gen=a, TPR=T, TNR=1 - F, acc=acc, J=T - F, raw=acc - a, cc=T - F - a)

    # instance-level bootstrap (instances resampled within size; the three repetitions stay together)
    rng = np.random.default_rng(2027)

    def boot(p, m, g="g_final", v="acc_final"):
        arr = []
        for n in S:
            a = np.array([np.mean([r[g] for r in G[(p, m, n)] if r["instance_idx"] == i]) for i in range(10)])
            T = np.array([np.mean([r[v] for r in V[(p, m, n, True)] if r["instance_idx"] == i]) for i in range(10)])
            N = np.array([np.mean([r[v] for r in V[(p, m, n, False)] if r["instance_idx"] == i]) for i in range(10)])
            arr.append((a, T, N))
        Js, Ds = [], []
        for _ in range(B):
            c = np.zeros(3)
            for a, T, N in arr:
                k = rng.integers(0, 10, 10)
                c += [a[k].mean(), T[k].mean(), N[k].mean()]
            c /= len(S)
            Js.append(c[1] - c[2])
            Ds.append(c[1] - c[2] - c[0])
        return np.percentile(Js, [2.5, 97.5]).tolist(), np.percentile(Ds, [2.5, 97.5]).tolist()

    tab = {}
    for p in ALLP:
        for m in M:
            first, final = agg(p, m, "g_first", "acc_first"), agg(p, m, "g_final", "acc_final")
            jci, dci = boot(p, m)
            hi, lo = agg(p, m, "g_committed", "acc_high"), agg(p, m, "g_upper", "acc_low")
            tab[f"{p}|{m}"] = dict(first=first, final=final, J_ci=jci, cc_ci=dci, gen_committed=hi["gen"], gen_upper=lo["gen"],
                                   J_low=lo["J"], J_high=hi["J"], cc_low=lo["J"] - lo["gen"], cc_high=hi["J"] - hi["gen"],
                                   cut_gen=float(np.mean([r["cut"] for n in S for r in G[(p, m, n)]])),
                                   unres=float(np.mean([r["unres"] for n in S for c in (True, False) for r in V[(p, m, n, c)]])),
                                   cut_ver=float(np.mean([r["cut"] for n in S for c in (True, False) for r in V[(p, m, n, c)]])),
                                   pathonly=agg(p, m, "g_pathonly", "acc_final"))
    res["problem_model"] = tab

    def mean(ps, ms, rule, k):
        return float(np.mean([tab[f"{p}|{m}"][rule][k] for p in ps for m in ms]))

    res["class"] = {c: {rule: {k: mean(ps, M, rule, k) for k in ("gen", "acc", "J", "raw", "cc")} for rule in ("first", "final")}
                    for c, ps in (("P", P), ("NP", NP_), ("P_without_matching", P[:2]), ("NP_without_coloring", NP_[:2]))}
    res["class_model"] = {f"{c}|{m}": {rule: {k: mean(ps, [m], rule, k) for k in ("gen", "TPR", "TNR", "acc", "J", "raw", "cc")} for rule in ("first", "final")}
                          for c, ps in (("P", P), ("NP", NP_)) for m in M}
    res["problem"] = {p: {rule: {k: mean([p], M, rule, k) for k in ("gen", "acc", "J", "raw", "cc")} for rule in ("first", "final")} for p in ALLP}

    # exact permutation test over problems (20 splits of six problems into two classes of three)
    for rule, k in (("first", "raw"), ("first", "cc"), ("final", "raw"), ("final", "cc")):
        x = np.array([res["problem"][p][rule][k] for p in ALLP])
        obs = x[:3].mean() - x[3:].mean()
        d = [x[list(c)].mean() - x[[i for i in range(6) if i not in c]].mean() for c in itertools.combinations(range(6), 3)]
        res[f"perm_{rule}_{k}"] = dict(diff=float(obs), p=float(np.mean([abs(v) >= abs(obs) - 1e-12 for v in d])), n_splits=len(d), min_p=2 / len(d))

    # positive cells
    for lab, g, v, f in (("raw_first", "g_first", "acc_first", lambda a, T, F, acc: acc - a),
                         ("raw_final", "g_final", "acc_final", lambda a, T, F, acc: acc - a),
                         ("cc_first", "g_first", "acc_first", lambda a, T, F, acc: T - F - a),
                         ("cc_final", "g_final", "acc_final", lambda a, T, F, acc: T - F - a)):
        c54 = {f"{p}|{n}": float(np.mean([f(*cell(p, m, n, g, v)) for m in M])) for p in ALLP for n in S}
        c162 = [f(*cell(p, m, n, g, v)) for p in ALLP for m in M for n in S]
        res[f"cells_{lab}"] = dict(pos54=sum(x > 1e-12 for x in c54.values()), pos54_P=sum(c54[f"{p}|{n}"] > 1e-12 for p in P for n in S),
                                   pos54_NP=sum(c54[f"{p}|{n}"] > 1e-12 for p in NP_ for n in S), pos162=int(sum(x > 1e-12 for x in c162)),
                                   negative54=[k for k, x in c54.items() if x <= 1e-12])

    # repetition agreement: identical prompts issued three times at temperature 0
    ag = collections.defaultdict(lambda: [0, 0, 0])
    grp = collections.defaultdict(list)
    for r in gen:
        grp[(r["model"], "gen", r["problem"], r["size"], r["instance_idx"])].append(r)
    for r in ver:
        if r["solution_is_correct"]:
            grp[(r["model"], "ver_correct", r["problem"], r["size"], r["instance_idx"])].append(r)
    for k, v in grp.items():
        a = ag[(k[0], k[1])]
        a[0] += 1
        a[1] += len(set(x["raw_response"] for x in v)) > 1
        a[2] += len(set(x["g_final"] if k[1] == "gen" else x["ok_final"] for x in v)) > 1
    res["repetitions"] = {f"{MN[m]}|{t}": dict(prompts=a[0], text_differs=a[1], outcome_differs=a[2]) for (m, t), a in ag.items()}

    # verdicts reversed by the model's own explanation
    rev = [r for r in ver if r["v_first"] is not None and r["v_first"] != r["v_final"]]
    res["reversals_by_model"] = {MN[m]: [sum(r["model"] == m for r in rev), sum(r["model"] == m and r["ok_final"] for r in rev)] for m in M}
    res["reversals"] = dict(total=len(rev), correct=sum(r["ok_final"] for r in rev),
                            by={f"{MN[k[0]]}|{k[1]}": v for k, v in collections.Counter((r["model"], r["problem"]) for r in rev).items()})
    res["verdict_position"] = {MN[m]: dict(first_line=sum(r["v_first"] is not None for r in ver if r["model"] == m),
                                           later=sum(r["v_first"] is None and r["v_final"] is not None for r in ver if r["model"] == m),
                                           unresolved=sum(r["unres"] for r in ver if r["model"] == m)) for m in M}
    res["unresolved"] = {f"{MN[k[0]]}|{k[1]}": v for k, v in collections.Counter((r["model"], r["problem"]) for r in ver if r["unres"]).items()}
    res["unresolved_all_cut"] = all(r["cut"] for r in ver if r["unres"])
    res["unresolved_sat_by_size"] = {n: [sum(r["unres"] for c in (True, False) for r in V[("3-SAT", HAIKU, n, c)]), sum(len(V[("3-SAT", HAIKU, n, c)]) for c in (True, False))] for n in S}
    res["conditional_on_verdict"] = {p: dict(TPR=float(np.mean([r["ok_final"] for r in ver if r["model"] == HAIKU and r["problem"] == p and r["solution_is_correct"] and not r["unres"]])),
                                             TNR=float(np.mean([r["ok_final"] for r in ver if r["model"] == HAIKU and r["problem"] == p and not r["solution_is_correct"] and not r["unres"]])),
                                             n_true=sum(1 for r in ver if r["model"] == HAIKU and r["problem"] == p and r["solution_is_correct"] and not r["unres"]),
                                             n_false=sum(1 for r in ver if r["model"] == HAIKU and r["problem"] == p and not r["solution_is_correct"] and not r["unres"])) for p in NP_}
    # Claude Haiku on 3-SAT by size band: verdicts on correct and on incorrect assignments
    res["haiku_sat_bands"] = {lab: {str(c): dict(n=len(x), yes=sum(r["v_final"] is True for r in x), no=sum(r["v_final"] is False for r in x), unres=sum(r["unres"] for r in x))
                                    for c in (True, False) for x in [[r for r in ver if r["model"] == HAIKU and r["problem"] == "3-SAT" and r["size"] in band and r["solution_is_correct"] == c]]}
                              for lab, band in (("5", [5]), ("8-25", [8, 10, 15, 20, 25]), ("30-50", [30, 40, 50]))}
    res["verification_cut"] = {MN[m]: dict(total=sum(r["cut"] for r in ver if r["model"] == m), after_verdict=sum(r["cut"] and not r["unres"] for r in ver if r["model"] == m),
                                           by={p: sum(r["cut"] for r in ver if r["model"] == m and r["problem"] == p) for p in ALLP}) for m in M}
    # acceptance by certificate type
    res["accept_correct_incorrect"] = {f"{MN[m]}|{p}": [float(np.mean([r["v_final"] is True for r in ver if r["model"] == m and r["problem"] == p and r["solution_is_correct"] == c])) for c in (True, False)] for m in M for p in ALLP}
    # TSP: incorrect tours whose explanation ends above the threshold although the written verdict accepts them
    over = re.compile(r"(exceed|greater than|is more than|larger than|> ?\d|not (?:<=|less)|above the threshold|does not (?:meet|satisfy))", re.I)
    res["tsp_contradictions"] = {}
    for m, lim in ((GPT, 10), (HAIKU, 50), (LLAMA, 50)):
        sel = [r for r in ver if r["model"] == m and r["problem"] == "TSP-Decision" and not r["solution_is_correct"] and r["size"] <= lim]
        ov = [bool(over.search(" ".join(r["raw_response"].split("\n")[-8:]))) for r in sel]
        res["tsp_contradictions"][MN[m]] = dict(max_n=lim, incorrect=len(sel), accepted=sum(r["v_final"] is True for r in sel), explanation_over=sum(ov),
                                                accepted_but_over=sum(o and r["v_final"] is True for o, r in zip(ov, sel)))
    # chance that a random well-formed answer is valid
    import random as _random
    rg = _random.Random(1)
    res["random_answer_valid"] = {}
    for n in (5, 8):
        hit = dict(TSP=0, SAT=0, COL=0)
        for i in range(10):
            a, b, c = R.instance("TSP-Decision", n, i), R.instance("3-SAT", n, i), R.instance("Graph-3-Coloring", n, i)
            for _ in range(2000):
                q = list(range(n))
                rg.shuffle(q)
                hit["TSP"] += R.PROB["TSP-Decision"].verify_solution(a, q)
                hit["SAT"] += R.PROB["3-SAT"].verify_solution(b, [rg.random() < 0.5 for _ in range(n)])
                hit["COL"] += R.PROB["Graph-3-Coloring"].verify_solution(c, [rg.choice("RGB") for _ in range(n)])
        res["random_answer_valid"][str(n)] = {k: v / 20000 for k, v in hit.items()}
    # Shortest Path: distance stated with a credited path
    res["sp_distance"] = {MN[m]: dict(path_right=sum(r["g_pathonly"] for r in gen if r["model"] == m and r["problem"] == "Shortest-Path"),
                                      path_and_distance_right=sum(r["g_final"] for r in gen if r["model"] == m and r["problem"] == "Shortest-Path")) for m in M}
    res["llama_sat_no_assignment"] = sum(r["cand"] is None for r in gen if r["model"] == LLAMA and r["problem"] == "3-SAT")
    res["generation_cut"] = {MN[m]: dict(total=sum(r["cut"] for r in gen if r["model"] == m), by={p: sum(r["cut"] for r in gen if r["model"] == m and r["problem"] == p) for p in ALLP}) for m in M}

    # matching: how do the API models justify rejecting the perturbed (one edge short) matching?
    mt = {}
    for m in (GPT, HAIKU):
        sel = [r for r in ver if r["model"] == m and r["problem"] == "Bipartite-Matching" and not r["solution_is_correct"] and r["v_final"] is False]
        both = 0
        for r in sel:
            k = R.instance(r["problem"], r["size"], r["instance_idx"]).instance_data["max_matching_size"]
            txt = r["raw_response"]
            both += any(re.search(rf"\b{k}\b", l) and re.search(rf"\b{k - 1}\b", l) for l in re.split(r"\n|(?<=[a-z\)])\.\s", txt) if re.search(r"size|maximum|expected|edges", l, re.I))
        false_claim = 0
        for r in sel:
            edges = set(map(tuple, R.instance(r["problem"], r["size"], r["instance_idx"]).instance_data["edges"]))
            for l in re.split(r"\n|(?<=\.)\s", r["raw_response"]):
                if re.search(r"(does not exist|doesn.t exist|not exist|is not (?:in|present|listed|an edge|part)|missing|not found)", l, re.I) and any(
                        (int(u), int(v)) in edges for u, v in re.findall(r"L(\d+)\s*-+\s*R(\d+)", l)):
                    false_claim += 1
                    break
        mt[MN[m]] = dict(correct_rejections=len(sel), cites_both_sizes=both, claims_present_edge_missing=false_claim)
    res["matching_rejections"] = mt
    sel = [r for r in gen if r["problem"] == "Bipartite-Matching" and r["model"] == GPT]
    echo = 0
    for r in sel:
        k = R.instance(r["problem"], r["size"], r["instance_idx"]).instance_data["max_matching_size"]
        mm = re.findall(r"size\s*:?\s*\**\s*(\d+)", r["raw_response"], re.I)
        echo += bool(mm) and int(mm[-1]) == k
    res["matching_generation_echo_GPT"] = dict(echo=echo, n=len(sel), valid=sum(r["g_final"] for r in sel))

    # solution extraction: responses credited (or lost) by the final-statement rule, and answers preceded by reasoning
    ex = {}
    for m in M:
        for p in ALLP:
            sel = [r for r in gen if r["model"] == m and r["problem"] == p]
            first = []
            for r in sel:
                ls = [l for l in r["raw_response"].strip().split("\n") if l.strip()]
                first.append(R.final_generation(p, r["size"], r["instance_idx"], ls[0])[0] is None if ls else True)
            ex[f"{MN[m]}|{p}"] = dict(n=len(sel), first_match=sum(r["g_first"] for r in sel), final=sum(r["g_final"] for r in sel),
                                      gained=sum(r["g_final"] and not r["g_first"] for r in sel), lost=sum(r["g_first"] and not r["g_final"] for r in sel),
                                      first_line_not_an_answer=float(np.mean(first)))
    res["extraction"] = ex
    res["gained_cut"] = sum(r["g_final"] and not r["g_first"] and r["cut"] for r in gen)

    # how far the incorrect certificates of the two "random" rules are from valid ones
    import math
    conf, ratio = [], {n: [] for n in S}
    for n in S:
        for i in range(10):
            for t in range(3):
                inst = R.instance("Graph-3-Coloring", n, i)
                w = R.wrong_certificate("Graph-3-Coloring", n, i, t)
                e = inst.instance_data["edges"]
                conf.append(sum(w[u] == w[v] for u, v in e) / len(e))
                inst = R.instance("TSP-Decision", n, i)
                w = R.wrong_certificate("TSP-Decision", n, i, t)
                c = inst.instance_data["cities"]
                ratio[n].append(sum(math.dist(c[w[j]], c[w[(j + 1) % n]]) for j in range(n)) / inst.instance_data["threshold"])
    res["coloring_conflict_share"] = float(np.mean(conf))
    res["tsp_wrong_length_over_threshold"] = {n: float(np.mean(v)) for n, v in ratio.items()}

    # response-policy facts quoted in the text (final-statement rule)
    res["yes_rate"] = {f"{MN[m]}|{p}": float(np.mean([r["v_final"] is True for r in ver if r["model"] == m and r["problem"] == p])) for m in M for p in ALLP}
    res["gen_by_size"] = {f"{MN[m]}|{p}": [float(np.mean([r["g_final"] for r in G[(p, m, n)]])) for n in S] for m in M for p in ALLP}
    res["J_by_size"] = {f"{MN[m]}|{p}": [float(cell(p, m, n, "g_final", "acc_final")[1] - cell(p, m, n, "g_final", "acc_final")[2]) for n in S] for m in M for p in ALLP}
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1, default=float)

    # ------------------------------------------------------------------ console summary
    print("records", res["records_in_log"], "analysed", res["records_analysed"], "excluded", res["excluded_records"], "| first-match rescoring mismatches", strict_mismatch)
    print("%-19s %-7s| gen first final [comm,upper] |  TPR  TNR  unres |  J(first) J [CI]            | raw(first) raw(final) cc(first) cc [CI]            (bounds)" % ("", ""))
    for p in ALLP:
        for m in M:
            t = tab[f"{p}|{m}"]
            print("%-19s %-7s|   %.2f  %.2f  [%.2f,%.2f] | %.2f %.2f %.2f |  %5.2f  %5.2f [%5.2f,%5.2f] |  %5.2f     %5.2f     %5.2f   %5.2f [%5.2f,%5.2f]  (%.2f..%.2f)" % (
                p, MN[m][:6], t["first"]["gen"], t["final"]["gen"], t["gen_committed"], t["gen_upper"], t["final"]["TPR"], t["final"]["TNR"], t["unres"],
                t["first"]["J"], t["final"]["J"], *t["J_ci"], t["first"]["raw"], t["final"]["raw"], t["first"]["cc"], t["final"]["cc"], *t["cc_ci"], t["cc_low"], t["cc_high"]))
    for k in ("class", "problem"):
        for kk, vv in res[k].items():
            print(k, kk, {r: {a: round(b, 3) for a, b in d.items()} for r, d in vv.items()})
    for k in res:
        if k.startswith(("perm_", "cells_", "revers", "repet", "verdict_pos", "unresolved", "conditional", "generation_cut", "matching_")):
            print(k, res[k])
    figures(res, tab, out)


def figures(res, tab, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    col = {GPT: "#1b6ca8", HAIKU: "#c8553d", LLAMA: "#5b8c5a"}
    mk = {GPT: "o", HAIKU: "s", LLAMA: "^"}
    short = {"Sorting": "Sorting", "Shortest-Path": "Shortest Path", "Bipartite-Matching": "Matching", "3-SAT": "3-SAT", "TSP-Decision": "TSP", "Graph-3-Coloring": "3-Coloring"}

    # Fig. 1: naive raw gap -> corrected gap, per problem and model
    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    y, yt, yl = 0, [], []
    for gi, p in enumerate(ALLP):
        if gi == 3:
            ax.axhline(y - 0.15, color="0.45", lw=0.7)
            y += 0.7
        c = []
        for m in M:
            t = tab[f"{p}|{m}"]
            a, b = t["first"]["raw"], t["final"]["cc"]
            ax.plot([b, a], [y, y], color=col[m], lw=0.8, alpha=0.45, zorder=1)
            ax.errorbar(b, y, xerr=[[b - t["cc_ci"][0]], [t["cc_ci"][1] - b]], fmt=mk[m], color=col[m], ms=4, lw=1.0, capsize=1.5, zorder=3)
            ax.plot(a, y, mk[m], mfc="white", mec=col[m], ms=4, zorder=2)
            c.append(y)
            y += 1
        yt.append(np.mean(c))
        yl.append(short[p])
        y += 0.7
    ax.axvline(0, color="k", lw=0.7)
    ax.axvline(0.5, color="0.6", lw=0.6, ls=":")
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.invert_yaxis()
    ax.set_xlim(-0.35, 1.02)
    ax.set_xlabel(r"raw gap $\Delta$ (open)   and   corrected gap $\Delta^{*}$ (filled)", fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.text(0.985, 0.985, "P", transform=ax.transAxes, ha="right", va="top", fontsize=8, fontweight="bold")
    ax.text(0.985, 0.015, "NP-complete", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")
    hs = [plt.Line2D([], [], color=col[m], marker=mk[m], lw=0, ms=4, label=MN[m]) for m in M]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.47, 1.10), ncol=3, frameon=False, fontsize=7, handletextpad=0.1, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "gap_decomposition.pdf"))
    fig.savefig(os.path.join(out, "gap_decomposition.png"), dpi=220)
    plt.close(fig)

    # Fig. 2: verification response profile
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    ax.plot([0, 1], [1, 0], "k--", lw=0.8, alpha=0.6)
    ax.text(0.555, 0.375, r"$J=0$", fontsize=7, alpha=0.8, ha="center")
    off = {(GPT, "Sorting"): (4, -8), (GPT, "Shortest-Path"): (-54, -6), (GPT, "Bipartite-Matching"): (-36, -2), (GPT, "3-SAT"): (-24, 4), (GPT, "Graph-3-Coloring"): (-45, -1),
           (HAIKU, "Sorting"): (5, -6), (HAIKU, "Shortest-Path"): (5, 1), (HAIKU, "Bipartite-Matching"): (5, 1), (HAIKU, "3-SAT"): (5, -2), (HAIKU, "TSP-Decision"): (5, -2), (HAIKU, "Graph-3-Coloring"): (5, -6), (LLAMA, "3-SAT"): (5, 2)}
    for m in M:
        for p in ALLP:
            t = tab[f"{p}|{m}"]["final"]
            ax.plot(t["TNR"], t["TPR"], mk[m], mfc=col[m] if p in NP_ else "white", mec=col[m], ms=4.6, mew=1.0, zorder=3)
            if (m, p) in off:
                ax.annotate(short[p], (t["TNR"], t["TPR"]), fontsize=6, xytext=off[(m, p)], textcoords="offset points", color=col[m])
    ax.annotate("accept nearly everything:\nLlama (five problems),\nGPT-4o-mini on TSP", (0.06, 0.975), xytext=(0.31, 0.655), fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.5))
    ax.set_xlabel("true negative rate (rejects incorrect certificates)", fontsize=7)
    ax.set_ylabel("true positive rate\n(accepts correct certificates)", fontsize=7)
    ax.set_xlim(-0.03, 1.27)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.05, 1.05)
    hs = [plt.Line2D([], [], color=col[m], marker=mk[m], lw=0, ms=4, label=MN[m]) for m in M]
    hs += [plt.Line2D([], [], color="0.3", marker="o", mfc="white", lw=0, ms=4, label="P"), plt.Line2D([], [], color="0.3", marker="o", lw=0, ms=4, label="NP-complete")]
    ax.legend(handles=hs, fontsize=6, loc="lower left", frameon=False, handletextpad=0.1, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "tpr_tnr_map.pdf"))
    fig.savefig(os.path.join(out, "tpr_tnr_map.png"), dpi=220)
    plt.close(fig)

    # Fig. 3: generation accuracy and informedness versus size
    fig, axes = plt.subplots(1, 6, figsize=(7.16, 1.55), sharey=True)
    for ax, p in zip(axes, ALLP):
        for m in M:
            ax.plot(S, res["gen_by_size"][f"{MN[m]}|{p}"], "-", marker=mk[m], color=col[m], ms=2.4, lw=0.9)
            ax.plot(S, res["J_by_size"][f"{MN[m]}|{p}"], "--", marker=mk[m], mfc="white", color=col[m], ms=2.4, lw=0.9, mew=0.6)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(short[p] + (" (P)" if p in P else " (NP-c)"), fontsize=7, pad=2)
        ax.set_xticks([5, 20, 35, 50])
        ax.tick_params(labelsize=6, length=2)
        ax.set_xlabel("$n$", fontsize=7, labelpad=1)
    axes[0].set_ylabel(r"$\mathrm{Acc}_{\mathrm{gen}}$ or $J_{\mathrm{ver}}$", fontsize=7, labelpad=1)
    axes[0].set_ylim(-0.36, 1.06)
    hs = [plt.Line2D([], [], color=col[m], marker=mk[m], ms=3, lw=0.9, label=MN[m]) for m in M]
    hs += [plt.Line2D([], [], color="0.2", lw=0.9, label="generation accuracy"), plt.Line2D([], [], color="0.2", lw=0.9, ls="--", label=r"informedness $J_{\mathrm{ver}}$")]
    fig.legend(handles=hs, loc="lower center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.03), columnspacing=1.4, handlelength=1.8)
    fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1), w_pad=0.4)
    fig.savefig(os.path.join(out, "curves.pdf"))
    fig.savefig(os.path.join(out, "curves.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
