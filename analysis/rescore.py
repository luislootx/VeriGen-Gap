"""Scoring rules used in the camera-ready paper.

Two rules are implemented for every logged response:

* as-prompted rule: the response is read as if it contained only what the prompt asks for. These are
  the values stored in the log by src/experiments/runner.py: all integers (Sorting), the first path
  (Shortest-Path, distance not checked), all pairs (Matching), the first n integers (TSP), the last value
  of every variable or vertex (3-SAT, 3-Coloring); verdict on the first line, else the last line, else keywords.
* final-statement rule (primary in the paper): generation = the last complete solution written in
  the response (for Shortest-Path, path and distance), verification = the last explicit verdict, i.e. a line
  that is YES or NO or a sentence that states it ("Correction: YES", "the answer is NO"). A verification
  response with no explicit verdict is "unresolved" and counts as incorrect.

Instances, prompts and certificates are regenerated from the deterministic seeds of the runner.
"""
import re

from src.problems import ALL_PROBLEMS

PROB = {c().name: c() for c in ALL_PROBLEMS}
SIZES = [5, 8, 10, 15, 20, 25, 30, 40, 50]
MAX_TOKENS = 1024
_cache = {}


def instance(name, n, i):
    """Instance i of size n, rebuilt with the seed used by the runner."""
    k = (name, n, i)
    if k not in _cache:
        _cache[k] = PROB[name].generate_instance(n, seed=n * 1000 + i)
    return _cache[k]


def wrong_certificate(name, n, i, trial):
    return PROB[name].make_wrong_solution(instance(name, n, i), seed=n * 1000 + i + trial + 10000)


def certificate_is_valid(name, n, i, cert):
    """Independent validity check (for Shortest-Path the claimed distance is checked as well)."""
    inst = instance(name, n, i)
    if name != "Shortest-Path":
        return bool(PROB[name].verify_solution(inst, cert))
    return bool(PROB[name].verify_solution(inst, cert)) and cert["distance"] == inst.instance_data["optimal_distance"]


def mislabelled():
    """(problem, n, instance, trial) whose 'incorrect' certificate is in fact valid."""
    out = set()
    for name in PROB:
        for n in SIZES:
            for i in range(10):
                for t in range(3):
                    if certificate_is_valid(name, n, i, wrong_certificate(name, n, i, t)):
                        out.add((name, n, i, t))
    return out


# ---------------------------------------------------------------- generation: final-statement parser
def _clean(t):
    return (t.replace("**", "").replace("`", "").replace("→", "->")
            .replace("–", "-").replace("−", "-"))


def _lines(t):
    return [l.strip() for l in _clean(t).split("\n") if l.strip()]


PAIR = re.compile(r"L\s*(\d+)\s*(?:-{1,2}>?|:|to)\s*R\s*(\d+)", re.I)


def _sorting(t, inst):
    for s in reversed(re.findall(r"\[([^\[\]]*)\]", _clean(t))):
        nums = re.findall(r"-?\d+", s)
        if nums:
            return [int(x) for x in nums]
    for l in reversed(_lines(t)):
        nums = re.findall(r"-?\d+", l)
        if len(nums) >= 2:
            return [int(x) for x in nums]
    return None


def _path(t, inst):
    ls = _lines(t)
    d = re.findall(r"distance\s*(?:is|=|of)?\s*[:=]?\s*(\d+(?:\.\d+)?)", _clean(t), re.I)
    d = float(d[-1]) if d else None
    for l in reversed(ls):
        m = re.search(r"path\s*(?:is|=)?\s*:?\s*((?:\d+\s*(?:,|->|-|>)\s*)+\d+)", l, re.I)
        if m:
            return {"path": [int(x) for x in re.findall(r"\d+", m.group(1))], "distance": d}
    for l in reversed(ls):
        m = re.search(r"((?:\d+\s*(?:,|->)\s*)+\d+)", l)
        if m and not re.search(r"weight|distance", l, re.I):
            return {"path": [int(x) for x in re.findall(r"\d+", m.group(1))], "distance": d}
    return None


def _matching(t, inst):
    ls = _lines(t)

    def pure(l):
        rest = re.sub(r"[\s,;.\-*\[\](){}]|and|\d+\.", "", PAIR.sub("", l), flags=re.I)
        return bool(PAIR.search(l)) and rest == ""

    idx = [k for k, l in enumerate(ls) if pure(l)]
    if idx:
        e = s = idx[-1]
        while s - 1 in idx and len(PAIR.findall(ls[s])) == 1 and len(PAIR.findall(ls[s - 1])) == 1:
            s -= 1
        return [(int(u), int(v)) for l in ls[s:e + 1] for u, v in PAIR.findall(l)]
    for l in reversed(ls):
        p = PAIR.findall(l)
        if len(p) >= 2:
            return [(int(u), int(v)) for u, v in p]
    return None


def _tsp(t, inst):
    n = len(inst.instance_data["cities"])
    for l in reversed(_lines(t)):
        l2 = re.sub(r"^(tour|answer|final answer|solution|optimal tour|best tour)\s*[:=]\s*", "", l, flags=re.I)
        if re.fullmatch(r"[\[\(]?\s*\d+(?:\s*(?:,|->|-)\s*\d+)+\s*[\]\)]?\.?", l2):
            a = [int(x) for x in re.findall(r"\d+", l2)]
            if len(a) == n + 1 and a[0] == a[-1]:
                a = a[:-1]
            return a
    return None


def _assignment(t, n, pat, conv, off):
    a = [None] * n
    for l in reversed(_lines(t)):
        for i, v in re.findall(pat, l, re.I):
            k = int(i) - off
            if 0 <= k < n and a[k] is None:
                a[k] = conv(v)
        if None not in a:
            return a
    return None


def _sat(t, inst):
    return _assignment(t, inst.instance_data["n_vars"], r"x(\d+)\s*=\s*([TF])", lambda v: v.upper() == "T", 1)


def _coloring(t, inst):
    return _assignment(t, inst.instance_data["n_vertices"], r"(\d+)\s*=\s*([RGB])\b", lambda v: v.upper(), 0)


FINAL = {"Sorting": _sorting, "Shortest-Path": _path, "Bipartite-Matching": _matching,
         "3-SAT": _sat, "TSP-Decision": _tsp, "Graph-3-Coloring": _coloring}


def final_generation(name, n, i, text, check_distance=True):
    """Last complete solution and its correctness. For Shortest-Path the stated distance must be right as well,
    because the certificates shown in the verification task are (path, distance) pairs."""
    inst = instance(name, n, i)
    cand = FINAL[name](text, inst)
    if cand is None:
        return None, False
    ok = bool(PROB[name].verify_solution(inst, cand))
    if name == "Shortest-Path" and check_distance:
        ok = ok and cand["distance"] == inst.instance_data["optimal_distance"]
    return cand, ok


# ---------------------------------------------------------------- verification: verdict lines
def _verdict_line(l):
    w = re.sub(r"^[#>\s]+", "", l.replace("**", "").replace("`", "").strip()).strip().upper().strip(".,!*: ")
    w = re.sub(r"^(ANSWER|VERDICT|FINAL ANSWER|CONCLUSION)\s*[:=]\s*", "", w)
    return {"YES": True, "NO": False}.get(w)


def first_verdict(text):
    ls = [l for l in text.strip().split("\n") if l.strip()]
    return _verdict_line(ls[0]) if ls else None


# a verdict may also be stated in a sentence: "YES - ...", "Correction: YES", "the answer is NO."
_LEAD = re.compile(r"^(?:let me reconsider(?: my answer)?:\s*|actually,?\s*(?:let me reconsider(?: my answer)?:\s*)?"
                   r"|correction:?\s*:?\s*|final (?:conclusion|answer|verdict):?\s*)?(YES|NO)(?=\s*(?:$|[-\u2013\u2014,.(:;]))", re.I)
_PHRASE = re.compile(r"\b(?:answer|response|verdict|conclusion|correction|reconsider(?: my answer)?|verification|actually)\b"
                     r"[^.?\n]{0,70}?(?:\bis\b|\bbe\b|\bremains\b|:|,)\s*(?:still |actually |confirmed as |indeed )?(YES|NO)\b", re.I)


def _verdict_sentence(l):
    c = re.sub("[*`_#>\"'\u201c\u201d\u2018\u2019]", "", l).strip()
    m = _LEAD.match(c)
    if m and m.group(1).isupper():
        return m.group(1) == "YES"
    out = None
    for s in re.split(r"(?<=[.!?])\s+", c):
        if re.search(r"\bwould (?:be|say)\b|starting with|must start|if only", s):
            continue                                   # hypothetical verdicts
        v = [x.group(1) for x in _PHRASE.finditer(s) if x.group(1).isupper()]
        if v:
            out = v[-1] == "YES"
        if re.search(r"\bYES\b.*\bbut\b.*\bNO\b", s):  # "YES for the first condition but NO for the second"
            out = False
    return out


def final_verdict(text):
    """Last explicit verdict of the response (True = YES), or None."""
    v = None
    for l in text.strip().split("\n"):
        w = _verdict_line(l)
        if w is None:
            w = _verdict_sentence(l)
        if w is not None:
            v = w
    return v
