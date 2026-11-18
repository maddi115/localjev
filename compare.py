"""
compare.py - grade every setup on the same questions and write one results file.

Setups:
    nli plain            DeBERTa, one wording
    nli + stasis         DeBERTa, probability stasis
    decider plain        Amazon Strands Decider 2B, one wording
    decider + stasis     Strands + probability stasis
    nli+decider stasis   both models + stasis
    mapika plain         Mapika decider-2b, one wording (76% on JevBench public)
    mapika + stasis      Mapika + probability stasis
    mapika+decider       the two strongest free models + stasis
Models run one at a time and are unloaded in between, so they fit in 8 GB.
    jev                  TypeSafe Jev API (only if TYPESAFE_API_KEY is set and has credits)

Question sets:
    ours        your 20 labeled tests in tests/
    jb-original JevBench public "original" tier (72)
    jb-easy     JevBench public easy tier (48)
    jb-hard     JevBench public hard tier (111)

No calibration is used, so nothing is graded on data it learned from.

Usage:
    python compare.py                      everything
    python compare.py --limit 20           quick run, 20 questions per set
    python compare.py --no-decider         skip Strands
Writes results.md (paste this to Claude) and results.json.
"""
import argparse
import json
import os
import statistics
import sys
import time

import localjev as L

JEV_URL = "https://api.typesafe.ai/v1/systemone"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- data

def load_ours(folder):
    items = []
    for name, req, labels in L.load_cases(folder):
        for qid, q in req["questions"].items():
            if qid in labels:
                sub = {"state": req["state"], "questions": {qid: q}}
                items.append({"id": f"{name}:{qid}", "req": sub, "qid": qid, "label": labels[qid]})
    return items


def load_jevbench(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = r["question"]
            exp = r["expected"]
            if q["type"] == "noul":
                label = str(exp).lower() in ("yes", "true", "1")
            elif q["type"] == "score":
                label = int(exp)
            else:
                label = exp
            items.append({"id": r["id"], "req": {"state": r["state"], "questions": {"q": q}}, "qid": "q", "label": label})
    return items


# ---------------------------------------------------------------- jev

class Jev:
    def __init__(self, key):
        import urllib.request
        self.urllib = urllib.request
        self.key = key
        self.dead = None

    def raw(self, req):
        if self.dead:
            raise RuntimeError(self.dead)
        body = json.dumps({"model": "jev-latest", "state": req["state"], "questions": req["questions"]}).encode()
        r = self.urllib.Request(JEV_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        try:
            with self.urllib.urlopen(r, timeout=60) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            detail = getattr(e, "read", lambda: b"")().decode(errors="replace")[:200]
            self.dead = f"{e} {detail}".strip()
            raise RuntimeError(self.dead)
        out = {}
        for qid, q in req["questions"].items():
            a = data["answers"][qid]
            if q["type"] == "noul":
                p = float(a["noul"])
                d = {"true": p, "false": 1 - p}
            elif q["type"] == "choice":
                d = L.normalize({k: float(a.get("probabilities", {}).get(k, 0.0)) for k in q["criteria"]})
            else:
                probs = a.get("probabilities", {})
                d = L.normalize({str(i): float(probs.get(str(i), 0.0)) for i in range(len(q["criteria"]))})
            out[qid] = [("jev", d)]
        return out


# ---------------------------------------------------------------- scoring

def answer_dist(q, ans):
    if q["type"] == "noul":
        return {"true": ans["noul"], "false": 1 - ans["noul"]}
    return ans["probabilities"]


def score_set(engine, rows, stasis, threshold):
    """rows: list of (item, variants, seconds)"""
    n = len(rows)
    if not n:
        return None
    correct, chance, brier, confs, wrong_confs = 0, 0.0, 0.0, [], []
    acted = acted_wrong = 0
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    secs = []
    for item, variants, sec in rows:
        q = item["req"]["questions"][item["qid"]]
        ans = engine.summarize(q, variants, stasis)
        ok = L.is_correct(q, ans, item["label"])
        keys = L.outcome_keys(q)
        dist = answer_dist(q, ans)
        tk = L.true_key(q, item["label"])
        correct += ok
        chance += 1.0 / len(keys)
        brier += sum((dist.get(k, 0.0) - (1.0 if k == tk else 0.0)) ** 2 for k in keys)
        top = L.top_prob(q, ans)
        b = bins[min(9, int(top * 10))]
        b[0] += 1; b[1] += top; b[2] += ok
        confs.append(ans["confidence"])
        if not ok:
            wrong_confs.append(ans["confidence"])
        if ans["confidence"] >= threshold:
            acted += 1
            acted_wrong += not ok
        secs.append(sec)
    acc = correct / n
    ch = chance / n
    safe = min(1.0, round(max(wrong_confs) + 0.01, 2)) if wrong_confs else 0.0
    return {
        "n": n,
        "accuracy": acc,
        "chance_corrected": max(0.0, (acc - ch) / (1 - ch)) if ch < 1 else 0.0,
        "acted": acted,
        "acted_wrong": acted_wrong,
        "safe_threshold": safe,
        "acted_at_safe": sum(c >= safe for c in confs) if safe < 1.0 else 0,
        "brier": brier / n,
        "ece": sum(abs(b[1] - b[2]) for b in bins if b[0]) / n,
        "sec_per_q": statistics.mean(secs),
    }


# ---------------------------------------------------------------- running

def run_backend(name, fn, datasets, limit):
    """fn(req) -> {qid: variants}. Returns {dataset: [(item, variants, seconds), ...]}."""
    out = {}
    for ds, items in datasets.items():
        items = items[:limit] if limit else items
        if getattr(getattr(fn, "__self__", None), "dead", None):
            out[ds] = {"rows": [], "errors": 0, "total": len(items)}
            continue
        rows, errors, t_start = [], 0, time.time()
        for i, item in enumerate(items, 1):
            t0 = time.time()
            try:
                raw = fn(item["req"])
                rows.append((item, raw[item["qid"]], time.time() - t0))
            except Exception as e:
                errors += 1
                if errors <= 3:
                    log(f"  [{name}] {item['id']}: {str(e)[:150]}")
                if getattr(getattr(fn, "__self__", None), "dead", None):
                    log("  [jev] stopping: API not usable")
                    break
            if i % 10 == 0 or i == len(items):
                el = time.time() - t_start
                log(f"  [{name}] {ds}: {i}/{len(items)}  ({el:.0f}s, ~{el / i * (len(items) - i):.0f}s left)")
        out[ds] = {"rows": rows, "errors": errors, "total": len(items)}
    return out


def pct(x):
    return f"{x:.0%}"


def write_report(results, path_md, path_json, args):
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    lines = ["# localjev comparison results", "",
             f"threshold for 'act': {args.threshold}  |  no calibration used  |  limit: {args.limit or 'none'}", "",
             "Columns: Acc = accuracy. CC = chance-corrected accuracy (JevBench style, 0 = guessing). "
             "Acted = answered automatically at the threshold. **Wrong-acted = confident but wrong (want 0).** "
             "Safe thr = lowest threshold with 0 wrong-acted (and how many it acts on). "
             "Brier/ECE = probability honesty, lower is better.", ""]
    datasets = [k for k in next(iter(results.values()))["sets"] if k != "ALL"] if results else []
    for ds in datasets + ["ALL"]:
        lines += [f"## {ds}", "",
                  "| setup | n | Acc | CC | Acted | Wrong-acted | Safe thr (acts on) | Brier | ECE | sec/q |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for setup, r in results.items():
            m = r["sets"].get(ds)
            if r.get("skipped"):
                if ds == datasets[0]:
                    lines.append(f"| {setup} | skipped: {r['skipped'][:80]} |||||||||")
                continue
            if not m:
                continue
            lines.append(
                f"| {setup} | {m['n']} | {pct(m['accuracy'])} | {pct(m['chance_corrected'])} | "
                f"{m['acted']}/{m['n']} | **{m['acted_wrong']}** | {m['safe_threshold']:.2f} ({m['acted_at_safe']}) | "
                f"{m['brier']:.3f} | {m['ece']:.3f} | {m['sec_per_q']:.2f} |")
        lines.append("")
    errs = [(s, ds, r["errors"][ds]) for s, r in results.items() for ds in r.get("errors", {}) if r["errors"][ds]]
    if errs:
        lines += ["## errors", ""] + [f"- {s} / {ds}: {e} questions failed" for s, ds, e in errs] + [""]
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Compare every setup on the same questions")
    ap.add_argument("--tests", default="tests")
    ap.add_argument("--jevbench", default=os.path.expanduser("~/jevbench/datasets/public"))
    ap.add_argument("--limit", type=int, default=0, help="max questions per set (0 = all)")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--no-decider", action="store_true")
    ap.add_argument("--no-jev", action="store_true")
    ap.add_argument("--no-mapika", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default=L.NLI_MODEL, help="NLI model")
    args = ap.parse_args()

    datasets = {"ours": load_ours(args.tests)}
    for tier in ("original", "easy", "hard"):
        p = os.path.join(args.jevbench, f"{tier}.jsonl")
        if os.path.exists(p):
            datasets[f"jb-{tier}"] = load_jevbench(p)
        else:
            log(f"JevBench {tier} not found at {p} (clone it, see instructions)")
    log("question sets: " + ", ".join(f"{k}={len(v)}" for k, v in datasets.items()))

    device = L.pick_device(args.device)
    raws = {}

    class Stub:  # stands in for a model after it's unloaded (setups only need its name)
        def __init__(self, name, model_name):
            self.name, self.model_name = name, model_name

    def run_model(name, make):
        """Load one model, run every question, then free the GPU for the next model."""
        try:
            log(f"Loading {name} model...")
            b = make()
        except ImportError as e:
            log(f"{name} not installed ({e}), skipping")
            return None
        raws[name] = run_backend(name, L.Engine([b], stasis=True, threshold=args.threshold).raw, datasets, args.limit)
        stub = Stub(b.name, b.model_name)
        del b
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return stub

    models = {"nli": run_model("nli", lambda: L.NLIBackend(args.model, device))}
    if not args.no_decider:
        models["decider"] = run_model("decider", lambda: L.DeciderBackend(L.DECIDER_MODEL, device))
    if not args.no_mapika:
        models["mapika"] = run_model("mapika", lambda: L.MapikaBackend(L.MAPIKA_MODEL, device))
    models = {k: v for k, v in models.items() if v is not None}
    nli = models["nli"]
    dec = models.get("decider")

    jev_skip = None
    key = os.environ.get("TYPESAFE_API_KEY")
    if args.no_jev:
        jev_skip = "--no-jev"
    elif not key:
        jev_skip = "no TYPESAFE_API_KEY set"
    else:
        log("Asking Jev...")
        jev = Jev(key)
        raws["jev"] = run_backend("jev", jev.raw, datasets, args.limit)
        if jev.dead and not any(raws["jev"][ds]["rows"] for ds in raws["jev"]):
            jev_skip = jev.dead
            del raws["jev"]

    # Build every setup from the raw runs (models ran once, setups are just different readouts).
    setups = {}
    def add(name, engine, parts, stasis):
        sets, errors = {}, {}
        all_rows = []
        for ds in datasets:
            per = [raws[p][ds] for p in parts]
            by_id = [{r[0]["id"]: r for r in x["rows"]} for x in per]
            common = [i for i in by_id[0] if all(i in b for b in by_id)]
            rows = []
            for i in common:
                item = by_id[0][i][0]
                variants = [v for b in by_id for v in b[i][1]]
                rows.append((item, variants, sum(b[i][2] for b in by_id)))
            errors[ds] = max(x["errors"] for x in per)
            m = score_set(engine, rows, stasis, args.threshold)
            if m:
                sets[ds] = m
            all_rows += rows
        sets["ALL"] = score_set(engine, all_rows, stasis, args.threshold)
        setups[name] = {"sets": sets, "errors": errors}

    e = lambda bs: L.Engine(bs, stasis=True, threshold=args.threshold)
    add("nli plain", e([nli]), ["nli"], False)
    add("nli + stasis", e([nli]), ["nli"], True)
    if "decider" in raws:
        add("decider plain", e([dec]), ["decider"], False)
        add("decider + stasis", e([dec]), ["decider"], True)
        add("nli+decider + stasis", e([nli, dec]), ["nli", "decider"], True)
    if "mapika" in raws:
        mp = models["mapika"]
        add("mapika plain", e([mp]), ["mapika"], False)
        add("mapika + stasis", e([mp]), ["mapika"], True)
        if "decider" in raws:
            add("mapika+decider + stasis", e([mp, dec]), ["mapika", "decider"], True)
    if "jev" in raws:
        add("jev", e([nli]), ["jev"], False)
    else:
        setups["jev"] = {"skipped": jev_skip, "sets": {}}

    write_report(setups, "results.md", "results.json", args)
    log("\nDone. Wrote results.md and results.json")
    print(open("results.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
