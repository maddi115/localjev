"""
localjev v2 - a local, Jev-compatible decision engine with probability stasis.

WHAT IT DOES (same contract as Jev / Strands Decider):
    request:  {"state": text or JSON, "questions": {id: {type, instructions, criteria}}}
    types:    noul   -> yes/no probability
              choice -> pick one option, probabilities for all
              score  -> rate on an ordered scale (criteria listed low -> high)
    answers:  probabilities + a confidence + an action ("act" or "escalate")

HOW CONFIDENCE WORKS (copied from how real decision models do it):
    choice:  (N * top_prob - 1) / (N - 1)        0 = coin flip, 1 = certain
    score:   how tightly probability clusters on the scale
    noul:    |2p - 1|                             0 = 50/50, 1 = certain
    then:    calibration (a fitted "temperature" per question type)
    action:  "act" if confidence >= threshold (default 0.9), else "escalate"

PROBABILITY STASIS (--stasis):
    Ask every question several meaning-preserving ways:
      - different framings of the statement
      - your own "rewordings" listed in the question
      - options in a different order          (choice)
      - JSON keys in a different order        (structured states)
      - a second model, if you enable one     (--backends nli,decider)
    Final probabilities  = average over all variants
    Final confidence     = base confidence x agreement x (1 - spread)
        agreement = share of variants that picked the same winner
        spread    = how far variants drift from the average (0..1)
    So an answer that flips when you reword it can never be "act".

IDEAS TAKEN FROM "LATENT LATTICE LANGUAGE MODELS" (MIT/Adobe, 2026):
    two axes      stasis now runs on two axes, like the paper's vertical + horizontal latents:
                  across variants of one moment (--stasis) and across time (stream)
    horizontal    `stream` keeps a running state per topic with the paper's gated update
                  r = g * r_prev + (1 - g) * p_now, so stasis over time is a real number
    zero-init     calibration starts every extra variant at weight 0 (= plain mode) and only
                  turns on what helps, so stasis can't start out worse than plain
    refiner       --refine N feeds confident answers back as facts and re-asks the rest,
                  like the paper's top-to-bottom Jacobi sweeps

COMMANDS:
    python localjev.py run  request.json [--stasis] [--refine 2]
    python localjev.py test tests/       [--stasis]      grade labeled tests
    python localjev.py calibrate ~/jevbench/datasets/public --stasis
                                                         learn weights + temperatures
    python localjev.py stream questions.json messages.jsonl [--stasis]
                                                         stasis over time
    python localjev.py distill --backends mapika         train a tiny student from a big teacher
    python localjev.py test tests/ --backends student    grade the student
    python localjev.py train-stasis                      train StasisNet from scratch (uses distill's cache)
    python localjev.py test tests/ --backends stasisnet  grade StasisNet
    python localjev.py test tests/ --backends student,mapika --cascade
                                                         student first, Mapika only when unsure
    python localjev.py serve             [--stasis]      Jev-style API on :8000
    (python localjev.py request.json still works)

BACKENDS:
    nli      DeBERTa zero-shot model (small, not an LLM) - default
    decider  Strands Decider 2B (pip install strands-decider) - optional
    mapika   Mapika decider-2b (pip install decider-ai) - optional, best public score
    student  a tiny fast copy of any of the above, made with `distill` (milliseconds, no warm-up)
    stasisnet  a tiny model trained FROM SCRATCH with stasis built into its weights (`train-stasis`)
"""
import argparse
import glob
import json
import math
import os
import random
import statistics
import sys
import time

NLI_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
DECIDER_MODEL = "StrandsAgents/strands-decider-2B-hobson-v19"
MAPIKA_MODEL = "Mapika/decider-2b"
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
EPS = 1e-6
STABLE_SD = 0.10   # stasis over time: spread at or under this = STABLE
SHAKY_SD = 0.25

# Statement framings for the NLI model (meaning must not change).
NLI_FRAMES = [
    "{}",
    "It is true that {}",
    "This text shows that {}",
    "Based on this message, {}",
    "The writer is saying that {}",
]
# Instruction framings for the decider model.
DECIDER_FRAMES = [
    "{}",
    "Based only on the message: {}",
    "Question: {}",
]


# ---------------------------------------------------------------- helpers

def lower_first(s):
    return s[:1].lower() + s[1:] if s else s


def sort_keys_deep(x):
    if isinstance(x, dict):
        return {k: sort_keys_deep(x[k]) for k in sorted(x)}
    if isinstance(x, list):
        return [sort_keys_deep(v) for v in x]
    return x


def shuffle_keys_deep(x, rng):
    if isinstance(x, dict):
        keys = list(x)
        rng.shuffle(keys)
        return {k: shuffle_keys_deep(x[k], rng) for k in keys}
    if isinstance(x, list):
        return [shuffle_keys_deep(v, rng) for v in x]
    return x


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def normalize(d):
    t = sum(d.values())
    if t <= 0:
        return {k: 1.0 / len(d) for k in d}
    return {k: v / t for k, v in d.items()}


def temper(d, T):
    """Temperature scaling on a probability distribution: p^(1/T), renormalized."""
    if abs(T - 1.0) < 1e-9:
        return dict(d)
    return normalize({k: max(v, EPS) ** (1.0 / T) for k, v in d.items()})


def argmax(d):
    return max(d, key=d.get)


def choice_confidence(probs):
    n = len(probs)
    if n <= 1:
        return 1.0
    return max(0.0, min(1.0, (n * max(probs) - 1.0) / (n - 1.0)))


def score_confidence(probs):
    """TypeSafe's Score confidence: 1 - (expected distance from the most likely level) /
    (mean distance of the levels from the middle of the scale), floored at 0."""
    n = len(probs)
    if n <= 1:
        return 1.0
    t = sum(probs) or 1.0
    p = [x / t for x in probs]
    k = max(range(n), key=p.__getitem__)
    spread = sum(x * abs(i - k) for i, x in enumerate(p))
    uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, min(1.0, 1.0 - spread / uniform))


def outcome_keys(q):
    kind = q.get("type")
    if kind == "noul":
        return ["true", "false"]
    if kind == "choice":
        return list(q["criteria"])
    if kind == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    raise ValueError(f"type '{kind}' not supported (use noul, choice or score)")


def validate(q, qid):
    kind = q.get("type")
    if kind not in ("noul", "choice", "score"):
        raise ValueError(f"{qid}: type must be noul, choice or score")
    if not q.get("instructions"):
        raise ValueError(f"{qid}: needs 'instructions'")
    if kind == "choice" and (not isinstance(q.get("criteria"), dict) or len(q["criteria"]) < 2):
        raise ValueError(f"{qid}: choice needs 'criteria' with 2+ options")
    if kind == "score" and (not isinstance(q.get("criteria"), list) or len(q["criteria"]) < 2):
        raise ValueError(f"{qid}: score needs 'criteria' as a list, low -> high")


def variant_category(label):
    """'nli | frame: It is true that {}' -> 'nli|frame'"""
    backend, _, rest = label.partition(" | ")
    kind = rest.split(":")[0].strip()
    if kind.startswith("json"):
        kind = "json"
    return f"{backend}|{kind}"


def fact_text(q, ans):
    if q["type"] == "noul":
        return f"{q['instructions'].rstrip('?.')}: {'yes' if ans['decision'] else 'no'}"
    if q["type"] == "choice":
        return f"{q['instructions'].rstrip('?.')}: {q['criteria'].get(ans['choice']) or ans['choice']}"
    return f"{q['instructions'].rstrip('?.')}: {ans['legend'][str(round(ans['score']))]}"


def add_facts(state, facts):
    if isinstance(state, str):
        return state + "\n\nAlready established:\n- " + "\n- ".join(facts)
    if isinstance(state, dict):
        return {**state, "already_established": facts}
    return {"input": state, "already_established": facts}


# ---------------------------------------------------------------- backends
# Each backend turns (state, question) pairs into probability distributions
# over the question's outcome keys. Everything else is shared.

class NLIBackend:
    name = "nli"
    frames = NLI_FRAMES

    def __init__(self, model_name, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.ent = next(i for i, l in labels.items() if "entail" in l and not l.startswith("not"))
        contra = [i for i, l in labels.items() if "contradict" in l]
        others = [i for i in labels if i != self.ent]
        self.neg = contra[0] if contra else others[0]
        self.model_name = model_name

    def _logits(self, pairs, batch=16):
        out = []
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                out.append(self.model(**enc).logits.float().cpu())
        return self.torch.cat(out) if out else self.torch.empty(0)

    def _statements(self, q, frame):
        """Hypotheses to check for one question, plus how to read them back."""
        def f(s):
            s = (s or "").strip()
            return s if frame == "{}" else frame.format(lower_first(s))
        kind = q["type"]
        if kind == "noul":
            crit = q.get("criteria") or {}
            t, fa = (crit.get("true") or "").strip(), (crit.get("false") or "").strip()
            if t and fa:
                return [f(t), f(fa)], "noul_pair"
            return [f(q["instructions"])], "noul"
        if kind == "choice":
            return [f(v or k) for k, v in q["criteria"].items()], "exclusive"
        return [f(level) for level in q["criteria"]], "exclusive"

    def distributions(self, states, items):
        pairs, plan = [], []
        for si, q, frame in items:
            hyps, mode = self._statements(q, frame)
            start = len(pairs)
            pairs += [(state_text(states[si]), h) for h in hyps]
            plan.append((q, mode, start, len(hyps)))
        logits = self._logits(pairs)
        torch = self.torch
        results = []
        for q, mode, start, n in plan:
            rows = logits[start:start + n]
            keys = outcome_keys(q)
            if mode == "noul":
                p = torch.sigmoid(rows[0, self.ent] - rows[0, self.neg]).item()
                results.append({"true": p, "false": 1 - p})
            elif mode == "noul_pair":
                pt = torch.sigmoid(rows[0, self.ent] - rows[0, self.neg]).item()
                pf = torch.sigmoid(rows[1, self.ent] - rows[1, self.neg]).item()
                p = pt / (pt + pf + EPS)
                results.append({"true": p, "false": 1 - p})
            else:  # exclusive: softmax over the options' entailment logits
                probs = torch.softmax(rows[:, self.ent], dim=0).tolist()
                results.append(dict(zip(keys, probs)))
        return results


class DeciderBackend:
    name = "decider"
    frames = DECIDER_FRAMES

    def __init__(self, checkpoint, device):
        from strands_decider.infer import load_engine
        from strands_decider.schema import SystemOneRequest

        self.Request = SystemOneRequest
        self.engine = load_engine(checkpoint, device=device)
        self.model_name = checkpoint

    @staticmethod
    def _clean(q, frame):
        out = {"type": q["type"], "instructions": frame.format(q["instructions"])}
        if q.get("criteria") is not None:
            out["criteria"] = q["criteria"]
        return out

    def distributions(self, states, items):
        by_state = {}
        for idx, (si, q, frame) in enumerate(items):
            by_state.setdefault(si, []).append((idx, q, frame))
        results = [None] * len(items)
        for si, group in by_state.items():
            qs = {f"v{idx}": self._clean(q, frame) for idx, q, frame in group}
            resp = self.engine.evaluate(self.Request.model_validate({"state": states[si], "questions": qs}))
            for idx, q, _ in group:
                a = resp.answers[f"v{idx}"]
                if q["type"] == "noul":
                    p = float(a.noul)
                    results[idx] = {"true": p, "false": 1 - p}
                elif q["type"] == "choice":
                    results[idx] = normalize({k: float(a.probabilities.get(k, 0.0)) for k in q["criteria"]})
                else:
                    probs = dict(a.probabilities)
                    levels = q["criteria"]
                    if all(str(k).isdigit() for k in probs):
                        d = {str(i): float(probs.get(str(i), 0.0)) for i in range(len(levels))}
                    else:
                        d = {str(i): float(probs.get(lv, 0.0)) for i, lv in enumerate(levels)}
                    results[idx] = normalize(d)
        return results


class MapikaBackend(DeciderBackend):
    """Mapika decider-2b (pip install decider-ai). Scored 76% on JevBench's public set."""
    name = "mapika"

    def __init__(self, checkpoint, device):
        from decider.infer import Decider
        self.dec = Decider(checkpoint, device=device, use_graphs=os.environ.get("MAPIKA_GRAPHS") == "1")
        self.model_name = checkpoint

    def distributions(self, states, items):
        by_state = {}
        for idx, (si, q, frame) in enumerate(items):
            by_state.setdefault(si, []).append((idx, q, frame))
        results = [None] * len(items)
        for si, group in by_state.items():
            qs = {f"v{idx}": self._clean(q, frame) for idx, q, frame in group}
            resp = self.dec.system_one(states[si], qs)
            for idx, q, _ in group:
                a = resp["answers"][f"v{idx}"]
                if q["type"] == "noul":
                    p = float(a["noul"])
                    results[idx] = {"true": p, "false": 1 - p}
                elif q["type"] == "choice":
                    results[idx] = normalize({k: float(a["probabilities"].get(k, 0.0)) for k in q["criteria"]})
                else:
                    results[idx] = normalize({str(i): float(a["probabilities"].get(str(i), 0.0))
                                              for i in range(len(q["criteria"]))})
        return results


# ---------------------------------------------------------------- engine

class Engine:
    def __init__(self, backends, stasis=False, threshold=0.9, calibration=None, refine=0, cascade=False,
                 first_threshold=None):
        self.backends = backends
        self.cascade = cascade
        self.first_threshold = first_threshold  # cascade: confidence the fast layers need to answer on their own
        self.stasis = stasis
        self.threshold = threshold
        self.cal = calibration or {}
        self.refine = refine

    def calibration_key(self, stasis=None):
        stasis = self.stasis if stasis is None else stasis
        return "+".join(b.name for b in self.backends) + ("|stasis" if stasis else "|plain")

    def _cal_entry(self, stasis=None):
        entry = self.cal.get(self.calibration_key(stasis), {})
        if entry and "temps" not in entry:  # old calibration.json format: {kind: T}
            entry = {"temps": entry}
        return entry

    def temperature(self, kind, stasis=None):
        return self._cal_entry(stasis).get("temps", {}).get(kind, 1.0)

    def weight(self, label, stasis=None):
        """How much one variant counts. The first model's base wording is always 1.
        Without calibration every variant counts 1. After calibration, weights are
        learned starting from 0 (zero-init), so stasis starts identical to plain
        and only adds variants that actually help."""
        cat = variant_category(label)
        anchor = cat == f"{self.backends[0].name}|base"
        weights = self._cal_entry(stasis).get("weights")
        if weights is None:
            return 1.0
        return weights.get(cat, 1.0 if anchor else 0.0)

    # ---- variant generation
    def _state_variants(self, state):
        base = sort_keys_deep(state)  # sorted keys: same content -> same answer
        states = [base]
        if self.stasis and isinstance(state, (dict, list)):
            for seed in (1, 2):
                states.append(shuffle_keys_deep(base, random.Random(seed)))
        return states

    def _variants(self, backend, q, n_states):
        """(label, state_index, question, frame) for one backend."""
        out = [("base", 0, q, "{}")]
        if not self.stasis:
            return out
        if getattr(backend, "fixed_questions", False):  # the student: only the message can vary
            return out + [(f"json key order #{si}", si, q, "{}") for si in range(1, n_states)]
        for fr in backend.frames[1:]:
            out.append((f"frame: {fr}", 0, q, fr))
        # The NLI model only reads noul instructions and ignores option order,
        # so those variants would be exact copies and fake extra agreement.
        reads_instructions = backend.name != "nli" or q["type"] == "noul"
        reads_order = backend.name != "nli"
        for alt in (q.get("rewordings", []) if reads_instructions else []):
            q2 = dict(q, instructions=alt)
            if q["type"] == "noul":
                q2.pop("criteria", None)
            out.append((f"reworded: {alt}", 0, q2, "{}"))
        if q["type"] == "choice" and reads_order:
            keys = list(q["criteria"])
            orders = [keys[::-1]] + ([keys[1:] + keys[:1]] if len(keys) > 2 else [])
            for order in orders:
                q2 = dict(q, criteria={k: q["criteria"][k] for k in order})
                out.append((f"option order: {','.join(order)}", 0, q2, "{}"))
        for si in range(1, n_states):
            out.append((f"json key order #{si}", si, q, "{}"))
        return out

    def raw(self, request, backends=None):
        """Every variant's distribution for every question: {qid: [(label, dist), ...]}."""
        questions = request["questions"]
        for qid, q in questions.items():
            validate(q, qid)
        states = self._state_variants(request.get("state", ""))
        out = {qid: [] for qid in questions}
        for b in (backends if backends is not None else self.backends):
            items, owners = [], []
            for qid, q in questions.items():
                for label, si, q2, frame in self._variants(b, q, len(states)):
                    items.append((si, q2, frame))
                    owners.append((qid, f"{b.name} | {label}"))
            for (qid, label), dist in zip(owners, b.distributions(states, items)):
                out[qid].append((label, dist))
        return out

    # ---- turning variants into one answer
    def summarize(self, q, variants, stasis=None):
        stasis = self.stasis if stasis is None else stasis
        kind = q["type"]
        if not stasis:
            variants = variants[:1]
        # Models with built-in stasis (StasisNet) attach their self-check score to each answer.
        self_checks = [d.get("__stasis__") for _, d in variants]
        variants = [(lab, {k: v for k, v in d.items() if k != "__stasis__"}) for lab, d in variants]
        T = self.temperature(kind, stasis)
        weighted = [(label, temper(d, T), self.weight(label, stasis)) for label, d in variants]
        weighted = [x for x in weighted if x[2] > 0] or [(variants[0][0], temper(variants[0][1], T), 1.0)]
        dists = [d for _, d, _ in weighted]
        ws = [w for _, _, w in weighted]
        W = sum(ws)
        keys = list(dists[0])
        mean = {k: sum(w * d[k] for d, w in zip(dists, ws)) / W for k in keys}
        win = argmax(mean)

        if kind == "noul":
            p = mean["true"]
            base_conf = abs(2 * p - 1)
            ans = {"type": "noul", "noul": round(p, 4), "decision": p >= 0.5}
        elif kind == "choice":
            base_conf = choice_confidence(list(mean.values()))
            ans = {"type": "choice", "choice": win, "probabilities": {k: round(v, 4) for k, v in mean.items()}}
        else:
            probs = [mean[str(i)] for i in range(len(q["criteria"]))]
            base_conf = score_confidence(probs)
            ans = {"type": "score",
                   "score": round(sum(i * p for i, p in enumerate(probs)), 4),
                   "legend": {str(i): lv for i, lv in enumerate(q["criteria"])},
                   "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)}}

        conf = base_conf
        threshold = self.threshold
        if self.cascade and self.first_threshold is not None and len(self.backends) > 1:
            if variants[0][0].split(" | ")[0] != self.backends[-1].name:  # answered by a fast layer
                threshold = self.first_threshold
        if len(dists) > 1:
            agreement = sum(w for d, w in zip(dists, ws) if argmax(d) == win) / W
            spread = sum(w * 0.5 * sum(abs(d[k] - mean[k]) for k in keys) for d, w in zip(dists, ws)) / W
            p_win = [d[win] for d in dists]
            mean_win = sum(w * p for p, w in zip(p_win, ws)) / W
            var = sum(w * (p - mean_win) ** 2 for p, w in zip(p_win, ws)) / W
            conf = base_conf * agreement * (1 - spread)
            ans["stasis"] = {
                "variants": len(dists),
                "agreement": round(agreement, 4),
                "spread": round(spread, 4),
                "base_confidence": round(base_conf, 4),
                "stasis_score": round(mean_win / (1 + var), 4),  # your original formula
                "runs": {label: round(d[win], 4) for label, d, _ in weighted},
            }
        checks = [c for c in self_checks if c is not None]
        if checks:
            built_in = sum(checks) / len(checks)
            conf *= built_in
            ans["self_check"] = round(built_in, 4)
        ans["confidence"] = round(conf, 4)
        ans["action"] = "act" if conf >= threshold else "escalate"
        if self.cascade:
            ans["answered_by"] = variants[0][0].split(" | ")[0]
        return ans

    def solve(self, request):
        """raw(), plus refine sweeps (Jacobi-style feedback from the Lattice paper):
        answers confident enough to act on are written back into the state as
        established facts, and the still-uncertain questions are asked again."""
        questions = request["questions"]
        if self.cascade:
            raw = self._cascade(request)
        else:
            raw = self.raw(request)
        for _ in range(self.refine):
            answers = {qid: self.summarize(q, raw[qid]) for qid, q in questions.items()}
            facts = [fact_text(questions[qid], a) for qid, a in answers.items() if a["action"] == "act"]
            pending = [qid for qid, a in answers.items() if a["action"] != "act"]
            if not facts or not pending:
                break
            sub = {"state": add_facts(request.get("state", ""), facts),
                   "questions": {qid: questions[qid] for qid in pending}}
            raw.update(self.raw(sub))
        return raw

    def _cascade(self, request):
        """Fast model first. Only questions it isn't confident about go to the next (bigger) model."""
        questions = request["questions"]
        raw = self.raw(request, backends=self.backends[:1])
        for b in self.backends[1:]:
            unsure = [qid for qid, q in questions.items() if self.summarize(q, raw[qid])["action"] != "act"]
            if not unsure:
                break
            sub = {"state": request.get("state", ""), "questions": {qid: questions[qid] for qid in unsure}}
            raw.update(self.raw(sub, backends=[b]))
        return raw

    def evaluate(self, request):
        t0 = time.time()
        raw = self.solve(request)
        answers = {qid: self.summarize(q, raw[qid]) for qid, q in request["questions"].items()}
        return {
            "model": "localjev-v2 (" + " + ".join(b.model_name for b in self.backends) + ")",
            "stasis": self.stasis,
            "answers": answers,
            "seconds": round(time.time() - t0, 3),
        }


# ---------------------------------------------------------------- grading & calibration

def is_correct(q, ans, label):
    if q["type"] == "noul":
        return ans["decision"] == bool(label)
    if q["type"] == "choice":
        return ans["choice"] == label
    return round(ans["score"]) == int(label)


def top_prob(q, ans):
    if q["type"] == "noul":
        return max(ans["noul"], 1 - ans["noul"])
    if q["type"] == "choice":
        return ans["probabilities"][ans["choice"]]
    return ans["probabilities"][str(round(ans["score"]))]


def true_key(q, label):
    if q["type"] == "noul":
        return "true" if label else "false"
    return str(label) if q["type"] == "score" else label


def jevbench_label(q, expected):
    if q["type"] == "noul":
        return str(expected).lower() in ("yes", "true", "1")
    if q["type"] == "score":
        return int(expected)
    return expected


def load_cases(paths):
    """Labeled cases from folders, .json test files, or JevBench .jsonl files."""
    if isinstance(paths, str):
        paths = [paths]
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.json")) + glob.glob(os.path.join(p, "*.jsonl")))
        else:
            files.append(p)
    cases = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        req = {"state": r["state"], "questions": {"q": r["question"]}}
                        cases.append((r.get("id", os.path.basename(path)), req, {"q": jevbench_label(r["question"], r["expected"])}))
                continue
            req = json.load(f)
        labels = req.get("labels", {})
        if not labels:
            print(f"skipping {path}: no \"labels\"", file=sys.stderr)
            continue
        cases.append((os.path.basename(path), req, labels))
    return cases


def collect(engine, cases):
    rows = []  # (file, qid, q, label, variants)
    for i, (name, req, labels) in enumerate(cases, 1):
        raw = engine.solve(req)
        for qid, q in req["questions"].items():
            if qid in labels:
                rows.append((name, qid, q, labels[qid], raw[qid]))
        if i % 25 == 0:
            print(f"  ...{i}/{len(cases)}", file=sys.stderr)
    return rows


def metrics(engine, rows, stasis):
    n = len(rows)
    correct = acted = acted_right = acted_wrong = 0
    brier, bins = 0.0, [[0, 0.0, 0.0] for _ in range(5)]
    wrong_confs, confs = [], []
    for _, _, q, label, variants in rows:
        ans = engine.summarize(q, variants, stasis)
        ok = is_correct(q, ans, label)
        confs.append(ans["confidence"])
        if not ok:
            wrong_confs.append(ans["confidence"])
        p = top_prob(q, ans)
        correct += ok
        brier += (p - ok) ** 2
        b = bins[min(4, int(p * 5))]
        b[0] += 1; b[1] += p; b[2] += ok
        if ans["action"] == "act":
            acted += 1
            acted_right += ok
            acted_wrong += not ok
    ece = sum(abs(b[1] - b[2]) for b in bins if b[0]) / max(n, 1)
    # Lowest cutoff that would have acted on zero wrong answers in this test set.
    safe = min(1.0, round(max(wrong_confs) + 0.01, 2)) if wrong_confs else 0.0
    return {
        "safe_threshold": safe,
        "acted_at_safe": sum(c >= safe for c in confs),
        "questions": n,
        "accuracy": correct / max(n, 1),
        "acted": acted,
        "accuracy_when_acted": acted_right / acted if acted else float("nan"),
        "acted_but_wrong": acted_wrong,
        "brier": brier / max(n, 1),
        "ece": ece,
    }


def print_metrics(title, m):
    print(f"\n== {title}")
    print(f"  accuracy             {m['accuracy']:.0%}  ({m['questions']} questions)")
    print(f"  acted automatically  {m['acted']}/{m['questions']}  ({m['acted'] / max(m['questions'], 1):.0%}), the rest escalated")
    awa = m["accuracy_when_acted"]
    print(f"  accuracy when acted  {'n/a' if awa != awa else f'{awa:.0%}'}")
    print(f"  ACTED BUT WRONG      {m['acted_but_wrong']}   <- the dangerous number, want 0")
    print(f"  brier / ece          {m['brier']:.3f} / {m['ece']:.3f}   <- lower = more honest probabilities")
    print(f"  safest threshold     {m['safe_threshold']:.2f}  (would act on {m['acted_at_safe']}/{m['questions']} with 0 wrong, on this test set)")


def cmd_test(engine, folder):
    t0 = time.time()
    rows = collect(engine, load_cases(folder))
    elapsed = time.time() - t0
    if not rows:
        sys.exit("no labeled tests found")
    print(f"{'file':<24}{'question':<18}{'expected':<16}{'got':<16}{'conf':>6}  action")
    for name, qid, q, label, variants in rows:
        ans = engine.summarize(q, variants)
        got = {"noul": lambda: f"{ans['decision']} ({ans['noul']:.2f})",
               "choice": lambda: ans["choice"],
               "score": lambda: f"{round(ans['score'])} ({ans['score']:.2f})"}[q["type"]]()
        mark = "OK " if is_correct(q, ans, label) else "XX "
        print(f"{name[:23]:<24}{qid[:17]:<18}{str(label)[:15]:<16}{got[:15]:<16}{ans['confidence']:>6.2f}  {mark}{ans['action']}")
    if engine.cascade:
        by = {}
        for *_, variants in rows:
            name = variants[0][0].split(" | ")[0]
            by[name] = by.get(name, 0) + 1
        print("\n== cascade: who answered")
        for name, n in by.items():
            print(f"  {name:<20} {n}/{len(rows)}  ({n / len(rows):.0%})")
        print_metrics("cascade", metrics(engine, rows, stasis=engine.stasis))
    else:
        print_metrics("plain (one wording, first model)", metrics(engine, rows, stasis=False))
        if engine.stasis:
            print_metrics("probability stasis (all variants)", metrics(engine, rows, stasis=True))
    print(f"\n  time to answer all questions: {elapsed:.1f}s (model loading not included)")


def blended_nll(rows, weights, temps, first_base, stasis):
    """Average log loss of the true answer under given variant weights and temperatures."""
    tot = 0.0
    for _, _, q, label, variants in rows:
        vs = variants if stasis else variants[:1]
        T = temps.get(q["type"], 1.0)
        tk = true_key(q, label)
        num = W = 0.0
        for lab, d in vs:
            d = {k: v for k, v in d.items() if k != "__stasis__"}
            cat = variant_category(lab)
            w = weights.get(cat, 1.0 if cat == first_base else 0.0)
            if w > 0:
                num += w * temper(d, T)[tk]
                W += w
        if W <= 0:
            return float("inf")
        tot -= math.log(max(num / W, EPS))
    return tot / len(rows)


def cmd_calibrate(engine, paths):
    rows = collect(engine, load_cases(paths))
    if not rows:
        sys.exit("no labeled questions found")
    key = engine.calibration_key()
    first_base = f"{engine.backends[0].name}|base"
    weights, temps = {}, {}

    if engine.stasis:
        # Zero-init (from the Lattice paper): every extra variant starts at weight 0,
        # so we start exactly at plain, then only turn on what lowers the loss.
        others = sorted({variant_category(lab) for r in rows for lab, _ in r[4]} - {first_base})
        cats = [first_base] + others  # the first model's base can be turned down too
        weights = {first_base: 1.0, **{c: 0.0 for c in others}}
        grid = [0.0, 0.25, 0.5, 1.0, 2.0]
        start = blended_nll(rows, weights, temps, first_base, True)
        for _ in range(2):  # two rounds of coordinate descent
            for c in cats:
                cur = blended_nll(rows, weights, temps, first_base, True)
                for v in grid:  # only move off the current value for a real improvement
                    trial = blended_nll(rows, {**weights, c: v}, temps, first_base, True)
                    if trial < cur - 1e-4:
                        weights[c], cur = v, trial
        print(f"variant weights (0 = ignored, 1 = same as base). log loss {start:.3f} -> "
              f"{blended_nll(rows, weights, temps, first_base, True):.3f}")
        for c in cats:
            print(f"  {c:<28} {weights[c]}")
        print()

    tgrid = [round(0.25 * i, 2) for i in range(1, 25)]  # 0.25 .. 6.0
    for kind in ("noul", "choice", "score"):
        kr = [r for r in rows if r[2]["type"] == kind]
        if not kr:
            continue
        f = lambda T: blended_nll(kr, weights, {kind: T}, first_base, engine.stasis)
        best = 1.0
        for T in tgrid:  # stay at 1.0 (no change) unless another value is really better
            if f(T) < f(best) - 1e-4:
                best = T
        temps[kind] = best
        print(f"{kind:<7} {len(kr):>4} questions   temperature {temps[kind]:<5}  (log loss {f(1.0):.3f} -> {f(temps[kind]):.3f})")
        if len(kr) < 30:
            print(f"        warning: only {len(kr)} {kind} examples, more labeled data = more trustworthy calibration")

    before = metrics(engine, rows, engine.stasis)
    engine.cal[key] = {"temps": temps, "weights": weights} if engine.stasis else {"temps": temps}
    after = metrics(engine, rows, engine.stasis)
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as fh:
        json.dump(engine.cal, fh, indent=2)
    print(f"\nsaved to {CALIBRATION_FILE} under \"{key}\"")
    print("note: these numbers are on the SAME data it learned from. Grade on different data with `test`.")
    print_metrics("before calibration", before)
    print_metrics("after calibration", after)


def clock(minutes):
    return f"{int(minutes // 60) % 24:02d}:{int(minutes % 60):02d}"


def parse_time(t, default):
    """Minutes. Accepts numbers or "HH:MM"."""
    if t is None:
        return float(default)
    if isinstance(t, str) and ":" in t:
        h, m = t.split(":")[:2]
        return int(h) * 60 + float(m)
    return float(t)


class Horizon:
    """Probability stasis over TIME: one running state per (topic, question).

    This is the Lattice paper's horizontal latent, applied to decisions:
        r <- prior + exp(-dt / tau) * (r - prior)    time: fade back toward "no idea"
        a = 1 - exp(-lam * relevance * confidence)   gate: how much this message counts
        r <- (1 - a) * r + a * p_now                 gated affine update (Mamba-style)
    - a long gap since the last report  -> forget more (dt / tau)
    - a confident new report            -> move more (lam * confidence)
    - shaky reports barely move it, so one weird message can't flip the state
    A running variance tracks how much reports disagree with the state, which gives
    "stasis over time": a likely outcome that keeps being confirmed.

    v3.1: time decay now pulls toward "no idea" instead of toward the new message, and a
    question can say what it is "about". Messages that aren't about it don't move it, so
    "Great game!" can no longer clear a hazard (absence of mention is not evidence)."""

    def __init__(self, keys, t):
        self.r = {k: 1.0 / len(keys) for k in keys}  # start from "no idea"
        self.v = 0.0
        self.t = t
        self.n = 0

    def update(self, dist, conf, t, tau, lam, relevance=1.0):
        # 1. time: memory fades back toward "no idea" (the learned initial state in the paper)
        dt = max(0.0, t - self.t)
        keep = math.exp(-dt / tau)
        prior = 1.0 / len(self.r)
        for k in self.r:
            self.r[k] = prior + keep * (self.r[k] - prior)
        self.v *= keep
        # 2. evidence: input-dependent gate. Off-topic or shaky messages barely move the state.
        a = 1 - math.exp(-lam * relevance * conf)
        win_before = argmax(self.r)
        surprise = dist[win_before] - self.r[win_before]
        for k in self.r:
            self.r[k] = (1 - a) * self.r[k] + a * dist[k]
        self.v = (1 - a) * self.v + a * surprise ** 2
        self.t, self.n = t, self.n + (1 if a > 0.05 else 0)
        return a

    def read(self):
        win = argmax(self.r)
        sd = math.sqrt(self.v)
        held = self.r[win]
        score = held / (1 + 10 * sd)
        if self.n < 2:
            verdict = "NEW"
        elif sd <= STABLE_SD and held >= 0.7:
            verdict = "STABLE"      # likely AND keeps being confirmed: your probability stasis
        elif sd <= STABLE_SD:
            verdict = "UNSURE"      # consistent, but not likely enough to count
        elif sd <= SHAKY_SD:
            verdict = "SHAKY"
        else:
            verdict = "UNSTABLE"
        return win, held, sd, score, verdict


def outcome_label(q, key):
    if q["type"] == "noul":
        return "yes" if key == "true" else "no"
    if q["type"] == "score":
        return f"L{key}"
    return key


def cmd_stream(engine, questions_path, messages_path, tau, lam):
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    questions = questions.get("questions", questions)
    trackers = {}
    print(f"tau={tau} min (memory length), lam={lam} (how much confidence moves the state)\n")
    print(f"{'time':>6}  {'topic':<14}{'question':<16}{'this msg':<14}{'conf':>5} {'rel':>5} {'moved':>5}  {'running state':<18}{'stasis':<9}")
    with open(messages_path, encoding="utf-8") as f:
        for i, line in enumerate(l for l in f if l.strip()):
            m = json.loads(line)
            t = parse_time(m.get("t", m.get("time")), i)
            topic = str(m.get("topic", "all"))
            asked = {qid: {k: v for k, v in q.items() if k != "about"} for qid, q in questions.items()}
            for qid, q in questions.items():
                if q.get("about"):
                    asked[f"__about_{qid}"] = {"type": "noul", "instructions": f"This message is about {q['about']}"}
            raw = engine.solve({"state": m["state"], "questions": asked})
            for qid, q in questions.items():
                ans = engine.summarize(asked[qid], raw[qid])
                rel = 1.0
                if q.get("about"):
                    rel = engine.summarize(asked[f"__about_{qid}"], raw[f"__about_{qid}"])["noul"]
                dist = {"true": ans["noul"], "false": 1 - ans["noul"]} if q["type"] == "noul" else \
                    {k: float(v) for k, v in ans["probabilities"].items()}
                h = trackers.setdefault((topic, qid), Horizon(outcome_keys(q), t))
                a = h.update(dist, ans["confidence"], t, tau, lam, rel)
                win, held, sd, score, verdict = h.read()
                now = outcome_label(q, argmax(dist))
                print(f"{clock(t):>6}  {topic[:13]:<14}{qid[:15]:<16}{now + f' ({dist[argmax(dist)]:.2f})':<14}"
                      f"{ans['confidence']:>5.2f} {rel:>5.2f} {a:>5.2f}  {outcome_label(q, win) + f' {held:.2f} +-{sd:.2f}':<18}{verdict}")
    print("\n== final state per topic")
    for (topic, qid), h in sorted(trackers.items()):
        win, held, sd, score, verdict = h.read()
        q = questions[qid]
        print(f"  {topic:<14}{qid:<16}{outcome_label(q, win):<8} held {held:.2f}  spread {sd:.2f}  "
              f"time-stasis {score:.2f}  {verdict}  ({h.n} reports)")


def cmd_serve(engine, port):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True, "stasis": engine.stasis})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "POST to /v1/systemone"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, engine.evaluate(req))
            except Exception as e:  # report bad requests instead of crashing the server
                self._send(400, {"error": str(e)})

        def log_message(self, fmt, *args):
            print(f"[serve] {fmt % args}", file=sys.stderr)

    print(f"Serving on http://localhost:{port}/v1/systemone  (Ctrl+C to stop)", file=sys.stderr)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


# ---------------------------------------------------------------- setup

def pick_device(device):
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("GPU asked for, but torch can't see it.")
    name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Running on: {name}", file=sys.stderr)
    return device


def build_engine(args):
    device = pick_device(args.device)
    backends = []
    for name in args.backends.split(","):
        name = name.strip()
        print(f"Loading {name} model...", file=sys.stderr)
        if name == "nli":
            backends.append(NLIBackend(args.model, device))
        elif name == "decider":
            try:
                backends.append(DeciderBackend(args.decider_model, device))
            except ImportError:
                sys.exit("decider backend needs: pip install strands-decider")
        elif name == "mapika":
            try:
                backends.append(MapikaBackend(args.mapika_model, device))
            except ImportError:
                sys.exit("mapika backend needs: pip install decider-ai")
        elif name == "stasisnet" or name.startswith("stasisnet:"):
            from stasisnet import StasisNetBackend
            folder = name.split(":", 1)[1] if ":" in name else "stasisnet"
            if not os.path.exists(os.path.join(folder, "model.pt")):
                sys.exit(f"no StasisNet in {folder}/ yet. Train one: python localjev.py train-stasis")
            backends.append(StasisNetBackend(folder, device))
        elif name == "student" or name.startswith("student:"):
            from distill import StudentBackend
            folder = name.split(":", 1)[1] if ":" in name else "student"
            if not os.path.exists(os.path.join(folder, "student.json")):
                sys.exit(f"no student in {folder}/ yet. Train one: python localjev.py distill --backends mapika")
            backends.append(StudentBackend(folder, device))
        else:
            sys.exit(f"unknown backend '{name}' (use nli, decider, mapika, student, stasisnet)")
    cal = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, encoding="utf-8") as f:
            cal = json.load(f)
    if args.cascade and len(backends) < 2:
        sys.exit("--cascade needs 2+ backends, fastest first, e.g. --backends student,mapika")
    return Engine(backends, stasis=args.stasis, threshold=args.threshold, calibration=cal, refine=args.refine,
                  cascade=args.cascade, first_threshold=args.first_threshold)


def main():
    argv = sys.argv[1:]
    if argv and argv[0].endswith(".json"):  # old style: python localjev.py request.json
        argv = ["run"] + argv

    ap = argparse.ArgumentParser(description="Local Jev-compatible decision engine with probability stasis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in [("run", "answer one request file"), ("test", "grade labeled tests"),
                           ("calibrate", "learn variant weights + temperatures"), ("serve", "Jev-style HTTP API"),
                           ("stream", "probability stasis over time on a message stream"),
                           ("distill", "teach a tiny fast student to copy a big model"),
                           ("train-stasis", "train StasisNet from scratch with built-in stasis")]:
        p = sub.add_parser(name, help=helptext)
        if name == "run":
            p.add_argument("request")
        elif name in ("test", "calibrate"):
            p.add_argument("paths", nargs="*", default=["tests"], help="folders, .json tests or JevBench .jsonl")
        elif name == "distill":
            p.add_argument("paths", nargs="*", default=["tests"], help="test folders whose questions to learn")
            p.add_argument("--questions", nargs="*", default=["stream/questions.json"],
                           help="extra question files to learn (default: stream/questions.json if present)")
            p.add_argument("--messages", nargs="*", help="your own example messages (.txt one per line, or .jsonl)")
            p.add_argument("--n", type=int, default=1500, help="how many training messages (default 1500)")
            p.add_argument("--out", default="student", help="where to save the student (default student/)")
            p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
            p.add_argument("--steps", type=int, default=500)
            p.add_argument("--seed", type=int, default=0)
        elif name == "train-stasis":
            p.add_argument("--cache", default="student/teacher_cache.jsonl", help="teacher answers saved by distill")
            p.add_argument("--out", default="stasisnet")
            p.add_argument("--epochs", type=int, default=8)
            p.add_argument("--d", type=int, default=256, help="model width")
            p.add_argument("--layers", type=int, default=4)
            p.add_argument("--heads", type=int, default=4)
            p.add_argument("--vocab", type=int, default=6000, help="tokenizer size (learned from your data)")
            p.add_argument("--batch", type=int, default=48)
            p.add_argument("--lr", type=float, default=5e-4)
            p.add_argument("--consistency", type=float, default=1.0, help="how hard to punish answers that move")
            p.add_argument("--seed", type=int, default=0)
        elif name == "stream":
            p.add_argument("questions", help="JSON file with the questions to track")
            p.add_argument("messages", help="JSONL: one {t, topic, state} per line")
            p.add_argument("--tau", type=float, default=30.0, help="memory length in minutes (default 30)")
            p.add_argument("--lam", type=float, default=2.0, help="how much a confident report moves the state")
        else:
            p.add_argument("--port", type=int, default=8000)
        p.add_argument("--refine", type=int, default=0,
                       help="refine sweeps: feed confident answers back as facts and re-ask the rest (try 2)")
        p.add_argument("--stasis", action="store_true", help="probability stasis: ask every question many ways")
        p.add_argument("--first-threshold", type=float, default=None,
                       help="cascade: confidence the fast model needs to answer on its own (default: same as --threshold)")
        p.add_argument("--cascade", action="store_true",
                       help="fastest backend first; only unsure questions go to the next one (e.g. --backends student,mapika)")
        p.add_argument("--backends", default="nli", help="any of nli, decider, mapika, comma separated")
        p.add_argument("--threshold", type=float, default=0.9, help="confidence needed to act (default 0.9)")
        p.add_argument("--model", default=NLI_MODEL, help="NLI model for the nli backend")
        p.add_argument("--decider-model", default=DECIDER_MODEL)
        p.add_argument("--mapika-model", default=MAPIKA_MODEL)
        p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args(argv)

    if args.cmd == "train-stasis":  # trains from cached teacher answers, needs no model loaded
        from stasisnet import cmd_train
        cmd_train(args)
        return
    engine = build_engine(args)
    try:
        dispatch(engine, args)
    except ValueError as e:  # bad request / untrained question: show the message, not a traceback
        sys.exit(f"error: {e}")


def dispatch(engine, args):
    if args.cmd == "run":
        with open(args.request, encoding="utf-8") as f:
            print(json.dumps(engine.evaluate(json.load(f)), indent=2))
    elif args.cmd == "test":
        cmd_test(engine, args.paths)
    elif args.cmd == "calibrate":
        cmd_calibrate(engine, args.paths)
    elif args.cmd == "stream":
        cmd_stream(engine, args.questions, args.messages, args.tau, args.lam)
    elif args.cmd == "distill":
        from distill import cmd_distill
        args.questions = [q for q in (args.questions or []) if os.path.exists(q)]
        cmd_distill(engine, args)
    else:
        cmd_serve(engine, args.port)


if __name__ == "__main__":
    main()
