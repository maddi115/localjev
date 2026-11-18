"""
distill.py - teacher -> student.

A big decision model (the teacher: Mapika, Amazon, ...) answers your questions on a
few thousand example messages. A tiny student learns to copy its probabilities.
After that only the student runs: no 4 GB model, no warm-up, milliseconds per message.

The student, for each question, is one formula:

    e = embed(message)                 384 numbers that capture the meaning (MiniLM)
    p = softmax(W @ e + b)             one row of W per answer option

Training is gradient descent on the teacher's soft answers (cross-entropy):
    error = p_student - p_teacher
    W    -= learning_rate * error (outer) e

Limits: the student only knows the questions it was trained on. Add a question -> distill again.

Used through localjev.py:
    python localjev.py distill --backends mapika            train  (saves to student/)
    python localjev.py test tests/ --backends student       grade the student
"""
import json
import math
import os
import random
import sys
import time

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- which question is which

def qsig(q):
    """A question's identity: type + instructions + options. Rewordings / 'about' are ignored."""
    crit = q.get("criteria")
    return json.dumps([q["type"], q["instructions"], crit], ensure_ascii=False, sort_keys=True)


def clean(q):
    out = {"type": q["type"], "instructions": q["instructions"]}
    if q.get("criteria") is not None:
        out["criteria"] = q["criteria"]
    return out


def collect_questions(test_paths, question_files):
    """Every distinct question from your tests and question files (plus stream 'about' checks)."""
    import localjev as L
    found = {}
    for name, req, _ in L.load_cases(test_paths) if test_paths else []:
        for q in req["questions"].values():
            found.setdefault(qsig(clean(q)), clean(q))
    for path in question_files or []:
        with open(path, encoding="utf-8") as f:
            qs = json.load(f)
        qs = qs.get("questions", qs)
        for q in qs.values():
            found.setdefault(qsig(clean(q)), clean(q))
            if q.get("about"):
                about = {"type": "noul", "instructions": f"This message is about {q['about']}"}
                found.setdefault(qsig(about), about)
    return list(found.values())


# ---------------------------------------------------------------- example messages

SLOTS = {
    "place": ["section 22", "section 210", "gate 14", "gate 7", "gate 11", "the east concourse", "row F", "row K",
              "the north entrance", "the food court", "parking lot B", "the upper deck", "suite level", "aisle 12"],
    "spot": ["stairs", "steps", "ramp", "floor", "walkway", "escalator", "bathroom floor", "seats"],
    "spill": ["big spill", "puddle of soda", "pile of broken glass", "wet spot", "mess of nachos", "beer spill"],
    "person": ["a kid", "an older lady", "a guy", "my friend", "someone", "a woman", "an usher", "a teenager"],
    "family": ["my dad", "my mom", "my grandpa", "my son", "my wife", "my buddy", "my sister"],
    "symptom": ["feels lightheaded and can barely stand", "is having trouble breathing", "fainted",
                "cut his hand pretty badly", "says her chest hurts", "is throwing up", "twisted an ankle on the steps"],
    "collapse": ["collapsed", "looks really out of it and confused", "fell down the stairs", "is shaking and sweating a lot",
                 "is bleeding from the head", "passed out in the heat"],
    "two": ["two guys", "a couple of fans", "some drunk guys", "two groups", "a man and a woman"],
    "fighting": ["shoving each other", "throwing punches", "about to fight", "screaming in each other's faces",
                 "fighting"],
    "item": ["blue tote bag", "phone", "wallet", "car keys", "jacket", "sunglasses", "kid's stuffed animal", "ID"],
    "seat": ["my seat in section 210", "the bench at gate 9", "seat 3 in row D", "the table at the food court"],
    "thing": ["the box office", "the team store", "the parking garage", "the kids zone", "the museum"],
    "facility": ["restroom", "water fountain", "first aid", "the ATM", "guest services", "the elevator",
                 "nursing room"],
    "policy": ["a clear bag", "an umbrella", "outside food", "a small backpack", "a camera"],
    "praise": ["they were super helpful", "they made our day", "amazing service", "so friendly and quick"],
    "food": ["a hot dog", "nachos", "drinks", "pizza", "a beer"],
    "wait": ["40 minutes", "half an hour", "an hour", "forever"],
    "problem": ["is out of order", "is filthy", "has no soap", "is flooded", "has been closed all game"],
    "purchase": ["parking pass", "tickets", "season pass", "food order", "seat upgrade"],
    "action": ["open my tickets", "try to log in", "add a card", "scan my pass", "transfer a ticket"],
    "group": ["a birthday party", "a company outing", "military families", "seniors", "a youth sports team"],
    "bad_event": ["my tickets disappeared", "it's raining on us", "the shuttle left without us",
                  "the app crashed again", "they lost my order"],
    "venue": ["stadium", "app", "place", "team"],
    "hour": ["midnight", "two hours after the game", "9pm", "after the fireworks"],
    "lot": ["team store", "north gate", "west garage", "kids zone"],
    "prize": ["$500 gift card", "free iPhone", "VIP upgrade", "cruise for two", "$1000 cash prize"],
    "link": ["bit.ly/claim-now", "this link", "the link below", "win-now.biz"],
    "sale": ["20% off jerseys this weekend", "buy one get one hats", "half price parking for members"],
    "friend_msg": ["are we still meeting at gate 7 before the game", "can you grab me a water on your way",
                   "running 10 minutes late, save my seat", "what section are you in"],
    "implied": ["is slurring and can't stand up straight", "is turning really pale and sweaty", "keeps zoning out and won't respond",
                "is breathing really fast and shaking", "is lying on the floor and not moving much", "says everything is spinning"],
    "weapon_threat": ["says he has a knife", "is threatening to beat someone up", "is waving a bottle at people",
                      "keeps yelling that he'll make people regret it", "is pushing people and saying he'll fight anyone"],
    "hedge_thing": ["food", "music", "new app", "parking setup", "halftime show"],
    "hedge": ["I wouldn't say I hate the {hedge_thing}, they're okay", "Not bad at all, the {hedge_thing} was decent",
              "I don't dislike the {hedge_thing}, I just liked the old ones better", "Honestly the {hedge_thing} isn't terrible"],
}

TEMPLATES = {
    "hazard": ["There's a {spill} on the {spot} in {place}.", "The {spot} near {place} are soaked, {person} almost slipped.",
               "Someone dumped a soda all over the {spot} at {place}.", "Watch out, the {spot} at {place} are really slippery.",
               "{person} just slipped on the {spot} by {place}.", "There's glass everywhere on the {spot} in {place}."],
    "fight": ["{two} are {fighting} near {place}.", "Things are getting heated at {place}, {two} are yelling and pushing.",
              "Security needed at {place}, {two} are {fighting}."],
    "threat": ["Some guy by {place} is screaming that people are going to get it.", "Someone at {place} is threatening people.",
               "There's a guy at {place} acting aggressive and scaring kids."],
    "medical": ["{family} {symptom} near {place}.", "{person} at {place} {collapse}.",
                "Need help at {place}, {family} {symptom}.", "The lady behind me in {place} {collapse}."],
    "lost": ["I lost my {item} at {place}.", "I left my {item} under {seat}.", "Has anyone turned in a {item}? I was at {place}."],
    "question": ["What time does {thing} open?", "Where is the nearest {facility}?", "Is {policy} allowed inside?",
                 "How do I get to {place}?", "When does {thing} close tonight?"],
    "compliment": ["Shoutout to the staff at {place}, {praise}.", "Just wanted to say thanks, the people at {place} were great.",
                   "Great game so far!", "Loving the atmosphere tonight."],
    "service": ["The {facility} at {place} {problem}.", "Waited {wait} for {food} at {place} and it was cold.",
                "My order has been wrong twice now, get me someone in charge.", "Why is the {facility} at {place} still closed?"],
    "crowd": ["The line for the {facility} near {place} is {wait} long.", "{place} is packed and nobody is directing people.",
              "Huge crowd stuck at {place}, people are getting squished.", "The line at {place} is moving now, thanks."],
    "billing": ["I got double charged for my {purchase}.", "My {purchase} payment went through but I never got it.",
                "Can I get a refund for my {purchase}?", "Why was I charged an extra fee on my {purchase}?"],
    "bug": ["The app keeps freezing when I {action}.", "I can't {action}, it just spins.", "Your website gives an error when I {action}."],
    "pricing": ["Do you offer discounts for {group}?", "How much is a {purchase} for {group}?", "Is there a deal on {purchase} this week?"],
    "sarcasm": ["Oh great, {bad_event}. Love this {venue}.", "Wow, {bad_event}. Best day ever.", "Fantastic, {bad_event}. Really impressive."],
    "negation": ["I don't want a refund, I just want to change my seat.", "Nobody is hurt, the {spot} at {place} are just wet.",
                 "It's not like the food is terrible, it's just a little pricey.", "No emergency, I just have a question about {thing}."],
    "spam": ["Congrats!! You've been selected for a {prize}, click {link} to claim.", "Your account will be suspended, verify now at {link}.",
             "WIN a {prize} now!!! Reply YES to claim.", "Final notice: claim your {prize} within 24 hours at {link}."],
    "promo": ["Team store sale: {sale}.", "Members only: {sale}, see you at the game!"],
    "personal": ["Hey, {friend_msg}?", "Yo {friend_msg}."],
    "implied_medical": ["{person} near {place} {implied}.", "Can someone come to {place}? {family} {implied}.",
                        "Not sure what's wrong but {person} in {place} {implied}."],
    "threat2": ["A guy at {place} {weapon_threat}.", "Someone near {place} {weapon_threat}, please send someone."],
    "hedged": ["{hedge}."],
    "neutral": ["The {lot} closes at {hour}.", "{place} is fine.", "Staff just mopped the {spot} at {place}, all dry now.",
                "All clear at {place}."],
}
TONE = [("", ""), ("Hey, ", ""), ("", " Please help."), ("", " Thanks!"), ("URGENT: ", ""), ("", " This is ridiculous."),
        ("Hi, ", " Thank you."), ("", "!!")]


def as_text(m):
    return m if isinstance(m, str) else json.dumps(m, ensure_ascii=False, sort_keys=True)


def wrap_structured(msg, rng):
    """Some messages arrive as JSON from apps and order systems, not plain text."""
    if rng.random() < 0.5:
        return {"source": rng.choice(["app", "web", "kiosk", "email"]), "tier": rng.choice(["free", "member", "vip"]),
                "text": msg, "tries": rng.randint(1, 4)}
    return {"order_items": rng.sample(["hot dog", "soda", "nachos", "pretzel", "beer", "water"], 2),
            "state": rng.choice(["delivered", "pending", "refunded", "cancelled"]),
            "comment": msg, "stars": rng.randint(1, 5)}


def synth_messages(n, seed=0, exclude=()):
    rng = random.Random(seed)
    exclude = {as_text(e) for e in exclude}
    out, seen, tries = [], set(), 0
    cats = list(TEMPLATES)
    while len(out) < n and tries < n * 50:
        tries += 1
        t = rng.choice(TEMPLATES[rng.choice(cats)])
        fill = {k: rng.choice(v) for k, v in SLOTS.items() if k != "hedge"}
        fill["hedge"] = rng.choice(SLOTS["hedge"]).format(**fill)
        msg = t.format(**fill)
        msg = msg[0].upper() + msg[1:]
        pre, post = rng.choice(TONE)
        msg = pre + (msg[0].lower() + msg[1:] if pre else msg)
        if post == "!!":
            msg = msg.rstrip(".?") + "!!"
        else:
            msg += post
        if rng.random() < 0.12:
            msg = wrap_structured(msg, rng)
        key = as_text(msg)
        if key not in exclude and key not in seen:
            seen.add(key)
            out.append(msg)
    return out


def read_messages(path):
    msgs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                r = json.loads(line)
                msgs.append(r.get("state", r.get("text", "")))
            else:
                msgs.append(line)
    return [m for m in msgs if m]


# ---------------------------------------------------------------- the student

class Embedder:
    def __init__(self, name, device):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device, self.name = torch, device, name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()

    def __call__(self, texts, batch=64):
        torch = self.torch
        outs = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=256,
                           return_tensors="pt").to(self.device)
            with torch.inference_mode():
                h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pooling
            outs.append(torch.nn.functional.normalize(e, dim=-1).float())
        return torch.cat(outs) if outs else torch.empty(0)


class StudentBackend:
    """Plugs into localjev like any other model. Only answers questions it was trained on."""
    frames = ["{}"]
    fixed_questions = True  # rewording a question means nothing to it, so stasis skips those variants

    def __init__(self, folder, device):
        import localjev as L
        self.L = L
        with open(os.path.join(folder, "student.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.embed = Embedder(data["embedder"], device)
        self.heads = data["heads"]
        base = os.path.basename(os.path.normpath(folder))
        self.name = "student" if base == "student" else f"student-{base}"
        self.model_name = f"student of {data['teacher']} ({base})"
        torch = self.embed.torch
        self.W = {s: torch.tensor(h["W"], device=device) for s, h in self.heads.items()}
        self.b = {s: torch.tensor(h["b"], device=device) for s, h in self.heads.items()}

    def distributions(self, states, items):
        torch = self.embed.torch
        need = sorted({si for si, _, _ in items})
        E = self.embed([self.L.state_text(states[si]) for si in need])
        row = {si: i for i, si in enumerate(need)}
        out = []
        for si, q, _ in items:
            s = qsig(clean(q))
            if s not in self.heads:
                raise ValueError(f"student was not trained on question: {q['instructions']!r} "
                                 f"(add it and run distill again)")
            p = torch.softmax(self.W[s] @ E[row[si]] + self.b[s], dim=0).tolist()
            out.append(dict(zip(self.heads[s]["keys"], p)))
        return out


def train_heads(E, targets, steps=500, lr=0.05, wd=1e-4):
    """E: [N, d] embeddings. targets: [N, K] teacher probabilities. Returns W [K, d], b [K]."""
    import torch
    d, K = E.shape[1], targets.shape[1]
    W = torch.zeros(K, d, device=E.device, requires_grad=True)
    b = torch.zeros(K, device=E.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logp = torch.log_softmax(E @ W.T + b, dim=1)
        loss = -(targets * logp).sum(1).mean() + wd * (W ** 2).sum()  # soft cross-entropy
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


# ---------------------------------------------------------------- the command

def cmd_distill(engine, args):
    import torch
    import localjev as L

    questions = collect_questions(args.paths, args.questions)
    if not questions:
        sys.exit("no questions found (pass test folders or --questions files)")
    log(f"{len(questions)} distinct questions to learn")

    # Never train on the test messages, so grading the student stays honest.
    test_states = {L.state_text(req["state"]) for _, req, _ in L.load_cases(args.paths)} if args.paths else set()
    msgs = []
    for path in args.messages or []:
        msgs += read_messages(path)
    msgs += synth_messages(max(0, args.n - len(msgs)), seed=args.seed, exclude=test_states | {as_text(m) for m in msgs})
    msgs = [m for m in msgs if as_text(m) not in test_states]
    log(f"{len(msgs)} training messages ({len(test_states)} test messages kept out)")

    # 1. teacher answers everything (answers are cached, so re-running only asks what's new)
    teacher_name = " + ".join(b.model_name for b in engine.backends) + (" + stasis" if engine.stasis else "")
    qmap = {f"q{i}": q for i, q in enumerate(questions)}
    os.makedirs(args.out, exist_ok=True)
    cache_path = os.path.join(args.out, "teacher_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("teacher") == teacher_name:
                    cache[(r["m"], r["q"])] = r["d"]
    teacher = {k: [] for k in qmap}
    asked, t0 = 0, time.time()
    with open(cache_path, "a", encoding="utf-8") as cf:
        for i, m in enumerate(msgs, 1):
            mt = as_text(m)
            missing = {k: q for k, q in qmap.items() if (mt, qsig(q)) not in cache}
            if missing:
                asked += 1
                raw = engine.solve({"state": m, "questions": missing})
                for k, q in missing.items():
                    a = engine.summarize(q, raw[k])
                    keys = L.outcome_keys(q)
                    d = {"true": a["noul"], "false": 1 - a["noul"]} if q["type"] == "noul" else \
                        {kk: float(a["probabilities"][kk]) for kk in keys}
                    cache[(mt, qsig(q))] = [d[kk] for kk in keys]
                    cf.write(json.dumps({"teacher": teacher_name, "m": mt, "q": qsig(q), "d": cache[(mt, qsig(q))]}) + "\n")
            for k, q in qmap.items():
                teacher[k].append(cache[(mt, qsig(q))])
            if i % 50 == 0 or i == len(msgs):
                el = time.time() - t0
                log(f"  teacher: {i}/{len(msgs)}  ({el:.0f}s, ~{el / i * (len(msgs) - i):.0f}s left, "
                    f"{i - asked} answered from cache)")
    teacher_sec = (time.time() - t0) / max(asked, 1)

    # 2. free the teacher's GPU memory, load the small embedder
    engine.backends.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    log(f"Loading embedder {args.embedder}...")
    emb = Embedder(args.embedder, device)
    E = emb([as_text(m) for m in msgs])

    # 3. hold out 20% to check how well the student copies the teacher
    idx = list(range(len(msgs)))
    random.Random(args.seed).shuffle(idx)
    cut = int(len(idx) * 0.8)
    tr, ho = idx[:cut], idx[cut:]
    heads, report = {}, []
    for k, q in qmap.items():
        T = torch.tensor(teacher[k], device=device)
        W, b = train_heads(E[tr], T[tr], steps=args.steps)
        P = torch.softmax(E[ho] @ W.T + b, dim=1)
        agree = (P.argmax(1) == T[ho].argmax(1)).float().mean().item()
        gap = (P - T[ho]).abs().max(1).values.mean().item()
        # final head trained on everything
        W, b = train_heads(E, T, steps=args.steps)
        heads[qsig(q)] = {"question": q, "keys": L.outcome_keys(q),
                          "W": [[round(x, 6) for x in r] for r in W.cpu().tolist()],
                          "b": [round(x, 6) for x in b.cpu().tolist()]}
        top = T.argmax(1)
        counts = torch.bincount(top, minlength=T.shape[1]).float() / len(top)
        rare = counts.min().item()
        mix = " / ".join(f"{c:.0%}" for c in counts.tolist()[:4])
        report.append((q["instructions"][:44], agree, gap, mix, rare))

    with open(os.path.join(args.out, "student.json"), "w", encoding="utf-8") as f:
        json.dump({"embedder": args.embedder, "teacher": teacher_name, "trained_on": len(msgs),
                   "dim": E.shape[1], "heads": heads}, f)

    # 4. speed of the student
    sample = [as_text(m) for m in msgs[:200]]
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    Es = emb(sample)
    for h in heads.values():
        torch.softmax(Es @ torch.tensor(h["W"], device=device).T + torch.tensor(h["b"], device=device), dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    student_sec = (time.time() - t1) / len(sample)

    n_numbers = sum(len(h["W"]) * (len(h["W"][0]) + 1) for h in heads.values())
    print(f"\nteacher: {teacher_name}")
    print(f"student: {args.embedder} + {len(heads)} formulas ({n_numbers:,} learned numbers), saved to {args.out}/student.json\n")
    print(f"{'question':<46}{'agrees with teacher':>20}{'avg prob gap':>14}{'teacher said yes/top':>22}")
    for name, agree, gap, mix, _ in report:
        print(f"{name:<46}{agree:>19.0%}{gap:>14.2f}{mix:>22}")
    print(f"\n{'average':<46}{sum(r[1] for r in report) / len(report):>19.0%}"
          f"{sum(r[2] for r in report) / len(report):>14.2f}")
    lonely = [r[0] for r in report if r[4] < 0.03]
    if lonely:
        print("\nwarning: the teacher almost never said 'yes' (or picked the rare option) for:")
        for name in lonely:
            print(f"  - {name}")
        print("  a high 'agrees' score there can be fake (always saying no). Add example messages for it with --messages.")
    print(f"\nspeed per message (all {len(heads)} questions): teacher {teacher_sec * 1000:.0f} ms, "
          f"student {student_sec * 1000:.1f} ms  ->  {teacher_sec / max(student_sec, 1e-9):.0f}x faster")
    print("(held-out check: 20% of messages the student never trained on)")
