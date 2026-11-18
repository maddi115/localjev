"""learnfacts: read your chat history and file what it learns into a cabinet, one drawer (file) per topic.

  twitch/facts/
  ├── games.json        one drawer per topic: its question, what YOU typed, and the facts it learned
  ├── food.json
  ├── ...
  ├── _unsure.json      facts the decision model couldn't file the same way twice (never used)
  ├── _blocked.txt      facts you told it to forget (never learned again)
  └── _private/         health, money, body, family... locked: the bot never opens this folder

How it learns:
  1. find    - messages that say something about you             (regex, then NLI on the GPU)
  2. pull    - Qwen writes each as a short fact: "i play valorant"  (batched on the GPU)
  3. verify  - NLI: does your original message really back it?   (no invented facts)
  4. lock    - private facts go to _private/ (NLI + word list, checked on the fact AND the message)
  5. file    - the decision model picks a drawer TWICE: from the fact, and from your message.
               Both agree -> filed. They disagree -> _unsure.json.   (probability stasis for filing)
  6. ask     - facts that fit no drawer: k-means groups them, Qwen makes a new drawer + question
  7. stasis  - inside a drawer, repeats merge; a fact stays only if said often, on different days
"""
import argparse, glob, json, math, os, re, shutil, sys, time
from collections import defaultdict
from datetime import date

import localjev as L
from twitch import CABINET, EMBEDDER, FACTS, OUT, WRITER, load_msgs

PRIVATE_DIR = os.path.join(CABINET, "_private")
UNSURE = os.path.join(CABINET, "_unsure.json")
BLOCKED = os.path.join(CABINET, "_blocked.txt")
OLD_DRAFT = os.path.join(OUT, "facts_draft.json")
ME = re.compile(r"\b(i|im|i'm|ive|i've|ill|i'll|id|i'd|my|me|mine|myself)\b", re.I)
SELF_FACT = "The speaker tells something lasting about themselves, like what they play, like, own, do, or where they live."
PULL_SYSTEM = (
    "You pull ONE lasting fact about the speaker out of a Twitch chat message. "
    "Lasting means: what they play, like, own, use, do for work or fun, or where they live. "
    "Not feelings right now, not opinions about the stream, not what is happening on screen, "
    "not opinions about themselves (like 'i am ugly' or 'i am a good person'). "
    "Write it in first person, lowercase, 2 to 8 words, like: i play valorant / i live in la / i use a logitech mouse. "
    "If there is no lasting fact, write NONE. Write only the fact.")
NAME_SYSTEM = (
    "These are facts one person said about themselves. Name the topic they share. "
    "Answer in exactly this format, one line:\n"
    "key | about | question\n"
    "key = one lowercase word. about = what the topic covers, starting with 'what' or 'where' or 'whether'. "
    "question = what a stranger in chat would ask to get these facts.\n"
    "Example: games | what games you play or like | what games do you play?")

# Drawers: key -> (about, question, NLI hypothesis the decision model checks)
TOPICS = {
    "games": ("what video games you play or like", "what games do you play?", "This is about video games they play."),
    "music": ("what music you listen to", "what music do you listen to?", "This is about music they listen to or bands."),
    "food": ("what you eat and drink", "what do you like to eat?", "This is about what they eat or drink."),
    "work": ("what work or jobs you do", "what do you do for work?",
             "This is about their job, their work, or a business they own."),
    "home": ("where you live or are from", "where are you from?", "This is about where they live or come from."),
    "hobbies": ("what you do for fun", "what do you do for fun?",
                "This is about a hobby or activity like hiking, cycling, sports or photography."),
    "setup": ("your PC, mouse, keyboard, headset or gear", "what setup do you have?",
              "This is about their computer, mouse, keyboard, headset or other gear."),
    "streaming": ("streaming, twitch, youtube or making videos", "do you stream?",
                  "This is about streaming, twitch, youtube, tiktok or making videos."),
    "watching": ("what movies and shows you watch", "what's your favorite movie?",
                 "This is about movies or tv shows they watch."),
    "projects": ("things you are building or making", "what are you working on?",
                 "This is about something they are building or making, like code, apps, maps or art."),
}
OTHER = "This is about something else."
ALIASES = {"from": "home", "location": "home", "where": "home", "job": "work", "jobs": "work",
           "movies": "watching", "shows": "watching", "gear": "setup", "pc": "setup"}
VAGUE = {"topics", "topic", "other", "misc", "general", "interests", "stuff", "things", "life", "random"}
PRIVATE = {
    "health": "This is about health, the body, weight, diet, medicine or sleep.",
    "money": "This is about money, benefits, debt or what someone can afford.",
    "intimate": "This is about sex or private body parts.",
    "personal": "This is about family, relationships, religion, how they feel about themselves, or the law.",
}
PRIVATE_WORDS = re.compile(
    r"\b(pounds?|lbs|weigh\w*|diet\w*|fat|apnea|ozempic\w*|wegovy|meds?|medication|pills?|doctor|hospital|"
    r"sick|injur\w*|pain|therapy|therapist|depress\w*|anxiety|adhd|ebt|food stamps|welfare|broke|debt|rent|"
    r"rich|poor|afford\w*|penis|dick|sex\w*|pregnan\w*|girlfriend|boyfriend|wife|husband|divorc\w*|jail|"
    r"arrest\w*|cops?|police|court|pray\w*|god|church|relig\w*|ugly|health\w*)\b", re.I)
FILLER = {"i", "im", "am", "a", "an", "the", "my", "me", "have", "has", "use", "used", "to", "be", "is", "do", "of"}


from colors import BLUES, bar, dim, dither, reset, rgb  # gradient blue dither


# ---------------------------------------------------------------- models
class Models:
    def __init__(self, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from distill import Embedder
        self.torch, self.device = torch, device
        self.nli = L.NLIBackend(L.NLI_MODEL, device)
        self.emb = Embedder(EMBEDDER, device)
        print(f"Loading writer {WRITER}...", file=sys.stderr)
        self.tok = AutoTokenizer.from_pretrained(WRITER, padding_side="left")
        self.llm = AutoModelForCausalLM.from_pretrained(
            WRITER, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

    def entail(self, pairs):
        """P(premise entails hypothesis) for many (premise, hypothesis) pairs, batched."""
        if not pairs:
            return []
        logits = self.nli._logits(pairs, batch=64)
        return self.torch.softmax(logits, dim=1)[:, self.nli.ent].tolist()

    def generate(self, system, users, max_new=16, batch=48):
        torch, outs = self.torch, []
        for i in range(0, len(users), batch):
            texts = [self.tok.apply_chat_template([{"role": "system", "content": system},
                                                   {"role": "user", "content": u}],
                                                  tokenize=False, add_generation_prompt=True)
                     for u in users[i:i + batch]]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                out = self.llm.generate(**enc, do_sample=False, max_new_tokens=max_new,
                                        pad_token_id=self.tok.eos_token_id)
            for o in out:
                outs.append(self.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                            .strip().split("\n")[0].strip().strip('"').strip())
        return outs


def kmeans(X, k, iters=40, seed=0):
    """Cosine k-means on the GPU (X rows are unit length)."""
    import torch
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        C = torch.zeros_like(C).index_add_(0, lab, X)
        C = torch.nn.functional.normalize(C + 1e-9, dim=1)
    return (X @ C.T).argmax(1).tolist()


def day(t):
    try:
        return date.fromisoformat(t[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------- the cabinet on disk
def read_cabinet():
    """{drawer: data} for every public drawer file."""
    out = {}
    for path in sorted(glob.glob(os.path.join(CABINET, "*.json"))):
        name = os.path.basename(path)[:-5]
        if name.startswith("_"):
            continue
        with open(path, encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def typed_by_you():
    """What you typed yourself: from the existing drawers, or (first run) from the old facts.json."""
    typed = {k: v.get("typed_by_you", "") for k, v in read_cabinet().items() if v.get("typed_by_you")}
    if typed or not os.path.exists(FACTS):
        return typed
    with open(FACTS, encoding="utf-8") as f:
        old = json.load(f)
    auto = {}
    if os.path.exists(OLD_DRAFT):  # answers --keep copied from the old draft were learned, not typed
        with open(OLD_DRAFT, encoding="utf-8") as f:
            auto = {v.get("answer", "") for v in json.load(f).values()}
    for key, v in old.items():
        ans = v.get("answer", "")
        if not ans or "EDIT ME" in ans or ans in auto:
            continue
        drawer = ALIASES.get(key, key)
        typed[drawer] = "; ".join(x for x in (typed.get(drawer), ans) if x)
    return typed


def blocked():
    if not os.path.exists(BLOCKED):
        return set()
    with open(BLOCKED, encoding="utf-8") as f:
        return {l.strip().lower() for l in f if l.strip()}


def write_cabinet(drawers, unsure, private, typed):
    os.makedirs(PRIVATE_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(CABINET, "*.json")) + glob.glob(os.path.join(PRIVATE_DIR, "*.json")):
        os.remove(path)  # rebuilt from scratch each time (typed facts and _blocked.txt are kept)
    for key in set(drawers) | set(typed):
        d = drawers.get(key, {})
        about, question = TOPICS.get(key, (d.get("about", key), d.get("question", "")))[:2]
        mine = typed.get(key, "").lower()
        learned = [r["fact"] for r in d.get("facts", []) if not r["old"] and r["fact"] not in mine][:4]
        answer = "; ".join(x for x in [typed.get(key, "")] + learned if x)
        with open(os.path.join(CABINET, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump({"drawer": key, "question": question, "about": about, "typed_by_you": typed.get(key, ""),
                       "answer": answer, "facts": d.get("facts", [])}, f, indent=2, ensure_ascii=False)
    with open(UNSURE, "w", encoding="utf-8") as f:
        json.dump(unsure, f, indent=2, ensure_ascii=False)
    for kind, rows in private.items():
        with open(os.path.join(PRIVATE_DIR, f"{kind}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- the pipeline
def learn(args):
    t0 = time.time()
    step = lambda msg: print(f"{dim(f'[{time.time() - t0:5.1f}s]')} {msg}", file=sys.stderr)
    msgs = load_msgs()
    typed = typed_by_you()
    skip = blocked()
    M = Models(L.pick_device(args.device))

    # 1. find
    cand = [m for m in msgs if ME.search(m["text"]) and len(m["text"].split()) >= 3]
    p_self = M.entail([(m["text"], SELF_FACT) for m in cand])
    cand = [m for m, p in zip(cand, p_self) if p >= args.self_threshold]
    step(f"1. find   : {len(cand)} of {len(msgs)} messages say something about you")

    # 2. pull
    facts = M.generate(PULL_SYSTEM, [m["text"] for m in cand])
    pulled = [(m, f.lower().rstrip(".!")) for m, f in zip(cand, facts)
              if f and "none" not in f.lower() and 2 <= len(f.split()) <= 10]
    pulled = [(m, f) for m, f in pulled if f not in skip and not re.fullmatch(r"i am (a |an )?\w+", f)]
    step(f"2. pull   : Qwen wrote {len(pulled)} facts")

    # 3. verify
    p_ok = M.entail([(m["text"], f) for m, f in pulled])
    good = [(m, f, p) for (m, f), p in zip(pulled, p_ok) if p >= args.verify]
    step(f"3. verify : {len(good)} backed by your own message, {len(pulled) - len(good)} thrown out")
    if not good:
        sys.exit("no facts found")

    # 4. lock private ones (fact AND message, each private kind on its own, low bar, plus a word list)
    nP = len(PRIVATE)
    pp = M.entail([(t, h) for m, f, _ in good for t in (f, m["text"]) for h in PRIVATE.values()])
    lock = []
    for i, (m, f, _) in enumerate(good):
        ps = pp[i * 2 * nP:(i + 1) * 2 * nP]
        best = max(range(2 * nP), key=lambda j: ps[j])
        kind = list(PRIVATE)[best % nP] if ps[best] >= args.private_bar else None
        if not kind and (PRIVATE_WORDS.search(f) or PRIVATE_WORDS.search(m["text"])):
            kind = "flagged words"
        lock.append(kind)
    step(f"4. lock   : {sum(1 for k in lock if k)} private facts locked in _private/")

    # 5. file twice: from the fact, and from your original message. Agree -> filed. Disagree -> _unsure.
    keys = list(TOPICS)
    hyps = [TOPICS[k][2] for k in keys] + [OTHER]
    H = len(hyps)
    pub = [i for i in range(len(good)) if not lock[i]]
    pairs = [(t, h) for i in pub for t in (good[i][1], good[i][0]["text"]) for h in hyps]
    lg = M.nli._logits(pairs, batch=64)[:, M.nli.ent].view(len(pub), 2, H)
    pr = M.torch.softmax(lg, dim=2)
    place = {}  # i -> drawer | "_unsure" | None (fits no drawer)
    for row, i in enumerate(pub):
        mean = (pr[row, 0] + pr[row, 1]) / 2
        win = mean.argmax().item()
        agree = (pr[row, 0].argmax().item() == win) + (pr[row, 1].argmax().item() == win)
        conf = mean[win].item() * agree / 2  # probability stasis: average reading x how much they agree
        if win == H - 1:
            place[i] = None
        elif conf >= args.topic:
            place[i] = keys[win]
        else:
            place[i] = "_unsure"
    filed = sum(1 for v in place.values() if v and v != "_unsure")
    step(f"5. file   : {filed} filed (two readings, stasis >= {args.topic}), "
         f"{sum(1 for v in place.values() if v == '_unsure')} unsure, "
         f"{sum(1 for v in place.values() if v is None)} fit no drawer")

    # 6. ask: new drawers from facts that fit nowhere
    E = M.emb([" ".join(w for w in f.split() if w not in FILLER) or f for _, f, _ in good])
    loose = [i for i, v in place.items() if v is None]
    new = {}
    if len(loose) >= 12:
        lab = kmeans(E[loose], max(2, min(args.k, len(loose) // 12)))
        groups = defaultdict(list)
        for i, c in zip(loose, lab):
            groups[c].append(i)
        groups = [v for v in groups.values() if len(v) >= 6]
        names = M.generate(NAME_SYSTEM, ["\n".join(f"- {good[i][1]}" for i in v[:10]) for v in groups],
                           max_new=40, batch=16)
        for v, name in zip(groups, names):
            parts = [x.strip() for x in name.split("|")]
            key = ALIASES.get(re.sub(r"[^a-z]", "", (parts[0] if parts else "").lower()),
                              re.sub(r"[^a-z]", "", (parts[0] if parts else "").lower()))
            if not key or key in VAGUE:
                continue
            if key in PRIVATE or PRIVATE_WORDS.search(key):
                for i in v:
                    lock[i] = "health" if key == "health" else "flagged words"
                    place.pop(i, None)
                continue
            if key not in TOPICS:
                about = parts[1] if len(parts) > 1 and parts[1] else key
                q = parts[2] if len(parts) > 2 and parts[2] else (about if about.endswith("?") else "")
                new.setdefault(key, (about, q))
            for i in v:
                place[i] = key
    step(f"6. ask    : {len(new)} new drawers made" + (f" ({', '.join(new)})" if new else ""))

    # 7. stasis inside each drawer
    all_days = [d for d in (day(m["time"]) for m in msgs) if d]
    end = max(all_days) if all_days else None

    def stasis_rows(idx):
        made = []
        for i in sorted(idx, key=lambda i: -good[i][2]):
            for lead, members in made:
                if float(E[i] @ E[lead]) >= args.same:
                    members.append(i); break
            else:
                made.append((i, [i]))
        rows = []
        for lead, members in made:
            days = sorted({d for d in (day(good[i][0]["time"]) for i in members) if d})
            n, nd = len(members), len(days)
            p = sum(good[i][2] for i in members) / n
            rows.append({"fact": good[lead][1], "mentions": n, "days": nd,
                         "first": str(days[0]) if days else "", "last": str(days[-1]) if days else "",
                         "stasis": round(p * (1 - math.exp(-n / 3)) * (1 - math.exp(-nd / 2)), 3),
                         "old": bool(end and days and (end - days[-1]).days > args.old_days),
                         "examples": [good[i][0]["text"] for i in members[:2]]})
        rows.sort(key=lambda r: -r["stasis"])
        return [r for r in rows if r["mentions"] >= args.min_mentions and r["days"] >= 2]

    by = defaultdict(list)
    for i, v in place.items():
        if v:
            by[v].append(i)
    drawers = {}
    for key, idx in by.items():
        if key == "_unsure":
            continue
        rows = stasis_rows(idx)
        if rows:
            about, q = TOPICS[key][:2] if key in TOPICS else new[key]
            drawers[key] = {"about": about, "question": q, "facts": rows}
    unsure = stasis_rows(by.get("_unsure", []))
    private = defaultdict(list)
    plock = defaultdict(list)
    for i, k in enumerate(lock):
        if k:
            plock[k].append(i)
    for k, idx in plock.items():
        rows = stasis_rows(idx)
        if rows:
            private[k] = rows
    write_cabinet(drawers, unsure, private, typed)
    step(f"7. stasis : {sum(len(d['facts']) for d in drawers.values())} stable facts in {len(drawers)} drawers")
    show()
    print(dim(f"Forget a fact for good:  python learnfacts.py --forget \"i own a liquor store\""))
    print(dim(f"Fix what you typed:      nano {CABINET}/games.json   (edit \"typed_by_you\")"))


# ---------------------------------------------------------------- showing the cabinet
def show(private=False):
    drawers = read_cabinet()
    if not drawers:
        sys.exit("cabinet is empty: python learnfacts.py")
    order = sorted(drawers, key=lambda k: -max([r["stasis"] for r in drawers[k]["facts"]] + [1 if drawers[k].get("typed_by_you") else 0]))
    print(f"\n{CABINET}/")
    for row, key in enumerate(order):
        d = drawers[key]
        print(f"├── {dither(' ' + key + '.json ', row, 0.0, 0.4)}  {d['question'] or d['about']}")
        if d.get("typed_by_you"):
            print(f"│     {rgb(BLUES[4])}✎ you typed{reset()}  {d['typed_by_you']}")
        for j, r in enumerate(d["facts"]):
            meta = dim(f"{r['mentions']}x · {r['days']} days · {r['first'][:7]} → {r['last'][:7]}")
            if r["old"]:
                print(f"│     {bar(r['stasis'])} {r['stasis']:.2f}  {dim('used to: ' + r['fact'])}  {meta}")
            else:
                print(f"│     {bar(r['stasis'])} {r['stasis']:.2f}  {dither(' ' + r['fact'] + ' ', row + j)}  {meta}")
        print("│")
    unsure = json.load(open(UNSURE, encoding="utf-8")) if os.path.exists(UNSURE) else []
    print(f"├── {dim('_unsure.json')}   {dim(f'{len(unsure)} facts the two readings filed differently (not used)')}")
    for r in unsure[:5]:
        print(f"│     {dim('? ' + r['fact'])}")
    n_block = len(blocked())
    print(f"├── {dim('_blocked.txt')}   {dim(f'{n_block} facts you told it to forget')}")
    kinds = {os.path.basename(p)[:-5]: json.load(open(p, encoding="utf-8"))
             for p in sorted(glob.glob(os.path.join(PRIVATE_DIR, "*.json")))}
    total = sum(len(v) for v in kinds.values())
    print(f"└── {dim('_private/')}      {dim(f'🔒 {total} facts locked, the bot never opens this folder')}  "
          f"{dim('(' + ', '.join(f'{k} {len(v)}' for k, v in kinds.items()) + ')') if kinds else ''}")
    if private:
        for k, rows in kinds.items():
            for r in rows:
                print(f"      {dim('🔒 ' + k + ': ' + r['fact'])}")
    elif total:
        print(dim("      see them: python learnfacts.py --show --private"))
    print()


def forget(text):
    text = text.strip().lower()
    os.makedirs(CABINET, exist_ok=True)
    with open(BLOCKED, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    hit = 0
    for key, d in read_cabinet().items():
        keep_rows = [r for r in d["facts"] if r["fact"] != text]
        if len(keep_rows) != len(d["facts"]):
            hit += 1
            d["facts"] = keep_rows
            learned = [r["fact"] for r in keep_rows if not r["old"]][:4]
            d["answer"] = "; ".join(x for x in [d.get("typed_by_you", "")] + learned if x)
            path = os.path.join(CABINET, f"{key}.json")
            if not d["answer"] and not keep_rows:
                os.remove(path)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"Forgot \"{text}\" ({'removed from ' + str(hit) + ' drawer' if hit else 'not in any drawer'}). "
          f"It won't be learned again.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="show the cabinet (no models, instant)")
    ap.add_argument("--private", action="store_true", help="with --show: also list the locked private facts")
    ap.add_argument("--forget", metavar="FACT", help="remove a fact and never learn it again")
    ap.add_argument("--k", type=int, default=6, help="most new drawers to make")
    ap.add_argument("--topic", type=float, default=0.25, help="stasis bar to file a fact in a drawer")
    ap.add_argument("--private-bar", dest="private_bar", type=float, default=0.3, help="NLI bar for private")
    ap.add_argument("--self-threshold", type=float, default=0.02)
    ap.add_argument("--verify", type=float, default=0.5, help="NLI bar for 'the message backs the fact'")
    ap.add_argument("--same", type=float, default=0.82, help="how similar two facts must be to merge")
    ap.add_argument("--min-mentions", type=int, default=2, help="stasis: said at least this many times")
    ap.add_argument("--old-days", type=int, default=180, help="not said for this long = 'used to'")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.forget:
        forget(args.forget)
    elif args.show:
        show(private=args.private)
    else:
        learn(args)


if __name__ == "__main__":
    main()
