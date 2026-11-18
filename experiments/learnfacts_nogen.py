"""learnfacts: read your chat history and file it into a cabinet - one drawer (JSON file) per topic.
NO generation anywhere: every fact is cut straight out of a message you typed. The JSON structure itself is
just Python writing files; the models only make yes/no and "how similar" decisions.

  twitch/facts/
  ├── games.json         a drawer: question, what YOU typed, every fact pulled from your chat (+ the messages)
  ├── ...
  ├── _unsure.json       groups the two readings filed differently (kept, never said)
  ├── _other.json        groups that fit no drawer yet (kept, never said)
  ├── _blocked.txt       facts you told it to forget
  ├── _decisions.json    your --review answers (kept every time you re-learn)
  └── _private/          DEFAULT-DENY: anything not clearly safe. The bot never opens this folder.

How it learns:
  1. find    messages about you ("i / my / me"), not questions
  2. pull    cut each message into pieces about you, minus filler words. Your words only.
  3. check   NLI: does your full message back the piece? (drops obvious junk)
  4. lock    private unless clearly safe: 6 private kinds (NLI, on piece AND message) + a sensitive-word list
             + a positive "ordinary harmless fact" check. Anything that fails goes to _private/
  5. group   embeddings group pieces that MEAN the same ("i play valorant" = "im playing valorant")
  6. file    each group gets ONE drawer: NLI reading + embedding reading. stasis = average x agreement
  7. stasis  a group said 3+ times on 2+ days, whose drawer both readings agreed on, may be said by the bot.
             Automatic. --review is optional if you ever want to veto one.
"""
import argparse, glob, json, math, os, re, sys, time
from collections import Counter, defaultdict
from datetime import date

import localjev as L
from colors import BLUES, NEON, bar, dim, dither, reset, rgb  # gradient dithers
from twitch import CABINET, EMBEDDER, FACTS, OUT, STOP, WORD, load_msgs

PRIVATE_DIR = os.path.join(CABINET, "_private")
UNSURE = os.path.join(CABINET, "_unsure.json")
OTHER_FILE = os.path.join(CABINET, "_other.json")
BLOCKED = os.path.join(CABINET, "_blocked.txt")
OLD_DRAFT = os.path.join(OUT, "facts_draft.json")
ME = re.compile(r"\b(i|im|i'm|ima|imma|ive|i've|ill|i'll|id|i'd|my|me|mine|myself)\b", re.I)
SELF_FACT = "The speaker tells something lasting about themselves, like what they play, like, own, do, or where they live."

# ---- direct pulling (no generation)
SPLIT = re.compile(r"[.!?;,:()\[\]]+|\s(?:and|but|cuz|cause|bc|because|so|while|then|tho|though|or|plus|when|if)\s"
                   r"|\s(?=(?:i|im|i'm|ima|imma|ive|i've)\s)")  # a new "i ..." mid-sentence starts a new piece
FIRST = re.compile(r"^(i|im|i'm|ima|imma|ive|i've|ill|i'll|id|i'd|my)\b|^\w+ my\b")  # "got my third job"
TRAIL_PHRASE = re.compile(r"\s+(right now|right meow|rn|today|tonight|yesterday|lately|again|"
                          r"yes i am|i am|for real|or something|and shit|n shit|and stuff|or whatever|"
                          r"anyways?|lo+l|lma+o+|ha(ha)+|x+d+)$")
LEAD_FILLER = {"dude", "bro", "lol", "lmao", "lmfao", "yeah", "yea", "ya", "yes", "nah", "no", "omg", "also", "btw", "tbh",
               "like", "ok", "okay", "well", "so", "and", "but", "oh", "yo", "bruh", "man", "ngl", "honestly", "fr",
               "anyways", "anyway", "actually", "literally", "basically", "and", "plus", "cuz", "bc", "um", "uh", "hmm"}
TRAIL_FILLER = {"dude", "bro", "lol", "lmao", "lmfao", "btw", "tbh", "man", "bruh", "fr", "ngl", "xd", "xdd", "too",
                "tho", "yes", "haha", "lmaoo", "rn", "atm", "lowkey", "fam", "frfr", "ong", "though", "and", "but",
                "anyways", "anyway", "now", "yet", "ok", "okay", "lmfaoo", "xddd", "meow", "right"}
FLEETING = re.compile(
    r"^(i|im|i'm|i am)\s+(feel|felt|think|thought|guess|hope|wish|mean|meant|know|dont know|don't know|bet|swear|"
    r"forgot|said|told|asked|wonder|need to|have to|gotta|gonna|wanna|cant believe|can't believe|agree|disagree|"
    r"see|saw that|heard that|love that|like that|hate that|miss|aint|ain't|dont care|don't care)\b"
    r"|^(im|i'm|i am)\s+(so |really |kinda |pretty |literally |just |still )?(tired|bored|hungry|sleepy|dying|dead|done|"
    r"confused|sad|mad|happy|excited|scared|sick|high|drunk|back|here|ready|good|fine|ok|okay|lost|late|crying|"
    r"weak|screaming|laughing|joking|kidding|serious|sorry|sure|not sure|watching|listening|leaving|going|"
    r"not lying|not joking|not kidding|lying|telling the truth|a kid|just saying|just kidding|curious|jealous)\b"
    r"|^i (was a|said|meant|forgot|remember)\b")
TALKS_TO = re.compile(r"\b(u|you|ur|your|youre|you're|yall|y'all|he|she|him|her|his|they|them|their|chat|streamer)\b")
FILLER = {"i", "im", "am", "a", "an", "the", "my", "me", "have", "has", "use", "used", "to", "be", "is", "do", "of"}


def addressees(msgs):
    """Names you talk TO (e.g. a friend in chat): words you often put at both the start AND the end of messages.
    They get trimmed off pieces so other people's names don't end up in your facts."""
    first, last = Counter(), Counter()
    for m in msgs:
        w = WORD.findall(m["text"].lower())
        if len(w) >= 3:
            first[w[0]] += 1; last[w[-1]] += 1
    skip = STOP | FILLER | LEAD_FILLER | TRAIL_FILLER
    return {w for w, c in last.items() if c >= 8 and first[w] >= 3 and len(w) >= 3 and w not in skip}


def pieces(text, names=()):
    """Cut one message into the pieces that are about you. Only removes words, never adds or changes one."""
    t = re.sub(r"@\w+", " ", text.lower())
    out = []
    for part in SPLIT.split(t):
        words = part.split()
        while words and words[0].strip("'") in LEAD_FILLER:
            words.pop(0)
        while words and (words[-1].strip("'") in TRAIL_FILLER or words[-1] in names):
            words.pop()
        piece = " ".join(words).strip(" '\"-*~")
        for _ in range(3):  # trim "... right meow sam" -> "..."
            piece = TRAIL_PHRASE.sub("", piece)
            words = piece.split()
            while words and (words[-1].strip("'") in TRAIL_FILLER or words[-1] in names):
                words.pop()
            piece = " ".join(words)
        words = piece.split()
        if not FIRST.match(piece) or FLEETING.match(piece) or TALKS_TO.search(piece):
            continue
        if not 2 <= len(words) <= 10:
            continue
        if not [w for w in WORD.findall(piece) if w not in STOP and w not in FILLER and len(w) >= 3]:
            continue
        out.append(piece)
    return out


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
# Privacy is DEFAULT-DENY: a fact goes public only if no private kind fires, no sensitive word appears,
# and it passes a positive "safe to say in public" check. Everything else is locked in _private/.
PRIVATE = {
    "health": "This is about health, the body, weight, diet, medicine, sleep or mental health.",
    "money": "This is about money, pay, benefits, debt or what someone can afford.",
    "intimate": "This is about sex, dating, attraction or private body parts.",
    "people": "This is about their family, a friend, a coworker or another real person.",
    "legal": "This is about the law, police, crime, drugs, alcohol or weapons.",
    "beliefs": "This is about politics, religion, race, ethnicity or immigration.",
}
SAFE = "This is an ordinary, harmless fact about their hobbies, games, food, music, gear or interests."
PRIVATE_WORDS = re.compile(
    r"\b(pounds?|lbs|weigh\w*|diet\w*|fat\w*|apnea|ozempic\w*|wegovy|semaglutide|meds?|medication|medicine|pills?|"
    r"vitamin\w*|iv|doctor|hospital|sick|infect\w*|injur\w*|pain|puk\w*|vomit\w*|therapy|therapist|depress\w*|"
    r"anxiety|panic|adhd|mental|suicid\w*|plasma|blood|body|armpits?|smell\w*|fart\w*|"
    r"ebt|food stamps|welfare|broke|debt|rent|rich|poor|afford\w*|money|paid|payed|pay|salary|bucks|dollars|"
    r"\$\d+|layed|laid|fired|unemployed|underemployed|hire\w*|rejected|"
    r"penis|dick|sex\w*|horny|nude\w*|pregnan\w*|dated|dating|date|chick|girlfriend|boyfriend|gf|bf|crush|"
    r"situationship|sitationship|femboys?|gay|lesbian|trans|wom[ae]n|girls?|"
    r"wife|husband|divorc\w*|mom|mother|mama|mommy|dad|daddy|papa|father|parents?|sister|brother|siblings?|"
    r"grand\w*|aunt|uncle|cousin|son|daughter|famil\w*|folks|roo?mm?ate|coworkers?|supervisors?|boss|manager|hr|"
    r"adopted|jail|prison|arrest\w*|cops?|police|court|lawyers?|sued|lawsuit|stole|steal\w*|theft|illegal|"
    r"deport\w*|immigra\w*|citizen\w*|border|weed|drugs?|high|drunk|beer|liquor|alcohol\w*|drinker|vape|smok\w*|"
    r"firearm|gun|weapon|knife|vote\w*|election|politic\w*|trump|democrat\w*|republican\w*|"
    r"pray\w*|god|church|relig\w*|brown|black|white|latino|latina|mexican|korean|iranians?|asian|race|"
    r"homeless|homless|ugly|health\w*|banned|ban)\b", re.I)
# harmless phrases that contain a flagged word ("black coffee", "white rice") are removed before the check
BENIGN = re.compile(r"\b(black|white|brown) (coffee|rice|tea|monster|beans?|bread|sugar|keyboard|mouse|shirt|car|tesla)\b"
                    r"|\bhigh (score|school|quality|res|resolution|fps|settings)\b", re.I)


def sensitive(text):
    return bool(PRIVATE_WORDS.search(BENIGN.sub(" ", text)))
BAD_NAMES = {"dont", "about", "think", "really", "thing", "things", "stuff", "other", "something", "anything",
             "everything", "because", "would", "could", "should", "just", "like", "know", "want", "going", "gonna",
             "wanna", "never", "always", "still", "even", "only", "much", "more", "some", "every", "people", "time"}
FILLER = {"i", "im", "am", "a", "an", "the", "my", "me", "have", "has", "use", "used", "to", "be", "is", "do", "of"}


class Models:
    """NLI (yes/no checks) + embeddings (meaning). No LLM: nothing here can write text."""
    def __init__(self, device):
        import torch
        from distill import Embedder
        self.torch, self.device = torch, device
        self.nli = L.NLIBackend(L.NLI_MODEL, device)
        self.emb = Embedder(EMBEDDER, device)

    def entail(self, pairs):
        """P(premise entails hypothesis) for many (premise, hypothesis) pairs, batched."""
        if not pairs:
            return []
        logits = self.nli._logits(pairs, batch=64)
        return self.torch.softmax(logits, dim=1)[:, self.nli.ent].tolist()


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


DECISIONS = os.path.join(CABINET, "_decisions.json")  # your y / n / p answers from --review (kept forever)


def decisions():
    if not os.path.exists(DECISIONS):
        return {}
    with open(DECISIONS, encoding="utf-8") as f:
        return json.load(f)


def save_decisions(dec):
    os.makedirs(CABINET, exist_ok=True)
    with open(DECISIONS, "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=2, ensure_ascii=False)


def build_answer(d):
    """What the bot may SAY from a drawer: what you typed + facts that passed every automatic check
    (stable 3+ times on 2+ days, both readings agreed on the drawer, passed default-deny privacy)."""
    mine = d.get("typed_by_you", "")
    approved = [r["fact"] for r in d.get("facts", []) if (r.get("speak") or r.get("approved")) and not r["old"]
                and r["fact"] not in mine.lower()][:4]
    return "; ".join(x for x in [mine] + approved if x)


def write_cabinet(drawers, unsure, private, typed, other=()):
    os.makedirs(PRIVATE_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(CABINET, "*.json")) + glob.glob(os.path.join(PRIVATE_DIR, "*.json")):
        if os.path.basename(path) != "_decisions.json":
            os.remove(path)  # rebuilt from scratch each time (typed facts, decisions and _blocked.txt are kept)
    for key in set(drawers) | set(typed):
        d = drawers.get(key, {})
        about, question = TOPICS.get(key, (d.get("about", key), d.get("question", "")))[:2]
        out = {"drawer": key, "question": question, "about": about, "typed_by_you": typed.get(key, ""),
               "facts": d.get("facts", [])}
        out["answer"] = build_answer(out)
        with open(os.path.join(CABINET, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    with open(UNSURE, "w", encoding="utf-8") as f:
        json.dump(unsure, f, indent=2, ensure_ascii=False)
    with open(OTHER_FILE, "w", encoding="utf-8") as f:
        json.dump(list(other), f, indent=2, ensure_ascii=False)
    for kind, rows in private.items():
        with open(os.path.join(PRIVATE_DIR, f"{kind.replace(' ', '_')}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- the pipeline
def pull(M, msgs, skip, args):
    """Step 1-3: messages -> (message, fact, score). Every fact is cut from your own words."""
    t0 = time.time()
    cand = [m for m in msgs if ME.search(m["text"]) and len(m["text"].split()) >= 3
            and not m["text"].rstrip().endswith("?")]
    names = addressees(msgs)
    pairs = [(m, p) for m in cand for p in pieces(m["text"], names)]
    pairs = [(m, f) for m, f in pairs if f not in skip and not re.fullmatch(r"(i am|im|i'm) (a |an )?\w+", f)]
    n_pulled = len(pairs)
    # check 1: is it a lasting fact about you?  check 2: does your whole message back it?
    lasting = M.entail([(f, SELF_FACT) for _, f in pairs])
    backed = M.entail([(m["text"], f) for m, f in pairs])
    # "Lasting" is decided by stasis over time (said again on other days), not by one NLI guess.
    # NLI's lasting score only drops obvious non-facts and softly weights the rest.
    good = [(m, f, b * (0.5 + 0.5 * l)) for (m, f), l, b in zip(pairs, lasting, backed)
            if l >= args.lasting and b >= args.verify]
    stats = {"messages": len(cand), "pulled": n_pulled, "kept": len(good), "secs": 0.0,
             "lasting_ok": sum(l >= args.lasting for l in lasting), "backed_ok": sum(b >= args.verify for b in backed)}
    stats["secs"] = round(time.time() - t0, 1)
    return good, stats


def organize(M, good, msgs, args, dec):
    """Lock private (default-deny) -> group by meaning -> file each group once -> stasis -> rows."""
    if not good:
        return {}, [], {}, [], {"private": 0, "filed": 0, "unsure": 0, "other": 0, "facts": 0, "speak": 0,
                                "drawers": 0, "new": [], "groups": 0}
    E = M.emb([" ".join(w for w in f.split() if w not in FILLER) or f for _, f, _ in good])

    # 1. lock: default-deny
    kinds = list(PRIVATE)
    nP = len(kinds)
    pp = M.entail([(t, h) for m, f, _ in good for t in (f, m["text"]) for h in PRIVATE.values()])
    safe = M.entail([(f, SAFE) for _, f, _ in good])
    lock = []
    for i, (m, f, _) in enumerate(good):
        said = dec.get(f)
        if said == "public":
            lock.append(None); continue          # you approved it
        if said == "private":
            lock.append("you said private"); continue
        ps = pp[i * 2 * nP:(i + 1) * 2 * nP]
        best = max(range(2 * nP), key=lambda j: ps[j])
        if ps[best] >= args.private_bar:
            lock.append(kinds[best % nP])
        elif sensitive(f) or sensitive(m["text"]):
            lock.append("sensitive words")
        elif safe[i] < args.safe_bar:
            lock.append("not clearly safe")
        else:
            lock.append(None)

    # 2. group by MEANING first (embeddings), so the same fact can never land in two drawers
    df = Counter(w for _, f, _ in good for w in set(WORD.findall(f)))
    rare = {w for w, c in df.items() if len(w) >= 4 and w not in STOP and w not in FILLER and c <= max(3, 0.1 * len(good))}

    def key_words(f):
        return {w.rstrip("s") for w in WORD.findall(f) if w in rare}

    def group(idx):
        made = []
        for i in sorted(idx, key=lambda i: -good[i][2]):
            ki = key_words(good[i][1])
            for g in made:
                sim = float(E[i] @ E[g[0]])
                if sim >= args.same or (ki & key_words(good[g[0]][1]) and sim >= 0.7):
                    g.append(i); break
            else:
                made.append([i])
        return made

    pub = [i for i in range(len(good)) if not lock[i]]
    groups = group(pub)

    # 3. file each GROUP once: reading 1 = decision model (NLI, every member, piece + message),
    #    reading 2 = embeddings (group centre vs each drawer's description + your typed facts)
    keys = list(TOPICS)
    hyps = [TOPICS[k][2] for k in keys] + [OTHER]
    H = len(hyps)
    place, sure = {}, {}
    if pub:
        lg = M.nli._logits([(t, h) for i in pub for t in (good[i][1], good[i][0]["text"]) for h in hyps],
                           batch=64)[:, M.nli.ent].view(len(pub), 2, H)
        pr = M.torch.softmax(lg, dim=2).mean(1)  # [n_pub, H]
        row_of = {i: r for r, i in enumerate(pub)}
        typed = typed_by_you()
        seeds = M.emb([f"{TOPICS[k][0]}. {TOPICS[k][1]} {typed.get(k, '')}" for k in keys]).cpu()
        Ec = E.cpu()
        pr = pr.cpu()
        for g in groups:
            nli = pr[[row_of[i] for i in g]].mean(0)
            centre = M.torch.nn.functional.normalize(Ec[g].mean(0), dim=0)
            sims = M.torch.cat([seeds @ centre, M.torch.tensor([0.3])])
            emb = M.torch.softmax(sims * 15, dim=0)
            mix = (nli + emb) / 2
            win = int(mix.argmax())
            agree = (int(nli.argmax()) == win) + (int(emb.argmax()) == win)
            conf = float(mix[win]) * agree / 2  # probability stasis: average reading x agreement
            where = None if win == H - 1 else (keys[win] if conf >= args.topic else "_unsure")
            sure[g[0]] = agree == 2  # both readings picked the same drawer
            for i in g:
                place[i] = where

    # 4. new drawers from groups that fit nowhere, named by their most common distinctive word
    loose = [g for g in groups if place.get(g[0]) is None]
    new = {}
    if len(loose) >= 6:
        C = M.torch.nn.functional.normalize(M.torch.stack([E[g].mean(0) for g in loose]), dim=1)
        lab = kmeans(C, max(2, min(args.k, len(loose) // 6)))
        clusters = defaultdict(list)
        for g, c in zip(loose, lab):
            clusters[c].append(g)
        for cl in clusters.values():
            members = [i for g in cl for i in g]
            if len(members) < 6:
                continue
            words = Counter(w for i in members for w in set(WORD.findall(good[i][1]))
                            if w not in STOP and w not in FILLER and w not in BAD_NAMES and len(w) >= 4)
            key = ALIASES.get(words.most_common(1)[0][0], words.most_common(1)[0][0]) if words else ""
            if not key or key in VAGUE or key in BAD_NAMES or sensitive(key):
                continue
            if key not in TOPICS:
                new.setdefault(key, (f"things about {key}", f"what about {key}?"))
            for i in members:
                place[i] = key

    # 5. stasis over time -> one row per group
    all_days = [d for d in (day(m["time"]) for m in msgs) if d]
    end = max(all_days) if all_days else None

    def row(g):
        days = sorted({d for d in (day(good[i][0]["time"]) for i in g) if d})
        n, nd = len(g), len(days)
        p = sum(good[i][2] for i in g) / n
        lead = min(g, key=lambda i: len(good[i][1]) - 3 * float(E[i] @ E[g[0]]))  # short and central
        text = good[lead][1]
        said = [dec.get(good[i][1]) for i in g]
        stable = n >= args.min_mentions and nd >= 2
        return {"fact": text, "mentions": n, "days": nd, "source": "pulled",
                "speak": "public" in said or (stable and sure.get(g[0], False)),  # automatic, no clicking
                "first": str(days[0]) if days else "", "last": str(days[-1]) if days else "",
                "stasis": round(p * (1 - math.exp(-n / 3)) * (1 - math.exp(-nd / 2)), 3),
                "stable": n >= args.min_mentions and nd >= 2,
                "approved": "public" in said,
                "old": bool(end and days and (end - days[-1]).days > args.old_days),
                "examples": list(dict.fromkeys(good[i][0]["text"] for i in g))[:3]}

    def rows(gs):
        out = [row(g) for g in gs]
        out.sort(key=lambda r: (-r["speak"], -r["stable"], -r["stasis"]))
        return out

    by = defaultdict(list)
    for g in groups:
        by[place.get(g[0])].append(g)
    drawers = {}
    for key, gs in by.items():
        if key and key != "_unsure":
            about, q = TOPICS[key][:2] if key in TOPICS else new[key]
            drawers[key] = {"about": about, "question": q, "facts": rows(gs)}
    unsure, other = rows(by.get("_unsure", [])), rows(by.get(None, []))
    plock = defaultdict(list)
    for i, k in enumerate(lock):
        if k:
            plock[k].append(i)
    private = {k: rows(group(idx)) for k, idx in plock.items()}
    allrows = [r for d in drawers.values() for r in d["facts"]]
    stats = {"private": sum(1 for k in lock if k), "other": len(other), "unsure": len(unsure),
             "filed": sum(len(d["facts"]) for d in drawers.values()), "facts": len(allrows),
             "speak": sum(r["speak"] for r in allrows), "drawers": len(drawers), "new": list(new),
             "groups": len(groups)}
    return drawers, unsure, private, other, stats


def learn(args):
    t0 = time.time()
    step = lambda msg: print(f"{dim(f'[{time.time() - t0:5.1f}s]')} {msg}", file=sys.stderr)
    msgs = load_msgs()
    typed, skip, dec = typed_by_you(), blocked(), decisions()
    skip |= {f for f, v in dec.items() if v == "forget"}
    M = Models(L.pick_device(args.device))

    good, ps = pull(M, msgs, skip, args)
    step(f"1. find   : {ps['messages']} of {len(msgs)} messages talk about you")
    step(f"2. pull   : {ps['pulled']} pieces cut straight from your messages (no generation)")
    step(f"3. check  : {ps['kept']} backed by your message and not obvious junk")
    drawers, unsure, private, other, st = organize(M, good, msgs, args, dec)
    step(f"4. lock   : {st['private']} locked in _private/ (default-deny: private unless clearly safe)")
    step(f"5. group  : the rest grouped by meaning into {st['groups']} distinct facts")
    step(f"6. file   : {st['filed']} filed into drawers (NLI + embeddings must agree), {st['unsure']} unsure, "
         f"{st['other']} fit no drawer" + (f", new drawers: {', '.join(st['new'])}" if st["new"] else ""))
    step(f"7. stasis : {st['speak']} facts said {args.min_mentions}+ times on 2+ days with both readings agreeing "
         f"-> the bot may say them")
    write_cabinet(drawers, unsure, private, typed, other)
    show(everything=args.all)
    print(dim("Optional: python learnfacts.py --review lets you veto any of them (n = forget, p = private)."))


# ---------------------------------------------------------------- showing the cabinet
def show(private=False, everything=False):
    drawers = read_cabinet()
    if not drawers:
        sys.exit("cabinet is empty: python learnfacts.py")

    def score(k):
        return -max([r["stasis"] for r in drawers[k]["facts"]] + [1 if drawers[k].get("typed_by_you") else 0])
    print(f"\n{CABINET}/   {dither(' pulled from your chat ', 0, palette=NEON)}  {dither(' drawers ', 0)}  "
          f"{rgb(BLUES[4])}✎ you typed{reset()}   ✔ = passed every check, the bot may say it")
    for row, key in enumerate(sorted(drawers, key=score)):
        d = drawers[key]
        facts = d["facts"]
        ok = [r for r in facts if r.get("speak") or r.get("approved")]
        wait = []
        rest = [r for r in facts if not (r.get("speak") or r.get("approved"))]
        print(f"├── {dither(' ' + key + '.json ', row, 0.0, 0.4)}  {d['question'] or d['about']}  "
              f"{dim(f'{len(facts)} facts · {len(ok)} the bot may say')}")
        if d.get("typed_by_you"):
            print(f"│     {rgb(BLUES[4])}✎ you typed{reset()}  {d['typed_by_you']}")
        for j, r in enumerate(ok + wait):
            meta = dim(f"{r['mentions']}x · {r['days']} days · {r['first'][:7]} → {r['last'][:7]}")
            mark = "✔"
            print(f"│   {mark} {bar(r['stasis'], palette=NEON)} {r['stasis']:.2f}  "
                  f"{dither(' ' + r['fact'] + ' ', row + j, palette=NEON)}  {meta}")
        if rest and everything:
            for j, r in enumerate(rest):
                print(f"│       {dim('·')} {dither(' ' + r['fact'] + ' ', j, 0.55, 1.0, palette=NEON)}  "
                      f"{dim(str(r['mentions']) + 'x ' + r['first'][:10])}")
        elif rest:
            print(f"│     {dim(f'+ {len(rest)} more kept (said less often, or the readings disagreed) - never said')}")
        print("│")
    for path, label in [(UNSURE, "facts the two readings filed differently"), (OTHER_FILE, "facts that fit no drawer")]:
        rows = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
        print(f"├── {dim(os.path.basename(path)):14} {dim(f'{len(rows)} {label} (kept, not used)')}")
        for r in rows[:15 if everything else 0]:
            print(f"│     {dim('? ' + r['fact'])}")
    print(f"├── {dim('_blocked.txt')}   {dim(f'{len(blocked())} facts you told it to forget')}")
    dec = decisions()
    print(f"├── {dim('_decisions.json')} {dim(f'{len(dec)} answers from your reviews (kept when re-learning)')}")
    kinds = {os.path.basename(p)[:-5]: json.load(open(p, encoding="utf-8"))
             for p in sorted(glob.glob(os.path.join(PRIVATE_DIR, "*.json")))}
    total = sum(len(v) for v in kinds.values())
    print(f"└── {dim('_private/')}      {dim(f'🔒 {total} facts locked, the bot never opens this folder')}  "
          f"{dim('(' + ', '.join(f'{k} {len(v)}' for k, v in kinds.items()) + ')') if kinds else ''}")
    if private:
        for k, rows in kinds.items():
            for r in rows:
                n = r["mentions"]
                print("      " + dim(f"🔒 {k}: {r['fact']} ({n}x)"))
    elif total:
        print(dim("      see them: python learnfacts.py --show --private"))
    print()


def set_fact(fact, verdict):
    """Apply one review answer to the cabinet files right away."""
    dec = decisions()
    dec[fact] = verdict
    save_decisions(dec)
    if verdict == "forget":
        with open(BLOCKED, "a", encoding="utf-8") as f:
            f.write(fact + "\n")
    for key, d in read_cabinet().items():
        hit = [r for r in d["facts"] if r["fact"] == fact]
        if not hit:
            continue
        if verdict == "public":
            for r in hit:
                r["approved"] = r["speak"] = True
        else:
            d["facts"] = [r for r in d["facts"] if r["fact"] != fact]
            if verdict == "private":
                os.makedirs(PRIVATE_DIR, exist_ok=True)
                path = os.path.join(PRIVATE_DIR, "you_said_private.json")
                rows = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(rows + hit, f, indent=2, ensure_ascii=False)
        d["answer"] = build_answer(d)
        with open(os.path.join(CABINET, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)


def review(everything=False):
    """OPTIONAL veto: go through what the bot will say. Enter = keep, n = forget it, p = private, q = quit."""
    drawers = read_cabinet()
    todo = [(k, r) for k, d in drawers.items() for r in d["facts"] if r.get("speak") or everything]
    todo.sort(key=lambda kr: -kr[1]["stasis"])
    if not todo:
        print("Nothing to veto. (--review --all also goes through the facts the bot does NOT say)")
        return
    print(f"\n{len(todo)} facts the bot may say.  Enter = keep   n = forget it   p = private   q = quit\n")
    done = Counter()
    for n, (key, r) in enumerate(todo, 1):
        print(f"{dim(f'[{n}/{len(todo)}]')} {dither(' ' + key + ' ', n, 0.0, 0.4)}  "
              f"{dither(' ' + r['fact'] + ' ', n, palette=NEON)}  {dim(str(r['mentions']) + 'x on ' + str(r['days']) + ' days')}")
        for m in r["examples"][:2]:
            print(f"      {dim('from: ' + m[:90])}")
        while True:
            try:
                a = input("   keep? [Enter]/n/p/q > ").strip().lower()[:1] or "y"
            except (EOFError, KeyboardInterrupt):
                a = "q"
            if a in ("y", "n", "p", "s", "q"):
                break
        if a == "q":
            break
        if a == "s":
            continue
        set_fact(r["fact"], {"y": "public", "n": "forget", "p": "private"}[a])
        done[a] += 1
    print(f"\nApproved {done['y']}, forgot {done['n']}, made private {done['p']}. "
          f"Saved in {DECISIONS} (kept when you re-learn).")


def forget(text):
    set_fact(text.strip().lower(), "forget")
    print(f"Forgot \"{text.strip().lower()}\". It won't be learned again.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="show the cabinet (no models, instant)")
    ap.add_argument("--all", action="store_true", help="also list facts said less often / review them too")
    ap.add_argument("--private", action="store_true", help="with --show: also list the locked private facts")
    ap.add_argument("--review", action="store_true", help="optional: veto facts the bot would say (n / p)")
    ap.add_argument("--forget", metavar="FACT", help="remove a fact and never learn it again")
    ap.add_argument("--k", type=int, default=6, help="most new drawers to make")
    ap.add_argument("--topic", type=float, default=0.25, help="stasis bar to file a group in a drawer")
    ap.add_argument("--private-bar", dest="private_bar", type=float, default=0.3, help="NLI bar for a private kind")
    ap.add_argument("--safe-bar", dest="safe_bar", type=float, default=0.2, help="NLI bar for 'clearly safe'")
    ap.add_argument("--lasting", type=float, default=0.005, help="NLI floor that drops obvious non-facts")
    ap.add_argument("--verify", type=float, default=0.5, help="NLI bar for 'your message backs the piece'")
    ap.add_argument("--same", type=float, default=0.8, help="how similar two facts must be to merge")
    ap.add_argument("--min-mentions", type=int, default=3, help="said this many times (on 2+ days) to be reviewed")
    ap.add_argument("--old-days", type=int, default=180, help="not said for this long = 'used to'")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.forget:
        forget(args.forget)
    elif args.review:
        review(everything=args.all)
    elif args.show:
        show(private=args.private, everything=args.all)
    else:
        learn(args)


if __name__ == "__main__":
    main()
