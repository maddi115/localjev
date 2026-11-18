"""Twitch pilot: decide first, then reply in YOUR style.

  1. prep   - read your Twitch chat export, keep only the message text (drops IP, city, ids)
  2. index  - embed your past messages on the GPU so similar ones can be found fast
  3. chat   - type a chat line, watch it get decided, gated by stasis, then answered in your voice

Pipeline for every incoming message:
  decision model -> what kind of message? which fact is it asking about?
  stasis gate    -> spam / harassment / not sure  => no reply
  retrieval      -> 8 of your past messages most like this one (your voice examples)
  replies        -> NO generation by default: built from your facts, or picked from your own past messages
  safe replies   -> also builds replies straight from the facts, in your habits ("mostly ... tbh") - true by construction
  truth check    -> every unusual word must come from your facts (valo ~ valorant); unknown => must say not sure
  style judge    -> your-chat vs normal-English letter patterns; keeps the one most like you
"""
import argparse, csv, json, os, random, re, statistics, sys
from collections import Counter
from types import SimpleNamespace

import localjev as L
from colors import BLUES, bar, dim, dither, reset, rgb  # gradient blue dither

OUT = "twitch"
MSGS = os.path.join(OUT, "my_messages.jsonl")
INDEX = os.path.join(OUT, "index.npy")
FACTS = os.path.join(OUT, "facts.json")
CABINET = os.path.join(OUT, "facts")  # one file per drawer, made by learnfacts.py
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
WRITER = "Qwen/Qwen2.5-1.5B-Instruct"

KINDS = {
    "question": "Asks a question someone could answer",
    "hype": "Hype, jokes, reactions or excitement about the stream or game",
    "greeting": "Says hi, bye, or welcomes someone",
    "personal": "Shares something about their own day or life",
    "spam": "Spam, ads, scams, links, or self promotion",
    "harassment": "Insults, hate, harassment, or threats toward someone",
}
NO_REPLY = {"spam", "harassment"}
EMOJI = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F]")
SAFE = {"hype", "greeting", "personal"}  # a wrong reply here is harmless, so a lower bar to answer
JUNK = re.compile(r"[\u200b-\u200f\u2060\u3000-\u9fff\uac00-\ud7af\ufffd]")  # zero-width + CJK/Korean leftovers
BOTISH = re.compile(r"\b(as an ai|computer program|language model|i'?m (just )?a (little )?bot|i'?m sorry|"
                    r"could you (please )?(clarify|tell me)|let me know|i need more information)\b")
PLAN = re.compile(r"\b(on my way|omw|packed|packing|heading out|head out|see (you|u|ya) there|i'?m ready|let'?s go again|"
                  r"gtfo|gotta go|gotta run|i'?m out|im off|let'?s go somewhere)\b")
QUESTION = re.compile(r"\?\s*$|^(what|whats|what's|wat|where|how|who|why|which|when|do|does|did|are|is|can|have|"
                      r"you|u|ur|fav|seen|ever|watched|played|tried|heard|wanna)\b|\banything\b")
HYPE_START = {"w", "gg", "ggs", "pog", "lets", "lmao", "lmaoo", "lol", "omg", "yooo", "sheesh", "clutch", "insane"}
GREET = re.compile(r"^(hi|hey|hello|yo|sup|wassup|bye|cya|goodnight|good night|gn|good morning|gm|later)\b")
PRIVATE_Q = re.compile(r"\b(pills?|meds?|medication|medicine|weigh\w*|pounds|lbs|money|salary|paid|income|rent|debt|"
                       r"sick|doctor|therapy|sex\w*|girlfriend|boyfriend|single|religio\w*|pray|drugs?|address)\b")
SLANG = set("""lol lmao lmfao rofl xd xdd xddd tbh idk idc imo imho ngl fr frfr ong bro bruh dude man yo ya yah yea yeah
yep nah naw u ur urs ya yall y'all im ive ill id dont cant wont isnt aint gonna wanna gotta kinda sorta tho bc cuz cos
pls plz thx ty np btw atm rn af asf omg omfg wtf wth smh fyi irl gg ez pog poggers kekw lul lulw w l ok okay k kk
haha hahaha hehe hmm hm uh uhh um umm ah ahh oh ohh ooh aye ayy ayo sus lowkey highkey deadass bet cap nocap vibe vibes
fam homie goat goated mid based cringe ppl prolly def lit dope sick fire valid ight aight finna boutta
couldn wouldn shouldn didn doesn isn wasn aren weren hasn haven ain dunno wbu hbu""".split())
WORD = re.compile(r"[a-z0-9']+")
OPENERS = ["mostly", "tbh", "ngl", "honestly", "lol", "bro", "dude", "man", "yo", "uhh", "um"]
FRAMES = {"games": "i play {}", "setup": "i use {}", "home": "im from {}", "music": "i listen to {}",
          "food": "i eat {}", "watching": "i watch {}", "work": "i work {}", "streaming": "im {}"}
STOP = {"i", "im", "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "my", "me", "is", "am", "are", "was",
        "have", "has", "do", "does", "you", "your", "it", "for", "with", "play", "eat", "use", "live", "alot", "lot",
        "from", "what", "whats", "that", "this", "any", "some", "like", "yet", "rn", "so", "get", "got", "into",
        # everyday verbs: they say how, not what ("make tiktoks" is about tiktoks, not about "make")
        "make", "making", "made", "work", "working", "playing", "watch", "watching", "go", "going", "doing",
        "using", "love", "want", "see", "saw", "know", "think", "here", "there", "now", "too", "really",
        "take", "taking", "took", "post", "posting", "lot", "lots", "good", "new", "anything", "something"}


def content_words(text):
    """Words that carry meaning, plus initials of multi-word names ("los angeles" -> "la")."""
    ws = [w for w in WORD.findall(text.lower()) if w not in STOP and len(w) >= 2]
    long = [w for w in WORD.findall(text.lower()) if w not in STOP and len(w) >= 3]
    if len(long) >= 2:
        ws.append("".join(w[0] for w in long))
    return ws


SYNONYMS = [{"pictures", "picture", "pics", "pic", "photos", "photo", "photography"},
            {"vids", "vid", "videos", "video", "clips"}, {"movie", "movies", "film", "films", "flick", "flicks"},
            {"job", "jobs", "work", "gig", "gigs"}, {"gpu", "graphics", "3050rtx", "rtx"},
            {"cali", "california", "la", "los angeles"}, {"bike", "bikes", "cycle", "cycling", "biking"},
            {"song", "songs", "music", "band", "bands"}, {"food", "eat", "eating", "snack", "snacks"}]
GENERIC_Q = {"what", "whats", "which", "you", "your", "yours", "do", "does", "did", "like", "the", "play", "use",
             "have", "got", "any", "are", "fav", "favorite", "favourite", "some", "kind", "yo", "hey", "bro"}
YESNO = re.compile(r"^(do|did|are|have|can|is|does|would|will|you|u)\b")
ENDERS = ["tbh", "lol", "lmao", "bro", "man", "dude", "xd", "wbu", "hbu", "ngl", "fr"]
UNSURE = re.compile(r"\b(idk|not sure|dunno|no idea|i forgot|good question)\b")
CRUDE = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|ass|asses|cheeks|dick\w*|pussy|cum|sex\w*|horny|suck\w*|"
                   r"eww+|kill\w*|die|dead|retard\w*|stupid|dumb|idiot|hate|gross|nasty|cock|tits?|boobs?|"
                   r"thot|hoe|slut|whore|gay|fag\w*|n[i1]gg\w*)\b")
AIMED = re.compile(r"\b(u|ur|you|your|youre|ya|she|he|her|him|they|them|their|his)\b")
ROLE = {"viewer": "You are a viewer chatting in someone else's stream, not the streamer. Don't say you're going "
                   "somewhere, packing, or ready to start anything.",
        "streamer": "You are the streamer, answering your own chat."}
NAMES_THING = "The speaker names a specific movie, game, show, product, brand or place."
NOT_SURE = "The speaker says they don't know or aren't sure, or asks a question back."

FACTS_TEMPLATE = {
    "games": {"about": "what games you play or like", "answer": "EDIT ME: games you play"},
    "setup": {"about": "your PC, mouse, keyboard, headset or setup", "answer": "EDIT ME: your setup"},
    "from": {"about": "where you are from or live", "answer": "EDIT ME: or delete this fact"},
    "streaming": {"about": "whether you stream or have a channel", "answer": "EDIT ME"},
}

DEMO = [
    "viewer1: yo what games do you play?",
    "viewer2: LETS GOOOO that clutch was insane",
    "viewer3: what mouse do you use?",
    "viewer4: buy cheap followers at bestfollows dot net",
    "viewer5: hey everyone just got here",
    "viewer6: what's your favorite movie?",
    "viewer7: you're trash and everyone hates you",
]


# ---------------------------------------------------------------- 1. prep
def clean(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"@\w+", "@viewer", text)  # never keep other people's names
    return text


def cmd_prep(args):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with open(os.path.expanduser(args.csv), encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    logins = Counter(r.get("login", "") for r in rows)
    me = logins.most_common(1)[0][0] if logins else ""
    seen, keep = set(), []
    dropped = Counter()
    for r in rows:
        if args.main_only and r.get("login", "") != me:
            dropped["someone else"] += 1; continue
        t = clean(r.get("body") or r.get("body_full"))
        if len(t) < 2:
            dropped["empty"] += 1; continue
        if t.startswith("!"):
            dropped["bot command"] += 1; continue
        if "http" in t or "www." in t:
            dropped["link"] += 1; continue
        k = t.lower()
        if k in seen:
            dropped["duplicate"] += 1; continue
        seen.add(k)
        keep.append({"text": t, "channel": r.get("channel", ""), "time": r.get("time", ""),
                     "reply": r.get("is_reply", "") in ("t", "true", "1")})
    keep.sort(key=lambda m: m["time"])
    with open(MSGS, "w", encoding="utf-8") as f:
        for m in keep:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    if not os.path.exists(FACTS):
        with open(FACTS, "w", encoding="utf-8") as f:
            json.dump(FACTS_TEMPLATE, f, indent=2)

    words = [len(m["text"].split()) for m in keep]
    lower = sum(m["text"] == m["text"].lower() for m in keep) / max(1, len(keep))
    toks = Counter(w for m in keep for w in m["text"].split() if re.fullmatch(r"[A-Z][a-z]+[A-Z]\w*|[A-Z]{3,}", w))
    chans = Counter(m["channel"] for m in keep)
    print(f"\nKept {len(keep)} of {len(rows)} messages from your account(s)  -> {MSGS}")
    print("Dropped:", ", ".join(f"{k} {v}" for k, v in dropped.most_common()) or "nothing")
    print("\nYour style, in numbers:")
    print(f"  typical length : {statistics.median(words) if words else 0:.0f} words (median)")
    print(f"  all lowercase  : {lower:.0%} of messages")
    print(f"  channels       : {len(chans)}")
    print(f"  emote-ish words: {', '.join(w for w, _ in toks.most_common(10)) or 'none'}")
    print("\n5 random examples:")
    for m in random.Random(0).sample(keep, min(5, len(keep))):
        print("  -", m["text"][:100])
    print(f"\nEdit your facts before chatting: {FACTS}")


def load_facts():
    """The filing cabinet (twitch/facts/*.json, skipping _private and _unsure) if it exists, else facts.json."""
    import glob
    paths = [p for p in sorted(glob.glob(os.path.join(CABINET, "*.json"))) if not os.path.basename(p).startswith("_")]
    if paths:
        out = {}
        for p in paths:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("answer"):
                out[d.get("drawer") or os.path.basename(p)[:-5]] = {"about": d["about"], "answer": d["answer"],
                                                                    "question": d.get("question", "")}
        return out
    with open(FACTS, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 2. index
def load_msgs():
    if not os.path.exists(MSGS):
        sys.exit("run prep first: python twitch.py prep ~/chat_messages.csv")
    with open(MSGS, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def cmd_index(args):
    import numpy as np
    from distill import Embedder
    msgs = load_msgs()
    emb = Embedder(EMBEDDER, L.pick_device(args.device))
    vecs = emb([m["text"] for m in msgs], batch=256).cpu().numpy().astype("float32")
    np.save(INDEX, vecs)
    print(f"Embedded {len(msgs)} messages -> {INDEX}  shape {vecs.shape}")


# ---------------------------------------------------------------- the questions the bot asks
def questions(facts):
    fact_criteria = {k: f"Asks about {v['about']}" for k, v in facts.items()}
    fact_criteria["none"] = "Asks about something else not listed here"
    return {
        "kind": {"type": "choice", "instructions": "What kind of Twitch chat message is this?", "criteria": KINDS},
        "fact": {"type": "choice", "instructions": "Which topic is the message asking about?",
                 "criteria": fact_criteria},
    }


def cmd_questions(args):
    """Write the bot's questions + a sample of your messages, so distill can train a Twitch student."""
    facts = load_facts()
    path = os.path.join(OUT, "questions.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(questions(facts), f, indent=2)
    msgs = load_msgs()
    pick = random.Random(0).sample(msgs, min(args.n, len(msgs)))
    sample = os.path.join(OUT, "sample.jsonl")
    with open(sample, "w", encoding="utf-8") as f:
        for m in pick:
            f.write(json.dumps({"text": m["text"]}, ensure_ascii=False) + "\n")
    print(f"Wrote {path} and {len(pick)} of your messages to {sample}")
    print("If you edit facts.json later, run this again and re-train the student.")


# ---------------------------------------------------------------- your style as numbers
class CharStyle:
    """Character 3-gram model of YOUR messages vs normal English (Wikipedia).
    Score > 0: the text looks more like you than like generic English. Pure counting, no training."""
    def __init__(self, mine, english_path, max_chars=3_000_000):
        self.me, self.me_ctx = self.count(mine)
        generic = []
        if english_path and os.path.exists(english_path):
            with open(english_path, encoding="utf-8", errors="ignore") as f:
                generic = [l.lower() for l in f.read(max_chars).splitlines() if l.strip()]
        self.gen, self.gen_ctx = self.count(generic) if generic else (Counter(), Counter())
        self.english_path = english_path
        self.V = 60

    def common_words(self, n, max_chars=25_000_000):
        """The n most common English words (from english.txt)."""
        if not (self.english_path and os.path.exists(self.english_path)):
            return set()
        with open(self.english_path, encoding="utf-8", errors="ignore") as f:
            c = Counter(WORD.findall(f.read(max_chars).lower()))
        return {w for w, _ in c.most_common(n)}

    @staticmethod
    def count(texts):
        tri, ctx = Counter(), Counter()
        for t in texts:
            t = f"^^{t}$"
            for i in range(len(t) - 2):
                tri[t[i:i + 3]] += 1; ctx[t[i:i + 2]] += 1
        return tri, ctx

    def logp(self, tri, ctx, g):
        import math
        return math.log((tri[g] + 0.1) / (ctx[g[:2]] + 0.1 * self.V))

    def __call__(self, text):
        t = f"^^{text}$"
        grams = [t[i:i + 3] for i in range(len(t) - 2)]
        if not grams:
            return 0.0
        d = [self.logp(self.me, self.me_ctx, g) - (self.logp(self.gen, self.gen_ctx, g) if self.gen else 0) for g in grams]
        return sum(d) / len(d)


# ---------------------------------------------------------------- 3. chat
class Bot:
    def __init__(self, args):
        import numpy as np, torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from distill import Embedder
        self.np, self.torch, self.args = np, torch, args
        self.msgs = load_msgs()
        if not os.path.exists(INDEX):
            sys.exit("run index first: python twitch.py index")
        self.vecs = np.load(INDEX)
        self.facts = load_facts()
        print(f"Facts: {len(self.facts)} drawers ({', '.join(self.facts)})", file=sys.stderr)
        texts = [m["text"] for m in self.msgs]
        lens = sorted(len(t.split()) for t in texts)
        self.words_med = statistics.median(lens)
        self.words_max = lens[int(len(lens) * 0.9)]  # 90% of your messages are this short or shorter
        self.lower_rate = sum(t == t.lower() for t in texts) / len(texts)
        self.emoji_rate = sum(bool(EMOJI.search(t)) for t in texts) / len(texts)
        self.tag_rate = sum("#" in t for t in texts) / len(texts)
        self.seen = {t.lower() for t in texts}
        self.secret = None  # set after the English word list loads (below)
        first, last = Counter(), Counter()
        for t in texts:
            w = WORD.findall(t.lower())
            if len(w) >= 3:
                first[w[0]] += 1; last[w[-1]] += 1
        self.names = {w for w, c in last.items() if c >= 8 and first[w] >= 3 and len(w) >= 3
                      and w not in STOP and w not in SLANG}  # friends you talk to: keep their names out
        low = [t.lower() for t in texts]
        first = Counter(next((o for o in OPENERS if t.startswith(o + " ")), None) for t in low)
        last = Counter(next((e for e in ENDERS if t.endswith(" " + e)), None) for t in low)
        first.pop(None, None); last.pop(None, None)
        self.openers = [o for o, _ in first.most_common(3)] or ["tbh"]
        self.enders = [e for e, _ in last.most_common(3)] or ["lol"]
        self.unsure_lines = [t for t in low if UNSURE.search(t) and 2 <= len(t.split()) <= 6
                             and not re.search(r"@|\d", t)][:200]
        self.style = CharStyle([t.lower() for t in texts], args.english)
        self.common = self.style.common_words(args.common)  # normal English words: safe to say
        self.secret = private_words() - self.common  # distinctive private words: never say them
        mine = Counter(w for t in texts for w in WORD.findall(t.lower()))
        self.mine = {w for w, c in mine.items() if c >= 3}  # words you yourself use a lot
        self.engine = L.build_engine(SimpleNamespace(
            device=args.device, backends=args.backends, model=L.NLI_MODEL, decider_model=L.DECIDER_MODEL,
            mapika_model=L.MAPIKA_MODEL, stasis=args.stasis, threshold=args.threshold, refine=0,
            cascade=args.cascade, first_threshold=None))
        device = L.pick_device(args.device)
        self.device = device
        nli = [b for b in self.engine.backends if b.name == "nli"]
        self.nli = nli[0] if nli else L.NLIBackend(L.NLI_MODEL, device)  # the truth checker
        self.emb = Embedder(EMBEDDER, device)
        self.llm = None  # default: NO generation. Replies come from your facts and your own past messages.
        if args.writer != "none":
            print(f"Loading writer {args.writer} (generation ON)...", file=sys.stderr)
            self.tok = AutoTokenizer.from_pretrained(args.writer)
            self.llm = AutoModelForCausalLM.from_pretrained(
                args.writer, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        else:
            print("No writer: nothing is generated (facts + your own words only)", file=sys.stderr)

    # --- step 1: decide
    def decide(self, text):
        req = {"state": text, "questions": questions(self.facts)}
        raw = self.engine.solve(req)
        d = {qid: self.engine.summarize(q, raw[qid]) for qid, q in req["questions"].items()}
        d["fact"] = self.pick_drawer(text, d["fact"])
        return d

    def pick_drawer(self, text, nli):
        """Which drawer? Read it two ways and only trust it when they agree (probability stasis):
        reading 1 = the decision model (NLI), reading 2 = how close the message is to each drawer's question."""
        import math
        keys = list(self.facts) + ["none"]
        if not hasattr(self, "drawer_vecs"):
            ks = list(self.facts)
            qv = self.emb([self.facts[k].get("question") or self.facts[k]["about"] for k in ks]).cpu().numpy() if ks else None
            av = self.emb([self.facts[k]["about"] for k in ks]).cpu().numpy() if ks else None
            self.drawer_vecs = (qv, av)
        q = self.emb([text]).cpu().numpy()[0]
        qv, av = self.drawer_vecs
        sims = list(self.np.maximum(qv @ q, av @ q)) if qv is not None else []  # closest of question / about
        sims.append(0.35)  # "none" wins when nothing is close
        ex = [math.exp(15 * x) for x in sims]
        match = {k: e / sum(ex) for k, e in zip(keys, ex)}
        p_nli = nli.get("probabilities", {})
        # reading 3: does the message share a word with a drawer's answer? ("do you hike" -> "i hike")
        asked = [self.norm(w) for w in content_words(text)]
        hits = {k: sum(1 for c in self.facts[k]["answer"].split(";") for w in content_words(c)
                       if any(self.same_word(a, self.norm(w)) for a in asked))
                for k in self.facts}
        total = sum(hits.values())
        readings, weights = [p_nli, match], [1.0, 1.0]
        if total:  # a shared word is strong evidence: it counts double
            readings.append({k: hits.get(k, 0) / total for k in keys}); weights.append(2.0)
        mix = {k: sum(w * r.get(k, 0.0) for r, w in zip(readings, weights)) / sum(weights) for k in keys}
        win = max(mix, key=mix.get)
        agree = sum(max(r, key=r.get) == win for r in readings if r)
        conf = mix[win] * (0.5 + 0.5 * agree / len(readings))
        return {"choice": win, "confidence": round(conf, 4), "agree": f"{agree}/{len(readings)}",
                "action": "act" if conf >= self.args.drawer_threshold else "escalate"}

    # --- step 3: your voice examples
    def examples(self, text, k=8):
        np = self.np
        q = self.emb([text]).cpu().numpy()[0]
        sims = self.vecs @ q
        top = np.argsort(-sims)[:k]
        rnd = random.Random(hash(text)).sample(range(len(self.msgs)), min(4, len(self.msgs)))
        return [self.msgs[i]["text"] for i in list(top) + rnd], q

    # --- step 4: write
    def write(self, text, facts_line, examples, n=8):
        if self.llm is None:
            return []  # generation is off
        torch = self.torch
        ex = "\n".join(f"- {e}" for e in examples)
        system = (
            "You write ONE short Twitch chat message in the voice of the person whose real messages are below. "
            f"{ROLE[self.args.role]} "
            "Copy their style exactly: length, lowercase or caps, slang, typos, emotes. "
            "Only state facts from the FACTS line. If FACTS says unknown, say you're not sure. Never invent facts. "
            "Reply with only the chat message, nothing else.\n\n"
            f"Their real messages:\n{ex}")
        rule = ""
        if facts_line.startswith(("none", "unknown")):
            rule = "Do NOT name any game, product, brand, movie, show or place. "
        if facts_line.startswith("unknown"):
            rule += "You don't know the answer: say you're not sure, or ask them back. "
        user = f"Chat message to reply to: {text}\nFACTS: {facts_line}\n{rule}Keep it under {self.words_max} words.\nYour reply:"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            out = self.llm.generate(input_ids=ids, attention_mask=mask, do_sample=True, temperature=0.9, top_p=0.95, max_new_tokens=40,
                                    num_return_sequences=n, pad_token_id=self.tok.eos_token_id)
        outs = [self.tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0] for o in out]
        return [o.strip('"') for o in outs if o]

    # --- step 4b: safe replies built straight from the facts (true by construction), in your habits
    def own_reactions(self, examples, text):
        """For hype / hi / bye: pick short things YOU actually said in similar moments (found, not written).
        Only close matches, nothing crude, nothing aimed at a person, nothing private."""
        said = {self.norm(w) for w in WORD.findall(text.lower())}
        q = self.emb([text]).cpu().numpy()[0]
        close = []
        for e in examples:
            v = self.emb([e]).cpu().numpy()[0]
            if float(v @ q) >= self.args.react_match:
                close.append(e)
        out = []
        for e in close:
            low = e.lower().strip()
            w = WORD.findall(low)
            if not 1 <= len(w) <= 6 or "@" in low or re.search(r"\d|http", low) or low.endswith("?"):
                continue
            if CRUDE.search(low) or AIMED.search(low):
                continue
            if set(w) & (self.secret | self.names) or PLAN.search(low) or BOTISH.search(low):
                continue
            if w and set(self.norm(x) for x in w) <= said:
                continue  # just repeats their message back
            out.append(low)
        return list(dict.fromkeys(out))[:5]

    def safe_replies(self, known, text, examples=()):
        if isinstance(known, tuple):
            key, answer = known
            dq = {self.norm(w) for w in WORD.findall((self.facts.get(key, {}).get("question", "") + " " + key).lower())}
            asked = [self.norm(w) for w in content_words(text)]
            asked = [a for a in asked if not any(self.same_word(a, d) for d in dq)]  # "games" in "what games" isn't specific
            clauses = [c.strip() for c in answer.split(";") if c.strip()]
            parts = [p.strip() for c in clauses for p in c.split(",") if len(p.strip()) >= 3]
            # the part of the drawer that matches what was asked ("do you like burritos?" -> "i eat burritos")
            hit = [p for p in parts if any(self.same_word(a, self.norm(w)) for a in asked for w in content_words(p))]
            dq = {self.norm(w) for w in WORD.findall((self.facts.get(key, {}).get("question", "") + " " + key).lower())}
            specific = [a for a in asked if a not in GENERIC_Q and not any(self.same_word(a, d) for d in dq)]
            # things this drawer covers ("mouse, keyboard, headset") - if they ask for one your facts don't name: hedge
            items = [self.norm(w) for w in content_words(self.facts.get(key, {}).get("about", ""))]
            asks_item = [a for a in specific if any(self.same_word(a, i) for i in items)]
            lexical = bool(hit)
            if not hit and specific and len(parts) > 1:  # "what MOUSE do you use" -> closest part ("logitech")
                v = self.emb([text] + parts).cpu().numpy()
                sims = v[1:] @ v[0]
                if sims.max() >= 0.2 and sims.max() - sorted(sims)[-2] >= 0.03:
                    hit = [parts[int(sims.argmax())]]
            pick = hit[0] if hit else clauses[0]
            if not re.match(r"(i|im|i'm|ima|imma|ive|i've|ill|i'll|id|i'd|my|got)\b", pick) and key in FRAMES:
                pick = FRAMES[key].format(pick)
            end = next((e for e in self.enders if e not in self.openers[:1]), self.enders[0])
            if asks_item and not lexical:  # they asked about something your facts don't name ("headset")
                return [f"idk {end}, {pick}", f"idk, {pick}"]
            if YESNO.match(text.lower().strip()):
                if lexical:  # only say yeah when the fact really names what they asked about
                    return [f"yeah {pick}", f"yeah {pick} {end}"]
                return [f"idk {end}, {pick}", f"idk, {pick}"]
            out = [pick, f"{pick} {end}"]
            if key in ("games", "music", "food", "watching", "hobbies") and not hit:
                out.append(f"{self.openers[0]} {pick}")  # "mostly i play ..." fits a list
            return list(dict.fromkeys(out))
        if known == "unknown":
            rng = random.Random(hash(text))
            own = rng.sample(self.unsure_lines, min(3, len(self.unsure_lines)))
            return own + [f"idk {self.enders[0]}", f"not sure {self.enders[0]}"]
        return self.own_reactions(examples, text)

    # --- step 5a: force your style rules (numbers come from your own messages)
    def polish(self, c):
        c = c.strip().strip('"').strip()
        if self.emoji_rate < 0.02:
            c = EMOJI.sub("", c)
        if self.tag_rate < 0.01:
            c = re.sub(r"#\w+", "", c)
        c = JUNK.sub("", c)
        if self.lower_rate > 0.7:
            c = c.lower()
        if len(c.split()) > self.words_max:  # too long: keep only the first sentence
            c = re.split(r"(?<=[.!?])\s", c)[0]
        return re.sub(r"\s+", " ", c).strip()

    # --- step 5b: truth check with the NLI decision model (premise -> does it support the reply?)
    def entail(self, premise, hyp):
        torch = self.torch
        logits = self.nli._logits([(premise, hyp)])[0]
        return torch.softmax(logits, dim=0)[self.nli.ent].item()

    @staticmethod
    def same_word(a, b):
        """valo ~ valorant, rdr ~ rdr2: one is the start of the other (3+ letters)."""
        if a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))):
            return True

        for group in SYNONYMS:
            if a in group and b in group:
                return True

        def stem(w):  # hiking -> hik, hike -> hik, pictures -> pictur
            for suf in ("ing", "ed", "es", "s"):
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    w = w[:-len(suf)]
                    break
            return w.rstrip("e")
        if min(len(a), len(b)) >= 4 and stem(a) == stem(b):
            return True
        from difflib import SequenceMatcher  # typos: shity ~ shitty
        return min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= 0.8

    def norm(self, w):
        """i'm -> i, valo's -> valo, clutchhhh -> clutch, lookin -> looking"""
        w = re.sub(r"(.)\1{2,}", r"\1", w.split("'")[0])
        if w.endswith("in") and w + "g" in self.common:
            w += "g"
        return w

    def grounded(self, cand, fact, incoming):
        """Lexical truth check: every unusual word must come from your facts or the message you reply to.
        Returns (ok, why)."""
        words = [self.norm(w) for w in WORD.findall(cand.lower())]
        fact_words = [w for w in WORD.findall(fact[1].lower()) if len(w) >= 2] if isinstance(fact, tuple) else []
        said = {self.norm(w) for w in WORD.findall(incoming.lower())}
        safe = self.common | SLANG | said
        if not isinstance(fact, tuple):
            safe = safe | self.mine  # no facts needed: your own usual words are fine
        rare = [w for w in words if len(w) > 1 and not w.isdigit() and w not in safe
                and not any(self.same_word(w, f) for f in fact_words)]
        if rare:
            return False, "not in facts: " + ",".join(rare[:3])
        if isinstance(fact, tuple):
            key = [f for f in fact_words if f not in self.common] or \
                  [f for f in fact_words if f not in STOP] or fact_words  # the distinctive words
            hits = [w for w in words if any(self.same_word(w, f) for f in key)]
            if not hits:
                return False, "doesn't use the facts"
            return True, "uses: " + ",".join(dict.fromkeys(hits[:3]))
        return True, "no new names"

    def truth(self, cand, fact, incoming):
        """(ok, why). fact is (key, answer), 'unknown' or 'none'."""
        ok, why = self.grounded(cand, fact, incoming)
        if not ok:
            return ok, why
        if isinstance(fact, tuple):  # words check passed; now: does the reply contradict the facts?
            logits = self.nli._logits([(f"About me. My {fact[0]}: {fact[1]}.", cand)])[0]
            pr = self.torch.softmax(logits, dim=0)
            contra = pr[self.nli.neg].item()
            if contra >= 0.5 and contra > pr[self.nli.ent].item():
                return False, f"contradicts facts {contra:.2f}"
            return ok, why
        names = self.entail(cand, NAMES_THING)
        if fact == "unknown":
            if names >= 0.5:
                return False, f"names a thing {names:.2f}"
            unsure = self.entail(cand, NOT_SURE)
            return unsure >= 0.5, f"says not sure {unsure:.2f}"
        return True, why

    # --- step 5c: judge style
    def judge(self, cands, examples, fact, incoming, allow=()):
        if not cands:
            return []
        c = self.emb(cands).cpu().numpy()
        e = self.emb(examples).cpu().numpy()
        sims = (c @ e.T).mean(1)
        grams = set()
        for x in examples + [incoming]:
            w = x.lower().split()
            grams |= {tuple(w[i:i + 4]) for i in range(len(w) - 3)}
        scored = []
        for cand, s in zip(cands, sims):
            w = cand.lower().split()
            if cand in allow:
                pass  # built from your facts / your own words on purpose
            elif not cand or cand.lower() in self.seen or any(tuple(w[i:i + 4]) in grams for i in range(len(w) - 3)):
                scored.append((0.0, "copied", cand)); continue
            real = [x for x in WORD.findall(cand.lower()) if len(x) > 1]
            if (len(real) < 2 and cand not in allow) or "__" in cand:
                scored.append((0.0, "too short / junk", cand)); continue
            if len([x for x in w if not x.startswith("@")]) > self.words_max + 3:
                scored.append((0.0, "too long", cand)); continue
            if BOTISH.search(cand.lower()):
                scored.append((0.0, "sounds like a bot", cand)); continue
            if PLAN.search(cand.lower()):
                scored.append((0.0, "claims a plan", cand)); continue
            leak = set(WORD.findall(cand.lower())) & self.secret
            if leak or (fact == "unknown" and re.search(r"\d", cand)):
                scored.append((0.0, "touches private" if leak else "invents a number", cand)); continue
            ok, why = (True, "built") if cand in allow and isinstance(fact, tuple) else self.truth(cand, fact, incoming)
            if not ok:
                scored.append((0.0, f"FAIL {why}", cand)); continue
            tag = "built from facts" if cand in allow and isinstance(fact, tuple) else \
                  "your own words" if cand in allow else why
            scored.append((self.style(cand.lower()) + float(s), f"ok ({tag})", cand))
        return sorted(scored, key=lambda x: -x[0])

    def handle(self, line):
        who, _, text = line.partition(":") if ":" in line[:30] else ("viewer", "", line)
        text = text.strip() or who
        d = self.decide(text)
        kind, fact = d["kind"], d["fact"]
        print(f"\n  1. decide : kind = {kind['choice']:<10} confidence {kind['confidence']:.2f}"
              f"   drawer = {dither(' ' + fact['choice'] + ' ', 0, 0.0, 0.5)} "
              f"({fact['confidence']:.2f}, {fact['agree']} readings agree)")
        low = text.lower().strip(" !.")
        words = WORD.findall(low)
        if kind["choice"] not in NO_REPLY and words and len(words) <= 3:
            if GREET.match(low):
                kind = dict(kind, choice="greeting", confidence=max(kind["confidence"], 0.5))  # hi chat, bye chat
            elif self.norm(words[0]) in HYPE_START or low in SLANG:
                kind = dict(kind, choice="hype", confidence=max(kind["confidence"], 0.5))  # gg, W stream, LMAOOO
        if QUESTION.search(text.lower().strip()) and kind["choice"] not in NO_REPLY | {"question"}:
            kind = dict(kind, choice="question")  # shaped like a question: treat it as one
        need = self.args.safe_threshold if kind["choice"] in SAFE else self.args.threshold
        if kind["choice"] == "question" and fact["choice"] != "none" and fact["confidence"] >= 0.3:
            need = min(need, 0.2)  # the drawer is clear, so a weak 'is it a question' score is fine
        if kind["confidence"] < need:
            print(dim(f"  2. gate   : NOT SURE (needs {need:.2f}) -> no reply, leave it for a human"))
            return {"kind": kind["choice"], "drawer": fact["choice"], "reply": None, "why": "not sure"}
        if kind["choice"] in NO_REPLY:
            print(f"  2. gate   : {dither(' ' + kind['choice'] + ' ', 2, 0.0, 0.3)} -> no reply")
            return {"kind": kind["choice"], "drawer": fact["choice"], "reply": None, "why": kind["choice"]}
        answer = self.facts.get(fact["choice"], {}).get("answer", "")
        private_q = bool(PRIVATE_Q.search(text.lower()))
        if private_q:
            print(dim("  2. gate   : private question -> no facts, only 'not sure'"))
        if kind["choice"] == "question" and fact["choice"] != "none" and fact["action"] == "act" \
                and "EDIT ME" not in answer and not private_q:
            facts_line, known = f"{fact['choice']}: {answer}", (fact["choice"], answer)
        elif kind["choice"] == "question":
            facts_line, known = "unknown (not in your facts list)", "unknown"
        else:
            facts_line, known = "none needed, just react naturally", "none"
        print(f"  2. gate   : ok -> reply.  FACTS = {dither(' ' + facts_line + ' ', 1)}")
        ex, _ = self.examples(text)
        print(dim(f"  3. voice  : using {len(ex)} of your past messages, e.g. \"{ex[0][:60]}\""))
        cands = [self.polish(c) for c in self.write(f"{who.strip()}: {text}", facts_line, ex)]
        safe = self.safe_replies(known, text, ex)
        scored = self.judge(list(dict.fromkeys(cands + safe)), ex, known, text, allow=set(safe))
        print("  4. " + ("write + " if self.llm else "pick (no generation) + ") + "5. check (score = sounds like you):")
        for sc, verdict, c in scored:
            if verdict.startswith("ok"):
                print(f"       {bar(min(1.0, sc / 3), 6)} {sc:5.2f}  {rgb(BLUES[3])}{verdict:<26}{reset()} {c}")
            else:
                print(dim(f"       {'':6} {sc:5.2f}  {verdict:<26} {c}"))
        good = [x for x in scored if x[1].startswith("ok")]
        used = known[0] if isinstance(known, tuple) else known
        if good:
            print(f"  >> REPLY: {dither(' ' + good[0][2] + ' ', 0)}")
            return {"kind": kind["choice"], "drawer": used, "reply": good[0][2], "why": good[0][1]}
        print(dim("  >> no safe reply -> stay quiet"))
        return {"kind": kind["choice"], "drawer": used, "reply": None, "why": "no safe reply"}


def private_words():
    """DISTINCTIVE words of your locked private facts: words that never appear in your public drawers or in
    what you typed. Everyday words like "play" or "angeles" are never blocked just because a private fact used them."""
    import glob
    priv, pub = set(), set()
    for p in glob.glob(os.path.join(CABINET, "_private", "*.json")):
        with open(p, encoding="utf-8") as f:
            for r in json.load(f):
                priv |= {w for w in WORD.findall(r["fact"].lower()) if len(w) > 3}
    for p in glob.glob(os.path.join(CABINET, "*.json")):
        if os.path.basename(p).startswith("_"):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        pub |= set(WORD.findall((d.get("typed_by_you", "") + " " + d.get("question", "") + " " + d.get("about", "") + " "
                                 + " ".join(r["fact"] for r in d.get("facts", []))).lower()))
    return priv - pub - STOP - SLANG


def run_tests(bot, path):
    """Each line:  viewerN: message | expected
    expected = a drawer name (games, food...), unknown (should admit it doesn't know),
               react (hype/greeting: any reply, no facts needed), silent (spam/harassment: no reply),
               private (must never leak a locked private fact)."""
    import time
    secret = private_words()
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    t0 = time.time()
    for line in lines:
        msg, _, want = line.partition("|")
        msg, want = msg.strip(), want.strip().lower()
        print(f"\n{msg}")
        r = bot.handle(msg)
        reply = r["reply"]
        if want == "silent":
            ok = reply is None
        elif want == "private":
            ok = reply is None or not (set(WORD.findall(reply.lower())) & secret)
        elif want == "react":
            ok = reply is not None and r["kind"] not in NO_REPLY
        elif want == "unknown":
            ok = reply is None or r["drawer"] == "unknown" or reply.lower().startswith(("idk", "not sure", "dunno"))
        else:
            ok = reply is not None and r["drawer"] == want
        rows.append((msg, want, r, ok))
    secs = time.time() - t0
    print("\n" + "=" * 100)
    print(f"{'':2} {'message':44} {'expected':9} {'got':10} reply")
    print("-" * 100)
    for msg, want, r, ok in rows:
        got = r["drawer"] if r["reply"] else f"({r['why']})"
        reply = (r['reply'] or '-')[:40]
        print(f"{'✅' if ok else '❌'} {msg[:44]:44} {want:9} {str(got)[:10]:10} "
              f"{dither(' ' + reply + ' ', len(got)) if r['reply'] else dim(reply)}")
    by = Counter(w for _, w, _, _ in rows)
    good = Counter(w for _, w, _, ok in rows if ok)
    groups = {"drawer questions": [w for w in by if w not in ("unknown", "react", "silent", "private")],
              "unknown questions": ["unknown"], "react (hype/hi)": ["react"],
              "silent (spam/hate)": ["silent"], "private probes": ["private"]}
    print("-" * 100)
    for name, ws in groups.items():
        n = sum(by[w] for w in ws)
        if n:
            g = sum(good[w] for w in ws)
            print(f"   {name:20} {g:>3}/{n:<3} passed  {bar(g / n, 10)}")
    hedged = [m for m, w, r, ok in rows if ok and w not in ("unknown", "react", "silent", "private")
              and (r["reply"] or "").lower().startswith("idk")]
    if hedged:
        print(dim(f"   {len(hedged)} drawer answers started with 'idk' (right drawer, but sounded unsure): "
                  + ", ".join(h.split(":")[0] for h in hedged)))
    total = sum(1 for *_, ok in rows if ok)
    print(f"   {dither(' TOTAL ', 0, 0.0, 0.5)}{'':13} {total:>3}/{len(rows):<3} passed  {bar(total / len(rows), 10)}"
          f"  ({secs / max(1, len(rows)):.1f}s per message)")


def cmd_chat(args):
    bot = Bot(args)
    if args.file:
        run_tests(bot, args.file)
        return
    if args.demo:
        for line in DEMO:
            print(f"\n{line}")
            bot.handle(line)
        return
    print("\nType a chat line like  viewer1: what games do you play?   (type quit to leave)")
    while True:
        try:
            line = input("\nchat> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if line.lower() in ("quit", "exit", "q"):
            break
        if line.startswith(("python ", "cd ", "ls", "nano ")):
            print(dim("  (that looks like a terminal command - type quit to leave the chat first)"))
            continue
        if line:
            bot.handle(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("csv")
    p.add_argument("--main-only", action="store_true", help="only your most used account")
    p = sub.add_parser("index"); p.add_argument("--device", default="auto")
    p = sub.add_parser("questions"); p.add_argument("--n", type=int, default=3000)
    p = sub.add_parser("chat")
    p.add_argument("--demo", action="store_true", help="run 7 sample chat lines instead of typing")
    p.add_argument("--file", help="run a test file (lines: viewerN: message | expected) and grade it")
    p.add_argument("--backends", default="nli", help="nli, mapika, student:student_twitch, or student:student_twitch,mapika with --cascade")
    p.add_argument("--stasis", action="store_true", help="ask each question several ways before trusting it")
    p.add_argument("--cascade", action="store_true", help="fast student first, Mapika only for unsure ones")
    p.add_argument("--threshold", type=float, default=0.5, help="confidence needed to answer a question (default 0.5)")
    p.add_argument("--safe-threshold", type=float, default=0.35, help="for hype/greeting/personal (default 0.35)")
    p.add_argument("--react-match", type=float, default=0.45,
                   help="how close one of your past messages must be to reuse it as a reaction")
    p.add_argument("--drawer-threshold", type=float, default=0.3, help="confidence needed to use a drawer")
    p.add_argument("--writer", default="none",
                   help="none (default: no generation) or an LLM to also write candidates, e.g. " + WRITER)
    p.add_argument("--role", default="viewer", choices=["viewer", "streamer"], help="who the bot speaks as")
    p.add_argument("--english", default="english.txt", help="normal English to compare your style against")
    p.add_argument("--common", type=int, default=30000, help="how many common English words count as safe")
    p.add_argument("--device", default="auto")
    args = ap.parse_args()
    {"prep": cmd_prep, "index": cmd_index, "questions": cmd_questions, "chat": cmd_chat}[args.cmd](args)


if __name__ == "__main__":
    main()
