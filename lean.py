"""
lean.py - a lean decision model, built the Blow / Davis way: do the least work that gives a
usable answer, keep everything in plain arrays, and make every line readable.

THE WHOLE MODEL
    text  -> lowercase words -> hash every word, word pair and 3-5 letter chunk into one of 2^18 slots
    vector = average of those slots' rows in one table E          (no tokenizer, no attention)
    message, question and each option become vectors this way     (the message is read ONCE)
    score(option) = small 2-layer net on [m, o, m*o, q*o, m*q]   -> softmax over the options
    self_check    = small 2-layer net, trained to predict "right AND stable when reworded"
    familiarity   = share of the message's words seen in training -> unfamiliar messages lose confidence

TEACHING IT ENGLISH (fastText-style, same table E)
    read plain text; for each word, its slot-average must predict the words around it.
    words used the same way end up with similar vectors ("fainted" ~ "collapsed"),
    and letter chunks give typos and unseen words a sensible vector too.

COMMANDS (through localjev.py)
    python localjev.py get-english                 download Wikipedia text (wikitext-103) once
    python localjev.py train-english               fill table E with English (GPU)
    python localjev.py train-lean                  learn decisions + stasis on top (GPU)
    python localjev.py test tests/ --backends lean run it (numpy only, no torch needed)

Training uses torch on the GPU. Running the model uses only numpy.
"""
import json
import math
import os
import random
import re
import sys
import time
import zlib

NB = 1 << 18                # number of hash slots (262,144)
WORD = re.compile(r"[a-z0-9']+")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- features: hashing, no tokenizer

def slot(s):
    return zlib.crc32(s.encode("utf-8")) & (NB - 1)


_word_cache = {}


def word_slots(w):
    """A word's slots: the word itself plus its 3-5 letter chunks (with < > marking the ends)."""
    got = _word_cache.get(w)
    if got is None:
        ww = f"<{w}>"
        got = [slot("w:" + w)] + [slot("c:" + ww[i:i + n]) for n in (3, 4, 5) for i in range(len(ww) - n + 1)]
        if len(_word_cache) < 2_000_000:
            _word_cache[w] = got
    return got


def words(text):
    return WORD.findall(text.lower())


def text_slots(text):
    """A text's slots: every word's slots plus every word pair (catches 'not cancelling')."""
    ws = words(text)
    out = []
    for w in ws:
        out += word_slots(w)
    out += [slot(f"b:{a}_{b}") for a, b in zip(ws, ws[1:])]
    return out or [slot("w:<empty>")]


# ---------------------------------------------------------------- 1. get English text

def cmd_get_english(args):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("needs: pip install datasets")
    log("downloading wikitext-103 (Wikipedia articles, about 100M words)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            line = row["text"].strip()
            if line and not line.startswith("="):
                f.write(line + "\n")
                n += len(line)
                if n > args.max_mb * 1_000_000:
                    break
    log(f"saved {n / 1e6:.0f} MB of text to {args.out}")


# ---------------------------------------------------------------- 2. teach table E English (GPU)

def cmd_train_english(args):
    import torch
    from collections import Counter
    from torch import nn

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    if not os.path.exists(args.text):
        sys.exit(f"no text at {args.text}. Run: python localjev.py get-english")

    # pass 1: count words (plus your own messages from distill's cache, so venue words are included)
    import numpy as np
    from array import array
    log("reading text...")

    def lines():
        with open(args.text, encoding="utf-8") as f:
            for line in f:
                yield line
        if os.path.exists(args.cache):
            seen = set()
            with open(args.cache, encoding="utf-8") as f:
                for line in f:
                    m = json.loads(line)["m"]
                    if m not in seen:
                        seen.add(m)
                        yield m

    counts, total = Counter(), 0
    for line in lines():
        ws = words(line)
        counts.update(ws)
        total += len(ws)
        if total >= args.words:
            break
    vocab = [w for w, c in counts.most_common(args.vocab) if c >= args.min_count]
    index = {w: i for i, w in enumerate(vocab)}
    # pass 2: the text as one array of word ids (compact: 4 bytes per word)
    buf, total = array("i"), 0
    for line in lines():
        ws = words(line)
        buf.extend(index[w] for w in ws if w in index)
        total += len(ws)
        if total >= args.words:
            break
    ids = torch.from_numpy(np.frombuffer(buf, dtype=np.int32).copy()).to(device)
    V, N = len(vocab), len(ids)
    log(f"{N / 1e6:.1f}M words, vocabulary {V:,}, training on {device}")

    # each vocab word's slots, packed end to end (no padding): flat list + where each word starts
    rows = [word_slots(w) for w in vocab]
    lens = torch.tensor([len(r) for r in rows], device=device)
    starts = torch.cumsum(lens, 0) - lens
    flat = torch.tensor([x for r in rows for x in r], dtype=torch.long, device=device)

    freq = torch.tensor([counts[w] for w in vocab], dtype=torch.float, device=device)
    keep = ((1e-4 * N / freq).sqrt() + 1e-4 * N / freq).clamp(max=1.0)   # drop very common words sometimes
    neg_table = torch.multinomial(freq ** 0.75, 5_000_000, replacement=True)  # negatives ~ freq^0.75

    E = nn.Embedding(NB, args.dim, sparse=True).to(device)          # the table the lean model will use
    nn.init.uniform_(E.weight, -0.5 / args.dim, 0.5 / args.dim)
    O = nn.Embedding(V, args.dim, sparse=True).to(device)           # context vectors (thrown away after)
    nn.init.zeros_(O.weight)
    opt = torch.optim.SparseAdam(list(E.parameters()) + list(O.parameters()), lr=args.lr)

    def word_vecs(w):  # average of each word's slot rows, reading only real slots
        n = lens[w]
        offs = torch.cumsum(n, 0) - n
        bag = torch.repeat_interleave(torch.arange(len(w), device=device), n)
        idx = flat[starts[w][bag] + torch.arange(len(bag), device=device) - offs[bag]]
        return nn.functional.embedding_bag(idx, E.weight, offs, mode="mean", sparse=True)

    steps = int(args.epochs * N / args.batch)
    t0 = time.time()
    for step in range(steps):
        pos = torch.randint(args.window, N - args.window, (args.batch,), device=device)
        center = ids[pos].long()
        ok = torch.rand(args.batch, device=device) < keep[center]
        pos, center = pos[ok], center[ok]
        off = torch.randint(1, args.window + 1, pos.shape, device=device) * \
            (torch.randint(0, 2, pos.shape, device=device) * 2 - 1)
        ctx = ids[pos + off].long()
        neg = neg_table[torch.randint(0, len(neg_table), (len(center), args.neg), device=device)]
        u = word_vecs(center)                                          # [B, d]
        pos_score = (u * O(ctx)).sum(1)
        neg_score = torch.bmm(O(neg), u.unsqueeze(2)).squeeze(2)
        loss = -(nn.functional.logsigmoid(pos_score).mean() + nn.functional.logsigmoid(-neg_score).sum(1).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 2000 == 0 or step == steps - 1:
            el = time.time() - t0
            log(f"  step {step}/{steps}  loss {loss.item():.3f}  ({el:.0f}s, ~{el / max(step, 1) * (steps - step):.0f}s left)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"E": E.weight.detach().half().cpu(), "dim": args.dim}, args.out)

    # show it learned meaning: nearest words
    with torch.no_grad():
        W = torch.cat([word_vecs(torch.arange(i, min(i + 4096, V), device=device)) for i in range(0, V, 4096)])
        W = nn.functional.normalize(W, dim=1)
        print(f"\nsaved English table to {args.out}. Nearest words (shows it learned meaning):")
        for probe in ["fainted", "fight", "refund", "spill", "angry", "crowd", "slippery", "ticket"]:
            if probe in index:
                sims = W @ W[index[probe]]
                top = [vocab[i] for i in sims.topk(7).indices.tolist() if vocab[i] != probe][:5]
                print(f"  {probe:<10} -> {', '.join(top)}")


# ---------------------------------------------------------------- 3. the lean decision model

def build_lean(dim, hidden):
    import torch
    from torch import nn

    class Lean(nn.Module):
        def __init__(self):
            super().__init__()
            self.E = nn.EmbeddingBag(NB, dim, mode="mean", sparse=True)
            self.s1, self.s2 = nn.Linear(5 * dim, hidden), nn.Linear(hidden, 1)   # option scorer
            self.c1, self.c2 = nn.Linear(3 * dim, hidden), nn.Linear(hidden, 1)   # self-check

        def encode(self, slot_lists, device):
            flat = torch.tensor([s for sl in slot_lists for s in sl], dtype=torch.long, device=device)
            offs = torch.tensor([0] + [len(sl) for sl in slot_lists[:-1]], device=device).cumsum(0)
            return self.E(flat, offs)

        def forward(self, m, q, o, g_idx, o_idx, G, K):
            mg, qg = m[g_idx], q[g_idx]
            x = torch.cat([mg, o, mg * o, qg * o, mg * qg], 1)
            logits = self.s2(torch.relu(self.s1(x))).squeeze(1)
            grid = torch.full((G, K), float("-inf"), device=m.device)
            grid[g_idx, o_idx] = logits
            p = grid.softmax(1)
            pooled = torch.zeros(G, o.shape[1], device=m.device).index_add_(0, g_idx, o * p[g_idx, o_idx].unsqueeze(1))
            check = self.c2(torch.relu(self.c1(torch.cat([m, q, pooled], 1)))).squeeze(1)
            return grid, check

    return Lean()


def cmd_train_lean(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from stasisnet import load_cache, option_texts, perturb_message, reword

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    if not os.path.exists(args.cache):
        sys.exit(f"no teacher answers at {args.cache}. Run: python localjev.py distill --backends mapika")
    rows = load_cache(args.cache)
    rng = random.Random(args.seed)
    msgs = sorted({m for m, _, _ in rows})
    rng.shuffle(msgs)
    held = set(msgs[:max(1, len(msgs) // 10)])
    train = [r for r in rows if r[0] not in held]
    val = [r for r in rows if r[0] in held]
    log(f"{len(rows)} teacher answers, {len(msgs)} messages ({len(held)} held out)")

    model = build_lean(args.dim, args.hidden).to(device)
    if args.english and os.path.exists(args.english):
        saved = torch.load(args.english)
        if saved["dim"] != args.dim:
            sys.exit(f"English table has dim {saved['dim']}, use --dim {saved['dim']}")
        model.E.weight.data.copy_(saved["E"].float())
        log(f"starting from the English table in {args.english}")
    else:
        log("no English table, starting from random (run train-english first for better results)")

    dense = [p for n, p in model.named_parameters() if not n.startswith("E.")]
    opt_e = torch.optim.SparseAdam(model.E.parameters(), lr=args.lr_table)
    opt_d = torch.optim.AdamW(dense, lr=args.lr, weight_decay=0.01)
    slot_cache = {}

    def slots_of(t):
        s = slot_cache.get(t)
        if s is None:
            s = slot_cache[t] = text_slots(t)
        return s

    def run(batch, variant):
        mtexts, qtexts, otexts, sizes = [], [], [], []
        for m, q, _ in batch:
            if variant:
                q = reword(q, rng)
                m = perturb_message(m, rng)
            mtexts.append(m)
            qtexts.append(q["instructions"])
            ops = option_texts(q)
            otexts += ops
            sizes.append(len(ops))
        # encode every distinct string once (the message is read once, not once per option)
        uniq = list(dict.fromkeys(mtexts + qtexts + otexts))
        where = {t: i for i, t in enumerate(uniq)}
        vec = model.encode([slots_of(t) for t in uniq], device)
        m = vec[[where[t] for t in mtexts]]
        q = vec[[where[t] for t in qtexts]]
        o = vec[[where[t] for t in otexts]]
        n = torch.tensor(sizes, device=device)
        g_idx = torch.repeat_interleave(torch.arange(len(sizes), device=device), n)
        o_idx = torch.arange(len(g_idx), device=device) - (torch.cumsum(n, 0) - n)[g_idx]
        return model(m, q, o, g_idx, o_idx, len(sizes), int(n.max()))

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train)
        tot = 0.0
        nb = math.ceil(len(train) / args.batch)
        for b in range(0, len(train), args.batch):
            br = train[b:b + args.batch]
            K = max(len(d) for _, _, d in br)
            T = torch.zeros(len(br), K, device=device)
            for i, (_, _, d) in enumerate(br):
                T[i, :len(d)] = torch.tensor(d)
            grid, chk = run(br, False)
            grid2, _ = run(br, True)
            logp, logp2 = F.log_softmax(grid, 1), F.log_softmax(grid2, 1)
            safe = lambda x: x.masked_fill(torch.isinf(x), 0.0)
            match = -(T * safe(logp)).sum(1).mean() - (T * safe(logp2)).sum(1).mean()
            p, p2 = logp.exp(), logp2.exp()
            consist = 0.5 * (p - p2).abs().sum(1).mean()
            with torch.no_grad():
                target = (p.argmax(1) == T.argmax(1)).float() * (1 - 0.5 * (p - p2).abs().sum(1))
            loss = match + args.consistency * consist + F.binary_cross_entropy_with_logits(chk, target)
            opt_e.zero_grad()
            opt_d.zero_grad()
            loss.backward()
            opt_e.step()
            opt_d.step()
            tot += loss.item()
        log(f"  epoch {ep + 1}/{args.epochs}  loss {tot / nb:.3f}  ({time.time() - t0:.0f}s)")

    # held-out check
    model.eval()
    agree = stay = 0
    checks, rights = [], []
    with torch.no_grad():
        for b in range(0, len(val), 512):
            br = val[b:b + 512]
            grid, chk = run(br, False)
            grid2, _ = run(br, True)
            p, p2 = grid.softmax(1), grid2.softmax(1)
            for i, (_, _, d) in enumerate(br):
                a = int(p[i].argmax()) == max(range(len(d)), key=d.__getitem__)
                agree += a
                stay += int(p[i].argmax()) == int(p2[i].argmax())
                checks.append(torch.sigmoid(chk[i]).item())
                rights.append(a)

    # save as plain arrays: numpy is all that's needed to run it
    os.makedirs(args.out, exist_ok=True)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(os.path.join(args.out, "lean.npz"), E=sd["E.weight"].astype(np.float16),
                        s1w=sd["s1.weight"], s1b=sd["s1.bias"], s2w=sd["s2.weight"], s2b=sd["s2.bias"],
                        c1w=sd["c1.weight"], c1b=sd["c1.bias"], c2w=sd["c2.weight"], c2b=sd["c2.bias"])
    train_words = sorted({w for m, _, _ in train for w in words(m)})
    with open(os.path.join(args.out, "lean.json"), "w") as f:
        json.dump({"dim": args.dim, "hidden": args.hidden, "english": bool(args.english and os.path.exists(args.english)),
                   "train_words": train_words}, f)

    n = len(val)
    hi = [r for c, r in zip(checks, rights) if c >= 0.7]
    lo = [r for c, r in zip(checks, rights) if c < 0.7]
    size = os.path.getsize(os.path.join(args.out, "lean.npz")) / 1e6
    print(f"\nlean model saved to {args.out}/ ({size:.0f} MB, runs on numpy only)")
    print(f"held-out check ({n} answers on messages it never trained on):")
    print(f"  agrees with teacher        {agree / n:.0%}")
    print(f"  same answer when reworded  {stay / n:.0%}   <- built-in stasis")
    if hi:
        print(f"  when self-check >= 0.7     {sum(hi) / len(hi):.0%} right  ({len(hi)} answers)")
    if lo:
        print(f"  when self-check <  0.7     {sum(lo) / len(lo):.0%} right  ({len(lo)} answers)")


# ---------------------------------------------------------------- 4. running it: numpy only

class LeanBackend:
    frames = ["{}", "Based only on the message: {}", "Question: {}"]

    def __init__(self, folder, device=None):
        import numpy as np
        import localjev as L
        self.np, self.L = np, L
        z = np.load(os.path.join(folder, "lean.npz"))
        self.E = z["E"]
        self.w = {k: z[k].astype(np.float32) for k in z.files if k != "E"}
        with open(os.path.join(folder, "lean.json")) as f:
            meta = json.load(f)
        self.known = set(meta["train_words"])
        base = os.path.basename(os.path.normpath(folder))
        self.name = "lean" if base == "lean" else f"lean-{base}"
        self.model_name = f"lean ({'English-pretrained' if meta['english'] else 'no English'}, {base})"
        self.vec_cache = {}

    def vec(self, text):
        v = self.vec_cache.get(text)
        if v is None:
            v = self.E[text_slots(text)].astype(self.np.float32).mean(0)
            if len(self.vec_cache) < 100_000:
                self.vec_cache[text] = v
        return v

    def familiarity(self, text):
        ws = words(text)
        return sum(w in self.known for w in ws) / len(ws) if ws else 0.0

    def distributions(self, states, items):
        from stasisnet import option_texts
        np, w = self.np, self.w
        relu = lambda x: np.maximum(x, 0)
        out = []
        for si, q, fr in items:
            text = self.L.state_text(states[si])
            m = self.vec(text)
            qv = self.vec(q["instructions"] if fr == "{}" else fr.format(q["instructions"]))
            o = np.stack([self.vec(t) for t in option_texts(q)])
            x = np.concatenate([np.broadcast_to(m, o.shape), o, m * o, qv * o, np.broadcast_to(m * qv, o.shape)], 1)
            logits = (relu(x @ w["s1w"].T + w["s1b"]) @ w["s2w"].T + w["s2b"])[:, 0]
            p = np.exp(logits - logits.max())
            p /= p.sum()
            pooled = (o * p[:, None]).sum(0)
            c = (relu(np.concatenate([m, qv, pooled]) @ w["c1w"].T + w["c1b"]) @ w["c2w"].T + w["c2b"])[0]
            check = 1 / (1 + math.exp(-c))
            d = dict(zip(self.L.outcome_keys(q), p.tolist()))
            d["__stasis__"] = check * (0.5 + 0.5 * self.familiarity(text))  # unfamiliar -> less sure
            out.append(d)
        return out
