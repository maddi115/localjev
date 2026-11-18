"""
stasisnet.py - a tiny decision model trained FROM SCRATCH, with probability stasis built in.

No pretrained weights at all: its tokenizer and every weight are learned from your data
(the teacher answers cached by `distill`).

How stasis is built in (instead of asking 5 ways at runtime):
  1. Option scoring. The model scores each (message, question, option) separately, then
     softmaxes over the options. Shuffling the options cannot change the answer: order
     invariance is part of the architecture.
  2. Consistency training. Every example is also shown in a changed form with the same meaning
     (reworded question, tone added, a typo, JSON keys shuffled). The loss punishes the model when
     its answer moves. Stability ends up in the weights.
  3. Self-check head. A second output learns to predict "my answer is right AND would survive
     rewording". At runtime it gives a stasis score in ONE pass.

Used through localjev.py:
    python localjev.py train-stasis                     train (from student/teacher_cache.jsonl)
    python localjev.py test tests/ --backends stasisnet grade it
"""
import json
import math
import os
import random
import sys
import time

FRAMES = ["{}", "Based only on the message: {}", "Question: {}", "{} (answer from the message)"]
TONES = [("", ""), ("Hey, ", ""), ("", " Please help."), ("", " Thanks!"), ("URGENT: ", ""), ("Hi, ", " Thank you.")]
MAX_MSG, MAX_Q, MAX_OPT = 112, 40, 40


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- questions -> option texts

def option_texts(q):
    """One text per answer option, in localjev's outcome_keys order."""
    t = q["type"]
    if t == "noul":
        c = q.get("criteria") or {}
        yes = "yes" + (f": {c['true']}" if c.get("true") else "")
        no = "no" + (f": {c['false']}" if c.get("false") else "")
        return [yes, no]  # outcome_keys(noul) == ["true", "false"]
    if t == "choice":
        return [f"{k}: {v}" if v else k for k, v in q["criteria"].items()]
    return [f"level {i}: {lv}" for i, lv in enumerate(q["criteria"])]


# ---------------------------------------------------------------- meaning-preserving changes

def reword(q, rng):
    fr = rng.choice(FRAMES)
    ins = q["instructions"]
    if rng.random() < 0.3:
        ins = ins.lower().rstrip("?.!")
    q2 = dict(q, instructions=fr.format(ins) if fr == "{}" else fr.format(ins[:1].lower() + ins[1:]))
    return q2


def typo(s, rng):
    if len(s) < 8:
        return s
    i = rng.randrange(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def perturb_message(m, rng):
    try:
        obj = json.loads(m) if m.startswith("{") else None
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        keys = list(obj)
        rng.shuffle(keys)
        return json.dumps({k: obj[k] for k in keys}, ensure_ascii=False)
    pre, post = rng.choice(TONES)
    out = pre + (m[:1].lower() + m[1:] if pre else m) + post
    if rng.random() < 0.3:
        out = typo(out, rng)
    return out


# ---------------------------------------------------------------- tokenizer (trained from zero)

def train_tokenizer(texts, vocab_size, path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    tok.save(path)
    return tok


class Encoder:
    """Turns text into token ids. Every distinct string is tokenized once and remembered, and new
    strings are tokenized together in one fast batch, so the GPU isn't left waiting on the CPU."""

    def __init__(self, tok):
        self.tok = tok
        self.pad, self.cls, self.sep = (tok.token_to_id(t) for t in ("[PAD]", "[CLS]", "[SEP]"))
        self.cache = {}

    def ids(self, texts):
        new = list({t for t in texts if t not in self.cache})
        if new:
            for t, e in zip(new, self.tok.encode_batch(new)):
                self.cache[t] = e.ids
            if len(self.cache) > 400_000:  # keep memory bounded on long runs
                self.cache = {t: self.cache[t] for t in new}
        return [self.cache[t] for t in texts]

    def batch(self, groups, device):
        """groups: list of (message, question_text, [option_texts]). -> ids, mask, group sizes"""
        import numpy as np
        import torch
        msgs = self.ids([g[0] for g in groups])
        ques = self.ids([g[1] for g in groups])
        opts = self.ids([o for g in groups for o in g[2]])
        seqs, sizes, k = [], [], 0
        for m, q, (_, _, os_) in zip(msgs, ques, groups):
            head = [self.cls] + m[:MAX_MSG] + [self.sep] + q[:MAX_Q] + [self.sep]
            for _ in os_:
                seqs.append(head + opts[k][:MAX_OPT])
                k += 1
            sizes.append(len(os_))
        L = max(len(x) for x in seqs)
        arr = np.full((len(seqs), L), self.pad, dtype=np.int64)
        for i, x in enumerate(seqs):
            arr[i, :len(x)] = x
        ids = torch.from_numpy(arr).to(device, non_blocking=True)
        return ids, ids != self.pad, sizes


# ---------------------------------------------------------------- the model

def build_model(vocab, d, layers, heads, maxlen):
    import torch
    from torch import nn

    class StasisNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, d)
            self.pos = nn.Embedding(maxlen, d)
            layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.score = nn.Linear(d, 1)                      # how well this option fits
            self.check = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))  # self-check

        def forward(self, ids, mask, sizes):
            pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            h = self.enc(self.tok(ids) + self.pos(pos), src_key_padding_mask=~mask)
            cls = self.norm(h[:, 0])
            logits = self.score(cls).squeeze(-1)
            # group options of the same question (softmax over its own options only), all at once
            G, K = len(sizes), max(sizes)
            n = torch.tensor(sizes, device=ids.device)
            g_idx = torch.repeat_interleave(torch.arange(G, device=ids.device), n)
            starts = torch.cumsum(n, 0) - n
            o_idx = torch.arange(len(g_idx), device=ids.device) - starts[g_idx]
            grid = torch.full((G, K), float("-inf"), device=ids.device, dtype=logits.dtype)
            grid[g_idx, o_idx] = logits
            pooled = torch.zeros(G, cls.shape[1], device=ids.device, dtype=cls.dtype)
            pooled.index_add_(0, g_idx, cls)
            pooled = pooled / n.unsqueeze(1).to(cls.dtype)
            return grid, self.check(pooled).squeeze(-1)

    return StasisNet()


# ---------------------------------------------------------------- training

def load_cache(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t, ins, crit = json.loads(r["q"])
            q = {"type": t, "instructions": ins}
            if crit is not None:
                q["criteria"] = crit
            rows.append((r["m"], q, r["d"]))
    return rows


def cmd_train(args):
    import torch
    import torch.nn.functional as F

    if not os.path.exists(args.cache):
        sys.exit(f"no teacher answers at {args.cache}. Run: python localjev.py distill --backends mapika")
    rows = load_cache(args.cache)
    rng = random.Random(args.seed)
    msgs = sorted({m for m, _, _ in rows})
    rng.shuffle(msgs)
    held = set(msgs[:max(1, len(msgs) // 10)])
    train = [r for r in rows if r[0] not in held]
    val = [r for r in rows if r[0] in held]
    log(f"{len(rows)} teacher answers, {len(msgs)} messages ({len(held)} held out for checking)")

    os.makedirs(args.out, exist_ok=True)
    texts = []
    for m, q, _ in rows:
        texts += [m, perturb_message(m, rng), q["instructions"], reword(q, rng)["instructions"]] + option_texts(q)
    tok = train_tokenizer(texts, args.vocab, os.path.join(args.out, "tokenizer.json"))
    enc = Encoder(tok)
    maxlen = MAX_MSG + MAX_Q + MAX_OPT + 4

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    model = build_model(tok.get_vocab_size(), args.d, args.layers, args.heads, maxlen).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"StasisNet: {n_params / 1e6:.1f}M weights, all starting from random, training on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    use_amp = device == "cuda"
    t0, step = time.time(), 0

    def run(batch_rows, variant):
        groups = []
        for m, q, _ in batch_rows:
            if variant:
                q2 = reword(q, rng)
                groups.append((perturb_message(m, rng), q2["instructions"], option_texts(q2)))
            else:
                groups.append((m, q["instructions"], option_texts(q)))
        ids, mask, sizes = enc.batch(groups, device)
        return model(ids, mask, sizes)

    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train)
        tot = 0.0
        for b in range(0, len(train), args.batch):
            br = train[b:b + args.batch]
            K = max(len(d) for _, _, d in br)
            T = torch.zeros(len(br), K, device=device)
            for i, (_, _, d) in enumerate(br):
                T[i, :len(d)] = torch.tensor(d)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                grid, chk = run(br, False)
                grid2, _ = run(br, True)
            grid, grid2, chk = grid.float(), grid2.float(), chk.float()
            logp, logp2 = F.log_softmax(grid, 1), F.log_softmax(grid2, 1)
            safe = lambda x: x.masked_fill(torch.isinf(x), 0.0)
            match = -(T * safe(logp)).sum(1).mean() - (T * safe(logp2)).sum(1).mean()   # copy the teacher
            p, p2 = logp.exp(), logp2.exp()
            consist = 0.5 * (p - p2).abs().sum(1).mean()                                   # don't move when reworded
            with torch.no_grad():
                right = (p.argmax(1) == T.argmax(1)).float()
                stable = 1 - 0.5 * (p - p2).abs().sum(1)
                target = right * stable                                                   # right AND stable
            check = F.binary_cross_entropy_with_logits(chk, target)
            loss = match + args.consistency * consist + check
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            step += 1
        log(f"  epoch {ep + 1}/{args.epochs}  loss {tot / math.ceil(len(train) / args.batch):.3f}  ({time.time() - t0:.0f}s)")

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    cfg = {"d": args.d, "layers": args.layers, "heads": args.heads, "maxlen": maxlen,
           "vocab": tok.get_vocab_size(), "trained_on": len(train), "weights": n_params}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- check on held-out messages
    model.eval()
    agree = stay = 0
    checks, rights = [], []
    with torch.no_grad():
        for b in range(0, len(val), 128):
            br = val[b:b + 128]
            grid, chk = run(br, False)
            grid2, _ = run(br, True)
            p, p2 = grid.float().softmax(1), grid2.float().softmax(1)
            for i, (_, _, d) in enumerate(br):
                a = int(p[i].argmax()) == max(range(len(d)), key=d.__getitem__)
                agree += a
                stay += int(p[i].argmax()) == int(p2[i].argmax())
                checks.append(torch.sigmoid(chk[i].float()).item())
                rights.append(a)
    n = len(val)
    hi = [r for c, r in zip(checks, rights) if c >= 0.7]
    lo = [r for c, r in zip(checks, rights) if c < 0.7]
    print(f"\nStasisNet trained from scratch: {n_params / 1e6:.1f}M weights, saved to {args.out}/")
    print(f"held-out check ({n} answers on messages it never trained on):")
    print(f"  agrees with teacher        {agree / n:.0%}")
    print(f"  same answer when reworded  {stay / n:.0%}   <- built-in stasis")
    if hi:
        print(f"  when self-check >= 0.7     {sum(hi) / len(hi):.0%} right  ({len(hi)} answers)")
    if lo:
        print(f"  when self-check <  0.7     {sum(lo) / len(lo):.0%} right  ({len(lo)} answers)")
    print("  (a good self-check means the first number is much higher than the second)")


# ---------------------------------------------------------------- plugging into localjev

class StasisNetBackend:
    """Answers any noul/choice/score question in one pass, plus a built-in stasis score."""
    frames = ["{}", "Based only on the message: {}", "Question: {}"]

    def __init__(self, folder, device):
        import torch
        from tokenizers import Tokenizer
        import localjev as L
        self.L, self.torch, self.device = L, torch, device
        with open(os.path.join(folder, "config.json")) as f:
            cfg = json.load(f)
        self.enc = Encoder(Tokenizer.from_file(os.path.join(folder, "tokenizer.json")))
        self.model = build_model(cfg["vocab"], cfg["d"], cfg["layers"], cfg["heads"], cfg["maxlen"])
        self.model.load_state_dict(torch.load(os.path.join(folder, "model.pt"), map_location=device))
        self.model.to(device).eval()
        base = os.path.basename(os.path.normpath(folder))
        self.name = "stasisnet" if base == "stasisnet" else f"stasisnet-{base}"
        self.model_name = f"StasisNet from scratch ({cfg['weights'] / 1e6:.1f}M, {base})"

    def distributions(self, states, items):
        groups = [(self.L.state_text(states[si]), q["instructions"] if fr == "{}" else fr.format(q["instructions"]),
                   option_texts(q)) for si, q, fr in items]
        out = []
        with self.torch.no_grad():
            for b in range(0, len(groups), 64):
                ids, mask, sizes = self.enc.batch(groups[b:b + 64], self.device)
                grid, chk = self.model(ids, mask, sizes)
                for g, n in enumerate(sizes):
                    p = grid[g, :n].float().softmax(0).tolist()
                    q = items[b + g][1]
                    d = dict(zip(self.L.outcome_keys(q), p))
                    d["__stasis__"] = float(self.torch.sigmoid(chk[g].float()))
                    out.append(d)
        return out
