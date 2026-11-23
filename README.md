# localjev

A local, Jev-style decision engine with probability stasis. It reads a message, answers typed
questions about it (yes/no, choice, score) with probabilities and a confidence, and says whether
it is sure enough to act or should escalate to a human. It never writes text, it only picks.

See [FINDINGS.md](FINDINGS.md) for results.

## Files

- `localjev.py`: the engine (run, test, calibrate, stream, serve, distill)
- `distill.py`: teacher → tiny student
- `compare.py`: grades every setup on the same questions, writes `results.md`
- `tests/`: 20 labeled test messages (44 questions)
- `stream/`: example message stream for probability stasis over time

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install torch transformers sentencepiece protobuf
python localjev.py test tests/
```

Optional models: `pip install strands-decider` (Amazon) and `pip install decider-ai` (Mapika).

## Main commands

```bash
python localjev.py run tests/07_fight.json --stasis
python localjev.py test tests/ --backends mapika
python localjev.py distill --backends mapika
python localjev.py test tests/ --backends student:student_qwen,mapika --cascade --first-threshold 0.7
python localjev.py stream stream/questions.json stream/messages.jsonl --stasis
python localjev.py serve --stasis
```

## Used by Winter Assistant

Winter reads the fact cabinet (`twitch/facts/*.json`) as its 4th result, shown in red-orange
(`retrieval/cabinet_system.py` in winter-assistant). It never opens `_private/` or `_unsure.json`
and skips "used to" facts.

Picking a drawer works like a Jev `choice` question: each drawer is an option (its `question` +
`about` lines), each gets a score, and Winter only shows it above a bar, otherwise it steps aside.
Winter scores with its Qwen3 embedder + shared words instead of the NLI model, to keep the GPU free.
The drawers already fit the `choice` format, so the real Jev engine can be plugged in later.
