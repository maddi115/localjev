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
