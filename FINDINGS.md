# localjev: findings

What we learned building a local, Jev-style decision engine with probability stasis.
All numbers are from runs on an RTX 3050 (8 GB), November 2026.

## 1. A working Jev-style decision engine

- `localjev` uses Jev's request format and answer types (yes/no "noul", choice, score), gives a
  confidence on every answer, says `act` or `escalate`, and has a Jev-style server (`/v1/systemone`).
- It runs free models locally, with no API bill.
- The test setup can be trusted: it reproduced the published JevBench scores for Amazon's
  Strands Decider (about 71% vs their 72.3%) and Mapika's decider-2b (about 77% vs 76%).

## 2. The models each have a personality

| Model | Accuracy (275 questions) | Confident mistakes |
|---|---|---|
| DeBERTa (small) | 64% | 21 |
| Amazon (Strands Decider 2B) | 73% | 0, careful |
| Mapika (decider-2b) | 79%, smartest | 18, overconfident |

## 3. Probability stasis: what it does and doesn't do

- It does **not** make a model smarter. Accuracy stays about the same.
- It **does** make a model know when it's unsure. It cut DeBERTa's confident mistakes nearly in half (21 → 12).
- Rewording only catches answers that wobble. It can't catch a model that is consistently wrong,
  which is why it barely helped Mapika.
- Stasis across two *different* models is the strongest version. Mapika + Amazon + stasis had
  0 confident mistakes, safely handled the most questions (58 of 275), and gave the most honest probabilities.
- Stasis over time (the `stream` command) works once messages that aren't about a topic stop moving its state.
- Bonus: it catches badly worded questions. A rewording ("Someone could get hurt") that didn't
  actually mean the original ("People are fighting") was flagged by disagreement.

## 4. Wording matters as much as the model

Ask about facts in the text ("People are fighting") and let code decide the action ("send staff").
Changing only the wording took DeBERTa from 0.0002 to 0.92 on the same message.

## 5. Teacher → student works

| | Speed per message (26 questions) | Accuracy (own tests) | Confident mistakes |
|---|---|---|---|
| Mapika (teacher) | about 400 ms | 86% | 1 |
| MiniLM student | 1.6 ms | 77% | 1 |
| Qwen3-Embedding student | 7 ms | 84% | 0 |

- A better embedder mattered more than the practice score.
- The student only knows the questions it trained on, and only the kinds of messages it practiced on
  (the spam gap proved that).

## 6. The cascade is the best overall setup

The Qwen student answers when it is at least 0.7 sure; Mapika handles the rest.

- 86% accuracy, the same as Mapika alone
- The student handles about half the questions in milliseconds
- It acts on its own more often (31 vs 27 of 44)

## 7. Honest limits

- The test sets are small: 44 own questions plus 231 public JevBench questions.
- The 0.7 cutoff was picked by looking at those same 44 questions.
- The student practiced on generated messages, not real ones.
- Jev itself was never tested (no API credits).

## Next steps

Test on real messages, build a bigger test set, and compare against Jev once credits are added.
