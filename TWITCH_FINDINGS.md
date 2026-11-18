# Twitch pilot: what we tried, what we found, where we landed

Goal: a Twitch chat bot that **decides first** (decision model + probability stasis), then answers **only from
true facts**, in the owner's own chat style. Everything runs locally on an RTX 3050.

## The models (two separate ones)

| Model | What it is | What it's good at | What it's bad at |
|---|---|---|---|
| **NLI** (DeBERTa zero-shot) | the decision model: "does text A support statement B?" -> yes/no score | concrete questions: "is this about video games?", "does the message back this fact?" | fuzzy questions: "is this lasting?", "is this harmless?" |
| **Embeddings** (MiniLM) | turns a sentence into numbers that capture meaning | "do these two mean the same?" ("i play valorant" ~ "im playing valorant" = 0.93) | negation ("i play" ~ "i don't play"), choosing a topic on its own |
| **Qwen2.5-1.5B** | small LLM that writes text | rewriting messy chat into clean, short facts | inventing things that weren't said |

## The data

- Twitch export: 9,063 rows from 2 accounts -> 8,214 messages after removing duplicates, links and bot commands.
- IP, city and IDs are dropped. Other people's @names become `@viewer`.
- Style, measured from the data: median 6 words, 88% lowercase, emotes like LUL / NotLikeThis, no emojis.

## Timeline: what we tried and what happened

| # | Tried | Result | Lesson |
|---|---|---|---|
| 1 | Decide -> stasis gate -> Qwen writes 4 replies from facts, in your style | spam and harassment blocked well; Qwen invented facts ("mousesquirrel mouse", "logitech g13") and sounded generic | the decision side works; a small LLM can't be trusted to state facts |
| 2 | "EDIT ME" counts as unknown; truth check = NLI "does it name something not in the facts?" | missed inventions (fortnite, top gun), flagged harmless lines | fuzzy NLI questions fail |
| 3 | Style rules taken from your data (lowercase, length, no emoji/hashtags); a character-pattern style judge (your chat vs Wikipedia) | replies started to sound like you | style can be measured and enforced without a model |
| 4 | Truth check = **lexical grounding**: every unusual word must come from the facts (valo ~ valorant, shity ~ shitty) | caught fortnite / brave / minecraft reliably | simple rules beat fuzzy models here |
| 5 | **Replies built straight from the facts** ("yeah i eat burritos") next to Qwen's replies | the bot almost never had to stay quiet; built replies won most of the time | true by construction > generated then checked |
| 6 | learnfacts v1: Qwen pulls facts, k-means makes topics, Qwen names them | 2 facts, then 49; topics mixed up (LA under "jobs"); private facts (weight, health) slipped through | topics must be fixed; privacy needs its own lock |
| 7 | Fixed drawers filed by NLI; private lock (NLI + word list); **filing twice (fact + message), both must agree** = stasis for filing | clean drawers, 132 private facts locked | your stasis idea works for organizing too |
| 8 | Cabinet = one JSON file per drawer, `_unsure`, `_private/`, `_blocked`, "you typed" survives re-learning | the organized structure you asked for | the JSON layout is plain Python; models only decide |
| 9 | Bot picks the drawer from **3 readings** (NLI + how close to the drawer's question + shared words), stasis on agreement | "what games do you play" stopped going to hobbies | combining readings beats any single one |
| 10 | 50 test questions + grader | climbed to **50/50** | ...but those tests were tuned against |
| 11 | 30 **new** questions (slang, typos, hard cases) | 23/30 -> **28/30** after fixes (stemming, synonyms, private questions never use drawers) | held-out tests are the honest score |
| 12 | **No generation**: cut facts straight out of your messages | 17x faster (6.8 s vs 117 s); **91% backed** by your message vs **53% for Qwen** | Qwen invents about half its pulls (the verify step throws them out) |
| 13 | ...but raw wording didn't merge, so I lowered the "lasting" bar | junk got in ("im fucking laughing hard"), the private folder grew to 331, and everyday words got blocked | optimizing a count (stable facts) instead of quality |
| 14 | Default-deny privacy + a fuzzy "is this harmless?" check + group-by-meaning filing | **1,555 of 1,566 facts locked**, cabinet almost empty, scores fell to 40/50 and 20/30 | fuzzy NLI question again; too many changes per run |

## Why we're back on "Qwen pulls, NLI files" (run 11, the best measured)

1. **Best measured results:** 50/50 on the tuned tests and 28/30 on new questions, with spam, harassment and private probes all at 100%.
2. **Qwen's rewriting makes repeats merge.** "im playing valorant dude" and "i play valorant while its cooking" both become "i play valorant", so stasis can count them as one fact said many times. Raw wording didn't merge well enough.
3. **Qwen's inventions don't survive.** The NLI check ("does your original message back this fact?") throws out about half of what Qwen writes. What stays is backed by your words.
4. **NLI files into fixed drawers, read twice, with stasis on agreement.** Concrete questions only.
5. **Embeddings only merge repeats,** which is the one thing they were clearly best at.
6. **In the bot, Qwen is a tool that reads from the cabinet.** It only gets the decided facts, and every reply it writes must pass the lexical truth check. Replies built straight from the cabinet compete with it, and usually win.

## What still needs work (one change at a time, test after each)

- **Privacy in learned facts:** the run-11 lock misses some family, legal and immigration topics. The fix is concrete per-fact checks plus a word list on the fact only, never fuzzy questions and never the whole message.
- **The truth check is lexical:** a made-up claim built from common words can still pass ("i use brave" was actually true, from your own history).
- **Tuned tests overstate quality:** always look at the held-out score.

## Rules we learned the hard way

- Ask the decision model **concrete** questions only.
- Change **one thing per run**, test both test files, and commit when it's good.
- Keep a **baseline commit** to go back to.
- Never block **everyday words** for privacy. Block distinctive private words only.
- Anything picked or generated for public chat needs a crude-word filter and an aimed-at-a-person filter.
