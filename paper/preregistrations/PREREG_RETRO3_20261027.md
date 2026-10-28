# PREREG RETRO3: does the cosine retrofit's small perplexity cost hide a retrieval cost?

DATE: 2026-10-27, registered before any RETRO3 run. Code: `../curvature_retrofit/exp_retrofit.py` with
R_RETRIEVAL=1 (the retrieval suite runs only after the perplexity evaluations; training is unchanged). Mac mini,
`../curvature_retrofit/run_retro3.sh`, after FMT5. Results in `results_ret/` (Qwen) and `results_ret_smol/` (SmolLM).

## Why

RETROFIT and RETRO2 switched a quarter of a pretrained model's attention layers to cosine (direction-only) scoring
and healed them by distillation. Against the heal-only control the cost was +0.3% perplexity on Qwen2.5-1.5B and
+1.7% on SmolLM-1.7B. Perplexity is an average over every token, and most tokens are predictable from nearby
context. Cosine scores are bounded, so the risk sits in the rarer tokens that need a sharp look far back: recalling a
value given earlier, or copying a sequence. Both RESULT files say that no downstream task was tested. This is that
test, and the claim it can break is "near-free".

## Tests (generated in the code from a fixed word list and seed; no dataset)

- **kv8, kv32, kv64:** a list of n "word: 4-digit number" pairs, then one word from the list; the model must give its
  number. Exact match. The prompts are about 65, 250 and 500 tokens long, and healing used 256-token sequences.
  100 items each.
- **ind40, ind150:** a random list of 40 or 150 words, " |", then the same list. Score: the share of the copy's tokens
  predicted correctly, after its first. 100 items each.
- **wiki_rep, wiki_new:** the held-out WikiText-2 perplexity of RETROFIT, split by target token: tokens that already
  appeared earlier in their 256-token window (names, terms) and the rest.

Every test is scored with teacher forcing (one forward pass per item). For kv, every token of the number must be the
model's top choice, which is the same as greedy decoding reproducing it.

## Conditions

Qwen2.5-1.5B and SmolLM-1.7B, the RETROFIT/RETRO2 recipe (LoRA r16 plus per-head temperatures, KL distillation from
the unmodified model, 1M WikiText-2 tokens), seeds 0-1:
- **cos_Q:** every fourth layer switched to cosine.
- **none:** heal-only control.

For each condition the suite scores the teacher, the swapped model before healing (cos_Q only) and the healed model.
For cos_Q it also scores two ablations of the healed model: the swapped heads reverted to the dot product, and
replaced by uniform attention. uni_Q is not rerun; the uniform ablation of healed cos_Q measures what those layers
carry, at no training cost.

## Validity gates (not predictions)

- **Reproduction:** Qwen cos_Q seed 0 and none seed 0 must reproduce RETROFIT's healed perplexities (15.6239 and
  15.5951) to 0.001. MPS training is deterministic per seed, so this checks the code path, not a replication. If it
  fails, the run is reported as a deviation and the RETROFIT comparison is dropped.
- **Floor:** a kv or ind test enters a prediction only where the heal-only model scores at least 0.20 on it.
  Otherwise that test is void for that model.

## Predictions (seed means)

Retention is the cos_Q healed score divided by the heal-only healed score, averaged over the tests that pass the
floor gate.

- **L1.** On Qwen, cos_Q keeps at least 0.90 of heal-only's kv accuracy (P = 0.55).
- **L2.** On SmolLM, the same holds (P = 0.45).
- **L3.** The cost grows with distance: on Qwen, heal-only minus cos_Q is larger at kv64 than at kv8 (P = 0.6).
- **L4.** Copying survives: on both models, cos_Q is within 0.05 of heal-only on ind40 and on ind150 (P = 0.5).
- **L5.** The perplexity average hides a retrieval cost: on Qwen, cos_Q's perplexity cost relative to heal-only is
  larger on wiki_rep than on wiki_new (P = 0.55).
- **L6.** The healed cosine heads do the retrieving: on Qwen, replacing them with uniform attention lowers mean kv
  accuracy by at least 0.20 from healed cos_Q (P = 0.65).

## Reading

- **L1 and L2 hold:** the retrofit is cheap on retrieval too, and the v2 addendum can say so for two families.
- **L1 fails:** the 0.3% hides a real cost. The v2 addendum reports perplexity and retrieval side by side, with
  retrieval as the limit.
- **L6:** separates "cosine heads retrieve" from "other layers retrieve and the cosine heads idle".

**Scope:** two 1.5-1.7B models, synthetic retrieval plus one corpus split, two seeds. **Energy:** 8 runs, about 12
Mac hours, about 0.4 kWh.

## Smoke run (disclosed; after registration, before any RETRO3 run)

CPU, Qwen2.5-1.5B, 3 items per test, the wiki split on two windows of local text (not WikiText), no healing. The
teacher scored kv8 1.00, kv32 0.67, kv64 0.67, ind40 1.00 and ind150 1.00. Swapped before healing, kv fell to 0 at
every size and copying to 0.92 and 0.75. Reverting the swapped heads gave the teacher's scores exactly. Nothing
registered above is changed.
