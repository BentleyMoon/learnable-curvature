# PREREG RETRO4: does healing on longer sequences remove the retrofit's long-range recall cost?

DATE: 2026-10-27, registered before any RETRO4 run and after RETRO3 was scored. Code:
`../curvature_retrofit/exp_retrofit.py` with R_RETRIEVAL=1, R_KV100=1 and R_SEQ=512. Mac mini,
`../curvature_retrofit/run_retro4.sh`. Results in `results_ret4_smol/`.

## Why

RETRO3 found the cost of switching every fourth layer to cosine scoring sits in long-range recall. On SmolLM-1.7B,
recall from 64 pairs (about 500 tokens) fell from 0.655 with heal-only to 0.27, while recall from 8 pairs was
unchanged. Healing used 256-token sequences, so the healed model never practised attention past 256 tokens. If
that is the cause, healing on 512-token sequences should close most of the gap at 64 pairs and leave a larger gap
at 100 pairs (about 900 tokens), past the new healing length. If the gap at 64 pairs stays, the cause is the
bounded cosine score itself, and longer healing is not the fix.

## Change from RETRO3

- **Healing length 512 tokens**, with 4 sequences per step and 1 per micro-batch, so tokens per step (2,048), tokens
  per micro-batch and the step count are the same as RETRO3. Budget 1M tokens. Evaluation windows stay at 256.
- **kv100 added:** 100 items of 100 pairs, drawn after RETRO3's items from the same seed, so RETRO3's 500 items are
  identical.

## Conditions

SmolLM-1.7B only (the larger RETRO3 cost), seeds 0-1: cos_Q and none (heal-only, also healed at 512). Ablations of
healed cos_Q as in RETRO3: cosine heads reverted to dot product, and made uniform.

## Validity gate

- **Floor:** a test enters a prediction only where heal-only scores at least 0.20 on it.

## Predictions (seed means; retention = cos_Q healed / heal-only healed, both at 512)

- **M1.** At kv64, retention is at least 0.70 (RETRO3: 0.41) (P = 0.45).
- **M2.** Retention at kv100 is lower than at kv64 (P = 0.65).
- **M3.** Mean retention over kv8, kv32 and kv64 is at least 0.90 (RETRO3: 0.79) (P = 0.35).
- **M4.** On ind150, cos_Q is within 0.05 of heal-only (RETRO3: 0.218 apart) (P = 0.35).
- **M5.** cos_Q's held-out perplexity is at most 2.0% above heal-only's (RETRO3: +1.7%) (P = 0.6).
- **M6.** At kv64, reverting the healed cosine heads to dot product scores lower than keeping them cosine (RETRO3:
  0.44 reverted against 0.27 cosine) (P = 0.5).

## Reading

- **M1 and M2 hold:** the cost follows the healing length, and the fix is to heal at the lengths the model will use.
  The v2 addendum says so with the 512 numbers.
- **M1 fails:** the cosine scoring itself limits long-range recall at this healing budget; the addendum reports
  that as the limit.
- **M6:** whether longer healing makes the network use the cosine heads at long range.

**Scope:** one 1.7B model, synthetic recall and copying up to about 900 tokens, two seeds. **Energy:** 4 runs, about
8 Mac hours, about 0.27 kWh.
