# PREREG RETROFIT: does curvature attention survive as a drop-in for a pretrained model?

DATE: 2026-10-21. Registered before any full run, after smoke runs disclosed below. Code: `exp_retrofit.py`. Mac
mini (`run_retrofit.sh`, after the B queue). Base: Qwen2.5-1.5B (28 layers, 12 heads). Results in results/.

## Why

The preprint (DOI 10.5281/zenodo.23165432) tested curvature attention on small models trained from scratch.
Adoption needs a better story: switch the mechanism into an existing model and heal it cheaply by distilling from
the original. The devil's advocate is that healing can teach the network to route around the new heads, giving
good scores while the mechanism does nothing. The ablations below test for that.

## Method

Per-head scoring rule as in the preprint: κ blends the dot product (0), length-boosted (<0) and cosine (>0).
- **Swap:** heads set to cosine (κ = +1, cosine weight 0.995) with a learnable per-head temperature.
- **Heal:** LoRA r16 on q/k/v/o plus the temperatures, KL distillation from the unmodified model on WikiText-2
  (first 90% of the text), 1M tokens, sequences of 256, seeds 0 and 1.
- **Evaluation:** perplexity on 200 held-out sequences (the last 10% of the text).

Conditions:
- **cos_L1:** layer 1, all heads cosine.
- **cos_Q:** layers 1, 5, ..., 25 (7 of 28), all heads cosine.
- **uni_Q:** the same 7 layers switched to uniform attention (all scores 0). Equal disruption, no geometry.
- **learn_all:** every head gets a learnable κ starting at about 0 (±0.001: the blend has a kink at exactly 0
  where MPS returns a zero gradient, found in smoke testing).

## Smoke runs (disclosed: 4 evaluation sequences, 2-4 training steps, seed 9)

- Teacher perplexity 20.59.
- cos_Q swapped, before healing: 38.09.
- After 4 steps: healed 29.13, reverted to dot 20.05, cosine replaced by uniform 41.40.
- Zeroing the 7 layers' attention: 503 (teacher) and 519 (healed). Too destructive to separate anything, so it is
  reported but not scored.

## Predictions (seed means on the full evaluation)

- **G2.** cos_Q healing recovers at least 80% of the gap: (swapped − healed) / (swapped − teacher) ≥ 0.80
  (P = 0.5).
- **G3.** The geometry matters for healing: uni_Q's healed perplexity is at least 5% above cos_Q's (P = 0.5).
- **G4.** The healed network depends on the cosine rule: reverting cos_Q's swapped heads to the dot product raises
  healed perplexity by at least 5% (P = 0.45).
- **G5.** Not routed around: replacing cos_Q's healed cosine heads with uniform attention raises perplexity by at
  least 10% (P = 0.55).
- **G6.** When free to choose, the network keeps the dot product: in learn_all, under 10% of heads end with
  |κ| > 0.1 (P = 0.5).
- **G7.** cos_L1 recovers at least 95% of its gap (P = 0.6).

## Reading

G2 + G5 (+ G4) together are the drop-in claim: heals cheaply, and the healed heads carry load in their new form.
G2 without G5 is the devil's advocate case: healed by routing around.

**Scope:** one 1.5B model, perplexity on WikiText-2 only; quality, not speed.
**Energy:** 8 runs, about 8 Mac hours, about 0.3 kWh.

## Addendum (2026-10-23, after cos_Q, uni_Q and cos_L1 read; before any run of this control)

Every healed condition ends BELOW the teacher's perplexity (cos_Q 15.62, cos_L1 15.50-15.61, uni_Q 16.64-16.83 vs
15.91). Distilling with a LoRA adapter on WikiText text may itself lower WikiText perplexity, independent of any
swap. **Control `none`:** the unmodified model healed by the same procedure (seeds 0-1). Descriptive, no threshold:
the healed-minus-teacher perplexity it gives is subtracted before any "beats the teacher" reading of a swapped
condition, and the curvature v2 reports swapped conditions relative to `none` as well as to the teacher.

## Read note (2026-10-23, after learn_all seed 0; written before seed 1)

**G6 is uninformative by design.** kappa trained at the adapter's learning rate (2e-4) for ~488 steps, so an
Adam-type optimiser could move it at most about 0.1 in total. "Under 10% of heads end with |kappa| > 0.1" was
therefore close to guaranteed; observed: 0 of 336, max 0.048. G6 will be reported as scored but not interpretable.
A test of whether the network chooses a geometry needs kappa on its own, larger learning rate (parked).

**learn_all is a near heal-only control.** kappa stays near 0, so the model is close to unmodified, and it heals to
15.51, below the teacher's 15.91. Against that baseline, cos_Q costs +0.7% (15.62) and uni_Q +7.9% (16.74). The
curvature v2 reports swaps against this baseline (and the registered `none` control when it lands), not as
"beating the teacher".
