# Paper skeleton

Headings, with the claims and numbers that belong under each. The prose is Bentley's; nothing below
is written for publication. Claim IDs point to [CLAIMS.md](../CLAIMS.md), which is generated from
the saved runs and is the authority if a number here ever disagrees with it.

The title is still to be chosen. The March titles ("Discovering Optimal Geometry for Neural
Computation") no longer fit what the data support.

---

## Abstract

- What κ is: a per-head switch between three scoring rules (M1).
- The gain at d_model 96 / 3 layers: −2.41% perplexity vs dot product, 4 of 4 paired seeds (C1).
- The reversal: a random fixed code keeps 111.6%, all-cosine keeps 100.7%, all-length-scaled keeps 2.1% (C2, C3).
- Compression: in a trained model the sign carries it; shuffling signs costs +12.79% (C4).
- Scope: character-level WikiText-2, under 0.9M parameters, one scale with a clear effect.

## 1. Introduction

- The March hypothesis: let each head learn its geometry (the original drafts).
- What the controls found instead (C2, C3; withdrawn W1).
- Contributions, as the data now support them:
  - a control ladder that separates "learning the geometry" from "having the rule" (Figure 2)
  - the finding that cosine scoring on some heads is what helps at this scale (C2)
  - sign-only compression of a trained model, with a permutation control (C4, C5)
  - an account of what the drafts claimed and which control removed each claim (W1 to W11)

## 2. Method

- The three scoring rules, from `CurvatureDistance` in `src/learnable_curvature_transformer.py`:
  - plain dot product: q·k / √d_k
  - length-scaled ("hyperbolic"): (q·k / √d_k) · (1 + ½‖q‖‖k‖ / √d_k)
  - cosine ("spherical"): cos(q, k) · √d_k
  - weights: max(0, tanh(−3κ)), max(0, tanh(3κ)), and the remainder for the dot product
- At |κ| = 1 the winning rule already has weight 0.9951; trained heads sit at |κ| 2.2 to 2.55 (M1, Figure 1).
- Hard switch equals the blend on a trained model: 3.745593699 vs 3.745593711 (M1).
- Data: character-level WikiText-2 (256 byte values, 64-character windows).
- Two protocols, not comparable with each other (Limitation 9):
  - March: d_model 16 to 128, 1 to 4 layers, 5 to 61 epochs, mostly one seed.
  - d_model 96: 4 heads × 3 layers, 20,000 steps, batch 64, 2,500 held-out sequences, paired seeds,
    each experiment preregistered before it ran.
- Arms: learned (continuous κ), zero (κ = 0), fixed code (±1 per head, frozen from step 1), random code,
  late binary (continuous then sign), native binary (sign from step 1, straight-through estimator).
- "Share of the gain kept" = (zero − arm) / (zero − learned), per seed.

## 3. Results

### 3.1 Heads leave zero early and split (C8; Figure 8)
- 4×2: 4 length-scaled / 4 cosine; 8×4: 17 / 15.
- 80% of the final range by epoch 4 (4×2) and 3 (8×4); d96 codes final by step 25.

### 3.2 The rule beats dot product at d_model 96 (C1)
- −2.32, −2.46, −2.18, −2.69% per seed; mean −2.41%, sd 0.22.

### 3.3 The control ladder (C2, C3; Figure 2): the central result
- All cosine, fixed: 101.1, 100.2% kept.
- Random ±1 code, fixed: 99.9 to 123.5% (mean 111.6%).
- All length-scaled, fixed: 3.4, 0.8% kept, the same as no rule.
- Frozen mixed codes: 18 of 20 runs keep ≥ 90%; the two failures share one code, which kept 98% on a third seed.
- March echo at d_model 128 / 2 layers: random 4.1812, learned 4.1834, fixed 4.1857 (one seed; Figure 3).

### 3.4 Compressing a trained model to one bit per head (C4, C5)
- 8×4: 3.7456 → 3.7455; d96: |change| ≤ 0.0041% on 4 seeds.
- Shuffling the signs across heads: +12.79% mean, every one of 10 worse.
- Late binary keeps 99.8 to 99.9%; code frozen after 1% keeps 100.5%; one bit from step 1 keeps 20%.
- Why one bit from step 1 fails is untested.

### 3.5 What the heads do (C6, C7; Figure 4)
- Wording: "attention-range proxy", not "syntactic".
- 8×4: r(κ, long-range mass) +0.880, r(κ, mean lag) +0.875, r(κ, local mass) −0.815.
- Forcing one rule with nothing learned: mean lag 5.33 (cosine) vs 2.87 (length-scaled) vs 2.89 (dot).
- The forced rule accounts for 64% of the trained 4×2 model's gap: most of the link is the rule itself.

### 3.6 Smaller scales (C9, C10; Figures 3 and 5)
- Sweep, 18 configurations at 55 epochs: −0.34% (same run, 15 of 18 better), −0.19% (separate run, 11 of 18).
- Two curvature runs of one configuration differ by 0.59% on average: the gain is inside the spread.
- d_model 48: learned vs no geometry −0.03% mean over 3 paired seeds (sd 0.78): no gain seen.

### 3.7 Exact distance scoring (C11, C12; Figure 6)
- Separate model, synthetic tasks: exact Euclidean vs dot product −28.30% (periodic), −13.09% (mixed), −0.01% (hierarchy).
- Curved geometry on top of exact distance: +1.41%, +0.58%, −0.01%.
- WikiText-2: exact Euclidean 4.0958 vs dot product 4.1131; a per-head temperature alone 4.1129. One seed.
- A point to test rather than claim: cosine scoring and exact distance both limit how much key length
  can move a score, which may be the shared ingredient.

### 3.8 The toy with true curvature (C13 to C15; Figure 7)
- Tree: κ −0.620 to −0.586 over 11 seeds; the flat control drifts to −0.424 to −0.007.
- Periodic: +0.146 to +0.164, above the script's own 0.15 threshold in 6 of 11 seeds.
- Mixed: periodic half +0.297 to +0.325, tree half +0.027 to +0.063 (not negative).

### 3.9 Ternary block weights (C17)
- −1.13% mean, sd 1.37, 3 of 4 seeds: about half, unreliably.

## 4. What the drafts claimed, and what removed each claim

- Table W1 to W11 from CLAIMS.md, as it stands. The −7% (W2) and the 169% (W6) need a sentence each,
  because they circulated.

## 5. Related work

Verify each before citing; these are pointers, not checked references.

- Query-key normalisation and cosine attention (Henry et al., 2020; Swin Transformer V2, Liu et al.,
  2022). The cosine rule here is close to both, so the surviving positive result is a small-scale
  instance of known prior work, and the paper should say that.
- Hyperbolic attention and embeddings (Gulcehre et al., 2019; Nickel and Kiela, 2017; Ganea et al., 2018).
- Mixed-curvature product spaces (Gu et al., 2019).
- Per-head learned span (Sukhbaatar et al., 2019).
- Ternary weights (BitNet b1.58, Ma et al., 2024) for C17.

## 6. Limitations

- The eleven in CLAIMS.md. The first three carry the most weight: one corpus at character level,
  under 0.9M parameters, few seeds.

## 7. The outline's three open items

- Convergence threshold: defined as the baseline's own final perplexity. The curvature model reaches
  5.1459 at epoch 13 of 20 (C16). One seed with no cosine-only arm, so it stays INCONCLUSIVE.
- Compression-sweep table: complete for 18 of 20 configurations from the saved files. d_model 128 with
  1 layer crashed; with 3 layers it stopped at epoch 5 (CLAIMS.md, sweep table).
- Proxy wording: "attention-range proxy", with C7's forced-rule control stated beside it.

## Figures

1. Scoring-rule weights against κ
2. The d_model 96 control ladder
3. The d_model 128 ablations
4. Attention range: trained heads vs forced rules
5. The compression sweep at 55 epochs
6. Exact distance on synthetic tasks
7. The numpy toy over 11 seeds
8. κ trajectories of the 8×4 model

Captions: `figures/CAPTIONS.md`.
