# Learnable Curvature in Attention Is a Per-Head Choice of Scoring Rule

**Bentley Moon**, independent researcher
Working draft, September 2026. Code, saved runs and every number: https://github.com/BentleyMoon/learnable-curvature

---

## Abstract

I gave each attention head in a small transformer one trainable number, κ, meant as the curvature of
the space that head measures similarity in, so the model could choose its own geometry. On
character-level WikiText-2 at width 96 with 3 layers, the model with κ beats plain dot-product
attention by 2.41% perplexity on all 4 paired seeds. The controls then took the geometry story
apart. Read in the code, κ is a switch between three scoring rules: the dot product, a dot product
scaled up by the vectors' lengths, and cosine similarity. A random fixed setting of that switch keeps
112% of the learned gain, cosine on every head keeps 101%, and the length-scaled rule on every head
keeps 2%. The gain is cosine attention, which is already known. What is left is a smaller and, I
think, more useful finding about how little of it is needed: one fixed cosine head in the first layer
keeps 81 to 95% of the gain at 3 layers, and at 6 layers it improves on plain attention on 4 of 4
seeds (+1.08 to +1.25%) where learned κ improves on nothing (mean +0.02%). Starting that head at
cosine and letting κ learn does as well as fixing it, and at 12 layers the head still helps on both
seeds tried. The same head placed in a later layer hurts. A trained model also compresses to one bit per head at no measurable cost. I
report the claims from my earlier drafts that did not survive, next to the control that removed each
one.

*Added 2026-10-23:* every run in this draft used one learning rate, 3·10⁻⁴. With each arm given its own
best of five learning rates, the cosine gain at width 96 is +0.04% (Section 4.1; C27 in CLAIMS.md), so
the gains reported here are gains at that rate.

## 1. Introduction

Most of the structure in a neural network has to be learned from data, and the geometry it computes
in is usually fixed before training starts: flat space, dot products, the same rule in every head of
every layer. There is good work showing that curved spaces suit some kinds of data, hyperbolic space
for trees and hierarchies, spheres for things that repeat or wrap around, and I wanted to know what a
model would do if it could pick for itself. So in March 2026 I gave each attention head a single
learnable scalar κ, negative for hyperbolic, positive for spherical and zero for flat, initialised at
zero, and watched where the heads went.

They left zero within the first few epochs and split between the two sides, and the model with κ
trained to a lower perplexity than the one without. My first drafts read this as a model discovering
the geometry of language. This paper is what happened when I tested that reading. Every experiment at
width 96 had its predictions written down before it ran, and most of the controls were built to break
the result rather than confirm it.

The geometry did not survive. What survived is this:

- κ is a per-head switch between three scoring rules, and trained heads sit fully on one rule (Section 2).
- The gain at width 96 comes from cosine scoring being present. A random or uniform fixed assignment
  does as well as a learned one (Section 4.2).
- How little is needed: one fixed cosine head in the first layer gets most of it at 3 layers, about as
  much as cosine on all 24 heads at 6 layers, and more than learned κ does at 6 layers. In a later layer
  the same head hurts (Sections 4.3 and 4.4).
- A trained model's κ compresses to one bit per head at no cost, and the bits are tied to the weights
  around them (Section 4.5).
- An account of eleven claims from my earlier drafts and the control that removed each (Section 5).

Per-head learnable curvature is not a new idea (Cho et al., 2023), and cosine attention is not new
either (Henry et al., 2020; Liu et al., 2022). I did not find prior work that uses cosine attention in
the first layer only, and I report that as not found rather than as absent.

## 2. What κ does

In the original code the score between a query q and a key k in a head with κ is a blend of three
rules, with d_k the head dimension:

- dot product: s_dot = q·k / √d_k
- length-scaled ("hyperbolic"): s_len = s_dot · (1 + ½‖q‖‖k‖ / √d_k)
- cosine ("spherical"): s_cos = cos(q, k) · √d_k

The weights are w_len = max(0, tanh(−3κ)), w_cos = max(0, tanh(3κ)) and w_dot = 1 − w_len − w_cos. No
curved space is ever computed; the names describe what each rule was meant to imitate. At |κ| = 1 the
winning rule already has weight 0.9951, and trained heads end far past that, with mean |κ| of 2.55 in
the 8-head, 4-layer model and 2.2 to 2.3 in the width-96 models (Figure 1). Replacing the blend with a
hard switch on the trained 8×4 model changes its perplexity from 3.745593699 to 3.745593711. So κ, in
practice, chooses one scoring rule per head.

## 3. Setup

**Data.** Character-level WikiText-2 (raw), with each character mapped to one of 256 values and cut
into 64-character windows. The March runs load the train and validation splits directly. The width-96
runs concatenate train, validation and test, hold out the final 10%, and score 2,500 held-out windows.
Perplexities from the two families are not comparable with each other.

**Models.** Pre-norm transformers with learned position embeddings. The March runs span d_model 16 to
128 and 1 to 4 layers. The width-96 runs use 4 heads per layer, d_ff 384 and 3 or 6 layers (about
0.37M parameters at 3 layers), trained for 20,000 steps at batch 64 with AdamW, learning rate 3·10⁻⁴
for the weights and 10⁻² for κ.

**Arms.** Learned (κ free, from zero); zero (κ fixed at 0, plain dot product); fixed code (each κ frozen
at +1 or −1 from the first step); random code; late binary (learned, then rounded to its sign for the
last 25% of training); and native binary (κ is its own sign from step 1, trained through a
straight-through estimator). All arms of a seed start from the same weights.

**Measure.** "Share of the gain kept" is (zero − arm) / (zero − learned), computed per seed against that
seed's own references. At 6 layers, where learned κ gains nothing, I report improvement over the same
seed's zero model instead.

## 4. Results

### 4.1 The switch beats the dot product at width 96

At 3 layers, learned κ lowers perplexity against κ = 0 by 2.32, 2.46, 2.18 and 2.69% on four paired
seeds (mean 2.41%, sd 0.22). This is the one size where the effect is clear and repeatable; smaller
models are discussed in Section 4.7.

*Added 2026-10-23:* Every run above used one constant learning rate, 3·10⁻⁴. A sweep over five learning
rates at d_model 96 with 3 layers (3 seeds each, C27 in CLAIMS.md) reproduces the gain at that rate (all
cosine against the dot product, −2.33%). With each arm at its own best rate the gap is +0.04%: dot
product 4.336 at 2.4·10⁻³, all cosine 4.338 at 1.2·10⁻³. At 4.8·10⁻³ the dot product is better by 1.7%
on all 3 seeds. Cosine scoring moves the best learning rate lower and makes runs more consistent across
seeds (sd 0.002 against 0.032 at each arm's best); at matched tuning it does not lower perplexity at
this scale. Learned κ and the 6-layer first-layer recipe were not swept.

### 4.2 The controls: a random switch does as well

If the model were finding a useful geometry, a random assignment should do worse. It does not (Figure
2). A random ±1 code frozen from the first step keeps 111.6% of the learned gain (99.9 to 123.5% across
four seeds). Every head on the cosine rule keeps 101.1 and 100.2% on two seeds. Every head on the
length-scaled rule keeps 3.4 and 0.8%, which is the same as no rule at all. Across 20 runs with frozen
mixed codes, 18 keep at least 90%; the two that fall below 60% (19.4 and 46.6%) share one code, and that
code kept 98% on a third seed. An earlier single-seed run at width 128 with 2 layers pointed the same
way: random 4.1812, learned 4.1834, fixed to a previously learned code 4.1857 (Figure 3).

So learning which head gets which rule adds nothing I can measure. What matters is that some heads
score by cosine.

### 4.3 Where the cosine head sits

At 3 layers I moved a fixed cosine code between layers (Figure 9, left). A whole layer of cosine heads
keeps 112 and 109% of the gain in layer 1, 6 and 3% in layer 2, and 14 and 35% in layer 3. A single
cosine head keeps 95 and 90% in layer 1 (head 0) and 81 and 88% (head 3), but 9 and 19% in layer 2 and
3 and 22% in layer 3. One fixed cosine head in the first layer, with nothing learned, gets most of
what learned κ gets.

### 4.4 Six layers

At 6 layers learned κ stops helping. Against the same seed's zero model it changes perplexity by
+0.64, −0.38, −0.52 and +0.35% (mean +0.02%), and a three times larger κ step does not rescue it
(+0.78, −0.47%). One fixed cosine head in layer 1 improves on zero by 1.25, 1.08, 1.10 and 1.16% (mean
1.15%), better than learned κ on each seed. A whole first layer of cosine heads gives 1.35 to 1.85%.
The same single head placed in layer 4 makes the model worse (−0.35, −0.43%). At twice the training
(40,000 steps, one seed) the fixed head gives 1.65% and learned κ 1.09%.

Cosine on all 24 heads gives 1.06, 1.29, 1.17 and 1.49% (mean 1.25%), so cosine attention alone
carries the effect. One head in layer 1 lands within a third of a point of that (differences +0.20,
−0.21, −0.07, −0.33; mean −0.10). The whole first layer beats cosine everywhere on all four seeds, by
0.16 to 0.37 points, but the bar I set in advance was at least 0.3 points on three of four seeds and it
was met on two. I do not claim that the first layer adds anything beyond cosine attention everywhere;
four more seeds would settle it.

I noticed a pattern in the learned runs after the fact: learning pushes nearly every first-layer head
to the length-scaled side, and the learned runs that end with exactly one cosine head in layer 1 are
the better ones. So I wrote down a test before running it (C23). Start that one head at cosine and let
κ learn from there, and the model lands within 0.04 points of the fixed recipe on all four seeds
(+1.21, +1.06, +1.07, +1.12%), with the head still cosine at the end every time. Start every head on
the length-scaled side instead and let κ learn, and it ends below plain attention on all four (−0.62,
−0.29, −0.23, −0.35%). Learned κ at depth fails because of where it starts, not because learning
walks away from the recipe. Fixing every head on the length-scaled side is worse than plain attention
too (−0.62, −0.28%, two seeds; C24), so the gain is measured against no curvature, not against a
length-scaled baseline.

At 12 layers (two seeds; C25) the one-head recipe still improves on plain attention, by 0.67 and
0.81%, and a whole cosine first layer by 0.82 and 1.01%, while learned κ gives 0.25 and 0.18%. I had
set 0.75% on both seeds as the bar for saying the recipe holds to 12 layers, and one seed falls short
of it, so I say only that it has not stopped working by 12 layers.

### 4.5 One bit per head

In a trained model the magnitude of κ carries nothing. Rounding every κ of the 8×4 model to ±1 moves
perplexity from 3.7456 to 3.7455, and at width 96 the change is at most 0.0041% on four seeds. The signs
are not interchangeable, though: shuffling the same set of signs across heads costs 12.79% on average
over ten shuffles, and every shuffle is worse (the mildest, 4.98%). A trained model's weights are built
around the rule each head uses.

Where the bit comes from matters too. Learning κ for 75% of training and then rounding it keeps 99.8 to
99.9% of the gain, and freezing the learned code after the first 1% keeps 100.5%. Training in one bit
from the first step, through a straight-through estimator, keeps only 20% (16, 6, 37 and 22%). I have
not tested why.

### 4.6 What the heads do

In the 8×4 model, κ correlates with how far back a head looks: r = +0.880 with attention mass 16 or more
characters back, +0.875 with mean attention distance, and −0.815 with mass in the nearest four
characters (32 heads). My earlier drafts read this as heads discovering their function. A forced-rule
control says most of it is the rule itself. Forcing one rule on every head of a 4×2 model with nothing
learned gives a mean attention distance of 5.33 characters for cosine, 2.87 for length-scaled and 2.89
for the dot product; in the trained 4×2 model, cosine heads look 5.90 back and length-scaled heads
2.06. The forced rule accounts for 64% of that gap (Figure 4). I use "attention-range proxy" for this
measure; it is not a syntactic probe.

### 4.7 Smaller models

The March sweep trained 18 matched configurations from width 16 to 96 and 1 to 4 layers for 55 epochs
(Figure 5). Against a baseline trained in the same run, κ changes perplexity by −0.34% on average and is
better in 15 of 18; a separately trained κ model against the same baseline gives −0.19%, better in 11 of
18. Two κ runs of one configuration differ by 0.59% on average, so the gain is inside the spread. At
width 48 with 2 layers and three paired seeds, learned κ against zero is −0.03% (sd 0.78). I see no
reliable gain below width 96.

### 4.8 Exact distance

A separate model scores attention by exact distance between queries and keys. On synthetic tasks
(20 epochs, one seed) exact Euclidean distance beats the dot product by 28.30% on a periodic task and
13.09% on a mixed one, and ties on a hierarchical one; adding curved geometry on top changes little
(+1.41, +0.58, −0.01%). On WikiText-2 (55 epochs, one seed) exact Euclidean distance gives 4.0958
against 4.1131 for the dot product, and a learned per-head temperature on the dot product gives 4.1129.
Both cosine scoring and exact distance limit how much a key's length can move its score. I suspect that
is the shared ingredient, and I have not tested it.

### 4.9 A toy with real curvature

Before the transformer work I built a small numpy model with its own autograd that computes true
Poincaré-ball and sphere distances with one learnable κ and a contrastive loss in six dimensions.
Re-run on 11 seeds, κ reaches −0.620 to −0.586 on tree-structured data, below a flat-cluster control on
every seed; but the control itself drifts to −0.424 to −0.007, so "negative" alone means little. On
periodic data κ reaches +0.146 to +0.164, positive while the control is negative on every seed, and
above the script's own 0.15 threshold for "spherical" on 6 of 11. On mixed data the periodic half gets
+0.297 to +0.325 and the tree half +0.027 to +0.063. The toy separates the two kinds of data by
direction; it does not recover hyperbolic and spherical geometry in any strong sense.

### 4.10 Ternary weights

With block weights restricted to three values in the style of BitNet b1.58 (Ma et al., 2024), learned κ
against zero gives −1.13% (sd 1.37, better on 3 of 4 seeds), about half of the float-weight gain and
not reliably.

## 5. What the earlier drafts claimed, and what removed it

| Earlier claim | What removed it |
|---|---|
| The model finds its own geometry, and that is the source of the gain | A random fixed code keeps 112%; all-cosine keeps 101% (4.2) |
| −7% at width 96, 3 layers, and up to 38% parameter savings | The baseline had stopped at epoch 10 against a κ model at epoch 55. Matched at 55 epochs: −0.68% (same run), −0.94% (separate run) |
| Heads discover their function (r = 0.88) | Forcing the rule with nothing learned reproduces most of the pattern (4.6) |
| The toy discovers spherical geometry for periodic data (κ = +0.15) | The script's own threshold labels the original run Euclidean (κ = +0.149); 6 of 11 seeds cross it (4.9) |
| The toy discovers product geometry | The tree half stays at +0.027 to +0.063, not negative (4.9) |
| Zeroing κ costs 169%, so the geometry is worth that much | That removes the rule a trained model's weights were built around. Retrained with κ = 0 the cost is 2.41% (4.1) |
| Fixed discovered κ beats learned κ (4×2, 20 epochs) | Run A: 4.1865 vs 4.1943; run B reverses it: 4.1857 vs 4.1834 |
| Geometry beats none at width 128, 2 layers, 20 epochs | Run A: +0.03%; run B: −0.78%. The two runs disagree |
| Bounded κ in [−2, 2] is slightly better | 4.1773 vs 4.1859 (−0.21%), one run each, smaller than the gap between two zero runs (+0.56%) |
| The WikiText-2 pilot beats its baseline by 1.15% | 5 epochs, one seed, inside the run-to-run spread of 4.7 |
| Only the sign of κ matters at inference | True of a trained model (4.5). From scratch, which heads get which sign does not matter either; what matters is cosine scoring in the first layer |

## 6. Related work

Cosine and query-key normalised attention scale the score by vector direction rather than length.
Henry et al. (2020) normalise queries and keys and learn the scale; my cosine rule uses a fixed √d_k.
Swin Transformer V2 (Liu et al., 2022) uses scaled cosine attention with a learnable temperature per
head and layer; ViT-22B (Dehghani et al., 2023) applies LayerNorm to queries and keys for stability at
scale; nGPT (Loshchilov et al., 2025) normalises the whole network onto a hypersphere; and Zhai et al.
(2023) relate training instability to attention entropy. All of these apply their normalisation to
every layer.

Curved spaces in neural networks go back to Poincaré embeddings (Nickel and Kiela, 2017), hyperbolic
network layers (Ganea et al., 2018), mixed-curvature product spaces (Gu et al., 2019), hyperbolic
attention (Gulcehre et al., 2019) and fully hyperbolic models (Chen et al., 2022). Closest to this work,
Cho et al. (2023) give each attention head its own learnable curvature and compute real stereographic
operations, for graphs; CAT (Lin et al., 2025) routes each token between curved attention branches.
Adaptive attention span (Sukhbaatar et al., 2019) is an earlier case of heads learning different
settings. Kim et al. (2025) argue that the first layer's attention is special in another sense, as an
output worth reusing.

## 7. Limitations

- Every language result is character-level WikiText-2. Nothing here is tested on subword tokens,
  another corpus, or any task beyond next-character prediction.
- The clear effects appear only at width 96, at 3 and 6 layers, and the largest model is under 0.9M
  parameters. I make no claim about larger models.
- Seeds are few: four for most width-96 claims, two for the uniform arms at 3 layers, and one for most
  of the March runs, which have no confidence intervals.
- "Spherical" and "hyperbolic" name scoring rules, not geometries.
- The attention-range analysis is a proxy for distance, not a probe of linguistic function.
- There is no speed, memory or deployment claim. The one-bit model still computes every branch, and
  nothing was timed.
- Two of the twenty sweep configurations did not finish (width 128 with 1 layer crashed; with 3 layers
  it stopped at epoch 5).
- The exact-distance runs use a separate implementation with one seed each.
- The width-96 experiments had their predictions written down before they ran; the March experiments
  did not.
- The 12-layer and all-length-scaled results rest on two seeds each, and do not settle how the recipe
  scales with depth.
- *Added 2026-10-23:* every run used one constant learning rate (3·10⁻⁴, no warmup, no schedule). Tuned
  per arm, the cosine gain at width 96 disappears (C27). The first-layer recipe and learned κ have not
  been re-tested that way.

## 8. Conclusion

I set out to let a model find its own geometry, and what it found was a switch. The part worth keeping
is small and concrete: at this scale a single attention head that scores by direction instead of
length, fixed in the first layer or simply started there, does about as much good as cosine attention
everywhere and more than the curvature the model learned for itself, while the same head placed deeper
does harm. I think that is a better result to hand on than the one I started with, because every step
of it can be checked, and because the reasons the bigger story fell apart are written down beside it.

## Reproducibility

Every number in this paper is computed from saved run files by `python reproduce.py check`, which
also verifies each file against a recorded SHA-256 hash and rebuilds all figures in seconds without
training. The toy re-runs bit for bit on 11 seeds, a retrained width-96 run matches its original on 4
of 4 seeds to every stored digit, and the corpus download is checked byte for byte against the file the
runs used. Code, results and the full claims table (CLAIMS.md) are at the repository linked above.

## References

- Chen, W. et al. (2022). Fully Hyperbolic Neural Networks. ACL 2022, 5672–5686.
- Cho, S. et al. (2023). Curve Your Attention: Mixed-Curvature Transformers for Graph Representation Learning. arXiv:2309.04082.
- Dehghani, M. et al. (2023). Scaling Vision Transformers to 22 Billion Parameters. ICML 2023.
- Ganea, O., Bécigneul, G. and Hofmann, T. (2018). Hyperbolic Neural Networks. NeurIPS 2018.
- Gu, A., Sala, F., Gunel, B. and Ré, C. (2019). Learning Mixed-Curvature Representations in Product Spaces. ICLR 2019.
- Gulcehre, C. et al. (2019). Hyperbolic Attention Networks. ICLR 2019.
- Henry, A., Dachapally, P. R., Pawar, S. and Chen, Y. (2020). Query-Key Normalization for Transformers. Findings of EMNLP 2020, 4246–4253.
- Kim et al. (2025). First Attentions Last. arXiv:2510.14614.
- Lin, Ojha and Bai (2025). CAT: Curvature-Adaptive Transformers. arXiv:2510.01634.
- Liu, Z. et al. (2022). Swin Transformer V2: Scaling Up Capacity and Resolution. CVPR 2022.
- Loshchilov, I. et al. (2025). nGPT: Normalized Transformer with Representation Learning on the Hypersphere. ICLR 2025.
- Ma, S. et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764.
- Nickel, M. and Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. NeurIPS 2017.
- Sukhbaatar, S., Grave, E., Bojanowski, P. and Joulin, A. (2019). Adaptive Attention Span in Transformers. ACL 2019, 331–335.
- Zhai, S. et al. (2023). Stabilizing Transformer Training by Preventing Attention Entropy Collapse. ICML 2023.
