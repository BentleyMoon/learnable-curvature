# learnable-curvature

Code, saved results and figures for the learnable-curvature paper draft (Bentley Moon, 2026).

In March 2026 each attention head of a small transformer was given one trainable number, kappa (κ),
described as the curvature of that head's representation space. This repository keeps every run
that result rests on, the controls run since, and a script that recomputes every number the paper
cites from those saved runs.

**What survived the controls.** κ is a per-head switch between three attention scoring rules:
plain dot product, a dot product scaled up by the vectors' lengths (called "hyperbolic"), and cosine
similarity (called "spherical"). At d_model 96 with 3 layers the switch beats plain dot-product
attention by 2.41% perplexity on every one of 4 paired seeds. A random fixed assignment keeps all of
that gain, and so does putting every head on the cosine rule from the first step with nothing
learned. Putting every head on the length-scaled rule keeps none of it. So the gain comes from
cosine scoring being present. Learning the curvature is not what produces it. Where the cosine
head sits matters: one fixed cosine head in the first layer keeps 81 to 95% of the gain, and at
6 layers it improves on no curvature on all 4 seeds while learned curvature does not. At twice the width
(d_model 192, 4 heads) the same head mostly loses its gain (-0.00% and +0.26% on 2 seeds, C26), so the
recipe is shown at d_model 96 only. In a trained model
the sign of κ alone carries the whole effect. The full list, with what each claim was tested
against, is in [CLAIMS.md](CLAIMS.md).

**Added 2026-10-23.** Every run above used one constant learning rate, 3·10⁻⁴. A sweep over five
learning rates at d_model 96 with 3 layers (3 seeds each, C27 in CLAIMS.md) reproduces the gain at that
rate (all cosine against the dot product, −2.33%). With each arm at its own best rate the gap is +0.04%:
dot product 4.336 at 2.4·10⁻³, all cosine 4.338 at 1.2·10⁻³. At 4.8·10⁻³ the dot product is better by
1.7% on all 3 seeds. Cosine scoring moves the best learning rate lower and makes runs more consistent
across seeds (sd 0.002 against 0.032 at each arm's best); at matched tuning it does not lower perplexity
at this scale. Learned κ and the 6-layer first-layer recipe were not swept.

**Added 2026-10-28: the cosine rule in a pretrained model.** Every fourth attention layer of
Qwen2.5-1.5B and SmolLM-1.7B was switched to the cosine rule and healed with LoRA and distillation on
1M WikiText-2 tokens (C28-C30). Against the same healing with nothing switched, perplexity rose 0.29%
(Qwen) and 1.73% (SmolLM). Perplexity hid a recall cost: from a list of 64 word-value pairs (about
590 tokens), SmolLM recalled 0.27 against 0.655, and Qwen 0.535 against 0.75. Healed on 512-token
sequences instead of 256, SmolLM recalled 0.65 from 64 pairs and 0.335 against 0.655 from 100 pairs
(about 910 tokens). Longer healing moved the recall limit out; it did not remove it.

## Reproduce

```bash
pip install -r requirements.txt
python reproduce.py check
```

`check` takes seconds and needs no download and no training. It confirms that every result file
matches the SHA-256 recorded when it was copied from the original run (`results/MANIFEST.json`),
rebuilds [CLAIMS.md](CLAIMS.md) from `results/` and fails if a single character differs, and
rebuilds every figure.

To go further than the saved runs:

| Command | What it does | Time |
|---|---|---|
| `python reproduce.py synthetic` | Re-runs the numpy toy (true Poincaré-ball and sphere distances) on seeds 42 and 0 to 9, and confirms the outputs equal the committed ones | about 6 min on 4 cores |
| `python scripts/get_data.py` | Downloads WikiText-2 and checks it is byte-identical to the corpus the runs used | 1 min |
| `python reproduce.py rerun --list` | Lists the d_model 96 runs that can be retrained from scratch | |
| `python reproduce.py rerun allsph_seed0` | Retrains one of them on CPU and compares its perplexity with the committed result | 1 to 1.5 h each |

The March language runs (the 4×2 and 8×4 models, the ablations, the sweep) can be retrained with
`src/learnable_curvature_transformer.py`, `src/random_kappa_ablation.py` and
`src/forced_geometry_suite.py`; their saved configs are inside each result file. They load
WikiText-2 through Hugging Face themselves.

## Layout

| Path | Contents |
|---|---|
| `CLAIMS.md` | Every claim, its number, the file it comes from, the control it faced, and its verdict. Generated; do not edit |
| `figures/` | Nine figures rebuilt from `results/`, with captions in `figures/CAPTIONS.md` |
| `paper/draft.md` | Working draft of the paper |
| `paper/SKELETON.md` | Section headings with the claims and numbers that belong under each |
| `results/` | The saved outputs of every run cited, copied unchanged, with provenance in `MANIFEST.json` |
| `src/learnable_curvature_transformer.py` | The model. `CurvatureDistance` is the scoring rule κ controls |
| `src/learnable_geometry.py` | The numpy toy with its own autograd engine |
| `src/random_kappa_ablation.py`, `src/forced_geometry_suite.py` | The March controls |
| `src/ternary/` | The d_model 96 runs (T0 to T3): continuous, zero, fixed-code, one-bit and ternary-weight arms |
| `src/mandel/` | The separate exact-distance attention model |
| `src/retrofit/exp_retrofit.py` | The cosine rule put into a pretrained model, healed by distillation (C28-C30) |
| `paper/preregistrations/` | The preregistrations for C28-C30, copied unchanged |
| `src/claims.py`, `src/make_figures.py` | Build CLAIMS.md and the figures |
| `scripts/get_data.py` | Corpus download |
| `tools/collect_results.py` | Maintainer tool that copied the results in; paths are the author's machine |

Changes made to copied scripts: file paths now point inside this repository, and the numpy toy
reads its seed from `LG_SEED` (default 42, as originally run). Nothing else was edited.
`src/retrofit/exp_retrofit.py` needs PyTorch, Transformers and PEFT and the cached Hugging Face models;
the runs used a WikiText-2 file with MD5 35b15569176735ed34ae61388bccef63.

## Not included

- The corpora (`scripts/get_data.py` fetches them).
- Model checkpoints. Claim C4's evaluations on the trained 8×4 model (`src/ternary/kappa_quantization.py`,
  `src/ternary/t0b_hard_branch.py`) need that checkpoint (9.7 MB). Point `LC_CKPT` at it if you have it.
- A run for d_model 128 with 1 layer (the standard arm crashed) and a finished d_model 128 with 3 layers.

## Citation

See [CITATION.cff](CITATION.cff). License: MIT.
