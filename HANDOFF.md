# Handoff

## 2026-10-04: T2h folded in; nothing is pending

- T2h's 20 runs are claims C23 to C25 and are in the draft (4.4, abstract, limitations, conclusion).
  Started at the recipe and then learned, the model matches the fixed recipe on 4 of 4 seeds and keeps
  the head cosine; started all hyperbolic it ends below zero on 4 of 4. Fixed all hyperbolic is below
  zero (2 seeds). At 12 layers the recipe helps on both seeds but is under the 0.75% bar on one.
- `PENDING` in `tools/collect_results.py` is empty. `python reproduce.py check` passes.
- What is left is Bentley's: choose the title (CITATION.cff and .zenodo.json still carry the
  descriptive one), read the draft, and make the GitHub release that Zenodo turns into a DOI.

## 2026-09-29: the lab's first-layer results folded in

- Added T2d, T2e (3 layers) and T2f, T2g (6 layers) from `C:\engine\projects\ternary_native` as
  claims C18 to C20 and Figure 9. The recipe the paper can state: one fixed cosine head in the first
  layer. At 6 layers it improves on no curvature on 4 of 4 seeds; learned curvature does not.
- T2i landed the same morning and is in as C21 and C22. Cosine on all 24 heads at 6 layers gives
  +1.06 to +1.49%, and one first-layer cosine head gets about the same. Layer 1 beat cosine everywhere
  on 4 of 4 seeds, but under the preregistered bar, so the paper does not claim the first layer adds
  anything beyond cosine attention.
- Left out until it finishes on the lab's lane: T2h (`PENDING` in `tools/collect_results.py`). When
  it lands: remove it from `PENDING`, run `python tools/collect_results.py`, add its claim in
  `src/claims.py`, then `python reproduce.py claims` and `python reproduce.py check`.
- Related work in `paper/SKELETON.md` was checked against primary sources. Per-head learnable
  curvature is prior art (Cho et al., 2023).
- Still no GitHub repository. The three publishing acts remain Bentley's.

## 2026-09-23: repository created

**What exists now**

- `CLAIMS.md`: 19 claims (24 after 29 Sept) and 11 withdrawn draft claims, each computed from `results/` by `src/claims.py`.
- `figures/`: 8 figures rebuilt from `results/`.
- `paper/SKELETON.md`: section headings with claims and numbers. No publication prose; that is Bentley's.
- Verified on a fresh clone of this repository:
  - `python reproduce.py check` passes: 167 result files match their hashes, every CLAIMS.md
    number rebuilds, all 8 figures rebuild.
  - `python reproduce.py synthetic` re-ran the numpy toy on 11 seeds: all identical to the committed
    results, and seed 42 is identical to the March result file.
  - `python scripts/get_data.py` rebuilt WikiText-2 byte for byte (SHA-256 1022cd9a...).
  - A 30-step training smoke run (all heads cosine, seed 0) matched the lab's original run's first
    training loss to every digit (5.543328762054443).

**Not done, and why**

- No d_model 96 retrain from this repository. Memory commit was below the lab lane's 16 GB gate and
  shoofly had a run in flight; the lab's own T2d was queued behind the same gate. R1 in CLAIMS.md
  records that the lab's separate retrain matched T1b on 4 of 4 seeds. When the machine is free:
  `python scripts/get_data.py` then `python reproduce.py rerun allsph_seed0`.
- No GitHub repository, push, release or Zenodo upload. Those are Bentley's acts.

**If T2d lands** (the lab's sweep of how many cosine heads are needed): run
`python tools/collect_results.py`, add a claim for it in `src/claims.py`, then
`python reproduce.py claims` and `python reproduce.py check`.

**Before the release:** the title in `CITATION.cff` and `.zenodo.json` is descriptive, not the
paper's title. Set both to the paper's title once it is chosen.
