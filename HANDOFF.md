# Handoff

## 2026-09-23: repository created

**Old sessions retired:** none. The session brief listed no old sessions under "Old sessions", so
nothing was stopped or archived.

**What exists now**

- `CLAIMS.md`: 19 claims and 11 withdrawn draft claims, each computed from `results/` by `src/claims.py`.
- `figures/`: 8 figures rebuilt from `results/`.
- `paper/SKELETON.md`: section headings with claims and numbers. No publication prose; that is Bentley's.
- `python reproduce.py check` passes on a clean clone (file hashes, every CLAIMS.md number, figures).
- `python reproduce.py synthetic` re-ran the numpy toy on 11 seeds; seed 42 reproduces the March
  result file exactly.

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
