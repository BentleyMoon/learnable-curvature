# -*- coding: utf-8 -*-
"""T0b -- if the code is one bit per head, can each head compute ONE metric instead of blending three?

T0 showed the 32 learned curvatures collapse to their signs at no cost (3.7456 -> 3.7455) and that the
assignment is load-bearing (permuting the signs costs +12.79% over 10 seeds). That is a statement
about the PARAMETER. This is the statement about the COMPUTATION.

The trained scorer blends three terms per head:

    scores = w_e(k)*dot + w_h(k)*[dot amplified by norms] + w_s(k)*[cosine rescaled]

with w_h = max(0, tanh(-3k)), w_s = max(0, tanh(3k)), w_e = 1 - w_h - w_s. At |k| >= 1 those weights
are already about 0.995 / 0.005, so the blend is nearly a selection -- but "nearly" is what gets
measured here, not assumed. Hard routing computes only the branch the sign selects, which is what
lets a deployment skip the other terms entirely (the norm product for a spherical head, the
normalisation for a hyperbolic one).

Inference only, CPU only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F

FRACTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo src/ (originally the author's fractaltest folder)
if FRACTAL not in sys.path:
    sys.path.insert(0, FRACTAL)

import learnable_curvature_transformer as lct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("LC_CKPT", os.path.join(os.path.dirname(FRACTAL), "checkpoints", "phaseB_8h4l_run.pt"))
DATA = os.environ.get("LC_DATA", os.path.join(os.path.dirname(FRACTAL), "data", "wikitext2_all.txt"))

_BLENDED = lct.CurvatureDistance.forward


def hard_forward(self, q, k, kappa):
    """One branch per head, selected by the sign bit. No blending, no unused terms."""
    d_k = q.size(-1)
    sign = torch.sign(kappa).view(-1, 1, 1)                       # [H,1,1] in {-1,0,+1}
    dot = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    out = dot.clone()                                             # kappa == 0 -> Euclidean
    hyp = sign < 0
    sph = sign > 0
    if bool(hyp.any()):
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        norm_product = q_norm * k_norm.transpose(-2, -1)
        hyp_scores = dot * (1.0 + 0.5 * norm_product / (d_k ** 0.5))
        out = torch.where(hyp, hyp_scores, out)
    if bool(sph.any()):
        cos = torch.matmul(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-2, -1))
        out = torch.where(sph, cos * (d_k ** 0.5), out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_val_seqs", type=int, default=2500)
    ap.add_argument("--perm_seeds", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "RESULT_T0B_20260920.json"))
    a = ap.parse_args()

    device = torch.device("cpu")
    state = torch.load(CKPT, map_location=device)
    config = dict(state["config"])
    model = lct.CurvatureTransformer(**lct._model_config(config)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    seq_len = int(config.get("max_seq_len", 64))
    _, val = lct.load_language_data(DATA, seq_len, vocab_size=int(config.get("vocab_size", 256)))
    val = val[: a.max_val_seqs]

    base = [b.attn.kappa.detach().clone() for b in model.blocks]
    flat = torch.cat([k.flatten() for k in base])

    def set_k(values):
        with torch.no_grad():
            for b, v in zip(model.blocks, values):
                b.attn.kappa.copy_(v)

    def run(label, values, hard):
        set_k(values)
        lct.CurvatureDistance.forward = hard_forward if hard else _BLENDED
        t0 = time.time()
        loss, ppl = lct.evaluate(model, val, device, batch_size=a.batch_size)
        dt = time.time() - t0
        print("  %-24s ppl %8.4f   %5.1fs" % (label, ppl, dt), flush=True)
        return {"ppl": float(ppl), "loss": float(loss), "seconds": dt}

    print("val sequences: %d; heads: %d (neg %d / pos %d)"
          % (len(val), flat.numel(), int((flat < 0).sum()), int((flat > 0).sum())))
    print("\nconditions:")
    res = {}
    res["blended_learned"] = run("blended, learned k", base, hard=False)
    res["hard_learned"] = run("HARD, learned signs", base, hard=True)
    res["hard_sign1"] = run("HARD, k -> sign(k)", [torch.sign(k) for k in base], hard=True)

    perms = []
    signs = torch.sign(flat)
    for seed in range(a.perm_seeds):
        g = random.Random(1000 + seed)
        order = list(range(signs.numel()))
        g.shuffle(order)
        sh = signs[torch.tensor(order)]
        vals, i = [], 0
        for k in base:
            vals.append(sh[i:i + k.numel()].view_as(k).clone())
            i += k.numel()
        perms.append(run("HARD, shuffled s%d" % seed, vals, hard=True)["ppl"])
    mu = sum(perms) / len(perms)
    sd = math.sqrt(sum((p - mu) ** 2 for p in perms) / max(1, len(perms) - 1))
    res["hard_shuffled"] = {"mean_ppl": mu, "sd": sd, "n": len(perms),
                            "min": min(perms), "max": max(perms)}

    lct.CurvatureDistance.forward = _BLENDED
    set_k(base)
    ref = res["blended_learned"]["ppl"]
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "prereg": "PREREG_T0_20260920.md", "checkpoint": CKPT,
               "val_sequences": len(val), "results": res,
               "relative_to_blended_pct": {
                   "hard_learned": 100.0 * (res["hard_learned"]["ppl"] - ref) / ref,
                   "hard_sign1": 100.0 * (res["hard_sign1"]["ppl"] - ref) / ref,
                   "hard_shuffled_mean": 100.0 * (mu - ref) / ref}}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print("\nREADING")
    print("  hard vs blended      : %+.2f%%" % payload["relative_to_blended_pct"]["hard_learned"])
    print("  hard sign-only       : %+.2f%%" % payload["relative_to_blended_pct"]["hard_sign1"])
    print("  hard shuffled        : %+.2f%% (sd %.4f)" % (payload["relative_to_blended_pct"]["hard_shuffled_mean"], sd))
    print("  wall clock blended   : %.1fs" % res["blended_learned"]["seconds"])
    print("  wall clock hard      : %.1fs" % res["hard_learned"]["seconds"])
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
