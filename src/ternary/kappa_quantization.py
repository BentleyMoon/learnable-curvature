# -*- coding: utf-8 -*-
"""T0 -- does the learned curvature survive as a sign, and do the signs carry information?

Inference only, CPU only. The 4090 belongs to the verifier_frontier daemon; loading models beside it
has hung that daemon once already (DEAD_ENDS G9), and these models are small enough not to need it.

    python kappa_quantization.py --max_val_seqs 2000
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from datetime import datetime

import torch

FRACTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo src/ (originally the author's fractaltest folder)
if FRACTAL not in sys.path:
    sys.path.insert(0, FRACTAL)

import learnable_curvature_transformer as lct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("LC_CKPT", os.path.join(os.path.dirname(FRACTAL), "checkpoints", "phaseB_8h4l_run.pt"))
DATA = os.environ.get("LC_DATA", os.path.join(os.path.dirname(FRACTAL), "data", "wikitext2_all.txt"))


def kappas_of(model):
    return [block.attn.kappa.detach().clone() for block in model.blocks]


def set_kappas(model, values):
    with torch.no_grad():
        for block, v in zip(model.blocks, values):
            block.attn.kappa.copy_(v)


def evaluate(model, val, device, batch_size):
    loss, ppl = lct.evaluate(model, val, device, batch_size=batch_size)
    return float(loss), float(ppl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_val_seqs", type=int, default=2000)
    ap.add_argument("--perm_seeds", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "RESULT_T0_20260920.json"))
    a = ap.parse_args()

    device = torch.device("cpu")
    state = torch.load(a.checkpoint, map_location=device)
    config = dict(state["config"])
    model = lct.CurvatureTransformer(**lct._model_config(config)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    seq_len = int(config.get("max_seq_len", 64))
    vocab = int(config.get("vocab_size", 256))
    _, val = lct.load_language_data(a.data, seq_len, vocab_size=vocab)
    val = val[: a.max_val_seqs]
    print("config: %s" % {k: config.get(k) for k in ("d_model", "n_heads", "n_layers", "max_seq_len")})
    print("val sequences: %d x %d tokens" % (len(val), seq_len))

    base = kappas_of(model)
    flat = torch.cat([k.flatten() for k in base])
    mean_abs = float(flat.abs().mean())
    print("learned kappa: n=%d  mean|k|=%.4f  neg=%d  pos=%d  min=%.3f  max=%.3f"
          % (flat.numel(), mean_abs, int((flat < 0).sum()), int((flat > 0).sum()),
             float(flat.min()), float(flat.max())))

    results = {}

    def run(name, values, extra=None):
        set_kappas(model, values)
        loss, ppl = evaluate(model, val, device, a.batch_size)
        results[name] = {"loss": loss, "ppl": ppl}
        if extra:
            results[name].update(extra)
        ref = results.get("learned", {}).get("ppl")
        delta = "" if not ref else "  (%+.2f%%)" % (100.0 * (ppl - ref) / ref)
        print("  %-22s ppl %8.4f%s" % (name, ppl, delta), flush=True)
        return ppl

    print("\nconditions:")
    run("learned", base)
    run("zero", [torch.zeros_like(k) for k in base])

    scales = [0.25, 0.5, 1.0, round(mean_abs, 4)]
    for s in scales:
        run("sign_%s" % s, [torch.sign(k) * s for k in base])

    for frac in (0.25, 0.5):
        tau = frac * mean_abs
        vals = [torch.sign(k) * mean_abs * (k.abs() > tau).float() for k in base]
        dead = int(sum(int((k.abs() <= tau).sum()) for k in base))
        run("ternary_tau%.2f" % frac, vals, {"tau": tau, "heads_zeroed": dead})

    best_sign = min((results[k]["ppl"], k) for k in results if k.startswith("sign_"))
    s_star = float(best_sign[1].split("_", 1)[1])
    print("  best sign scale: %s (ppl %.4f)" % (s_star, best_sign[0]))

    run("flipped", [-torch.sign(k) * s_star for k in base])
    run("all_hyp", [torch.full_like(k, -s_star) for k in base])
    run("all_sph", [torch.full_like(k, s_star) for k in base])

    # THE CONTROL: the same signs, permuted across heads. Same count of each regime, same magnitude.
    perm_ppls = []
    signs = torch.sign(flat)
    for seed in range(a.perm_seeds):
        g = random.Random(seed)
        order = list(range(signs.numel()))
        g.shuffle(order)
        shuffled = signs[torch.tensor(order)] * s_star
        vals, i = [], 0
        for k in base:
            vals.append(shuffled[i:i + k.numel()].view_as(k).clone())
            i += k.numel()
        perm_ppls.append(run("shuffled_seed%d" % seed, vals))
    mu = sum(perm_ppls) / len(perm_ppls)
    sd = math.sqrt(sum((p - mu) ** 2 for p in perm_ppls) / max(1, len(perm_ppls) - 1))
    results["shuffled_summary"] = {"mean_ppl": mu, "sd": sd, "n": len(perm_ppls),
                                   "min": min(perm_ppls), "max": max(perm_ppls)}

    set_kappas(model, base)
    ref = results["learned"]["ppl"]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prereg": "PREREG_T0_20260920.md",
        "checkpoint": a.checkpoint, "data": a.data,
        "val_sequences": len(val), "seq_len": seq_len, "batch_size": a.batch_size,
        "kappa": {"n": int(flat.numel()), "mean_abs": mean_abs,
                  "neg": int((flat < 0).sum()), "pos": int((flat > 0).sum()),
                  "min": float(flat.min()), "max": float(flat.max()),
                  "per_layer": [[float(x) for x in k] for k in base]},
        "results": results,
        "relative_to_learned_pct": {k: 100.0 * (v["ppl"] - ref) / ref
                                    for k, v in results.items() if "ppl" in v},
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("\nwrote %s" % a.out)

    sign_best = best_sign[0]
    print("\nREADING")
    print("  sign-only vs learned : %+.2f%%" % (100.0 * (sign_best - ref) / ref))
    print("  shuffled vs learned  : %+.2f%% (sd %.4f over %d seeds)" % (100.0 * (mu - ref) / ref, sd,
                                                                        len(perm_ppls)))
    print("  shuffled vs sign-only: %+.2f%%" % (100.0 * (mu - sign_best) / sign_best))
    print("  zero vs learned      : %+.2f%%" % (100.0 * (results["zero"]["ppl"] - ref) / ref))
    return 0


if __name__ == "__main__":
    sys.exit(main())
