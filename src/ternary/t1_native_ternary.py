# -*- coding: utf-8 -*-
"""T1 -- train with the curvature in its discrete form from the start, against continuous curvature.

T0 established that a TRAINED model's 32 curvatures collapse to their signs at no cost, that the sign
assignment is load-bearing (permuting it: +12.79%), and that the learned alphabet is binary -- the
dead zone never fired. T1 asks the forward question: if the discrete code is the native format during
training, does the model reach the same place, and does a zero (Euclidean) regime get USED when it is
reachable?

Conditions, identical except for how a head's curvature is formed:
    continuous  kappa as a free scalar (the original)
    binary      kappa_eff = sign(kappa_latent), straight-through estimator
    ternary     kappa_eff = sign(kappa_latent) * 1[|kappa_latent| > tau], straight-through
    zero        kappa fixed at 0 -- no geometry (control)
    randsign    kappa fixed at random +/-1 -- geometry, but not a learned assignment (control)

Every condition starts from the same weights and the same small latent-kappa noise for its seed, so
the only difference is the curvature parameterisation. CPU only: the 4090 belongs to the verifier lane.

    python t1_native_ternary.py --condition binary --seed 0 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

FRACTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo src/ (originally the author's fractaltest folder)
if FRACTAL not in sys.path:
    sys.path.insert(0, FRACTAL)

import learnable_curvature_transformer as lct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("LC_DATA", os.path.join(os.path.dirname(FRACTAL), "data", "wikitext2_all.txt"))
CONDITIONS = ("continuous", "binary", "ternary", "zero", "randsign", "late_binary", "fixedcode")

# The training step, visible to the late_binary parameterisation: continuous until the switch, one bit
# after it. T1b found native binary keeps about a fifth of the curvature's gain at d_model 96 while T0
# found post-hoc binarisation keeps all of it; late_binary asks whether the path between them does.
STEP = {"now": 0, "switch": None}


def ste(value, quantised):
    """Forward: the quantised value. Backward: identity, so the latent keeps learning."""
    return value + (quantised - value).detach()


def install(model, condition, tau, seed, code=None):
    """Replace each head's effective_kappa according to the condition. Returns fixed params to freeze.

    fixedcode (T2b): kappa frozen at +/-1 from step 1 to a GIVEN sign code ("-+--..." one char per head,
    block by block) -- the oracle arm: hand the model the code a continuous run found, and ask whether
    the code is all the geometry is.
    """
    frozen = []
    g = torch.Generator().manual_seed(10_000 + seed)
    if condition == "fixedcode":
        n = sum(b.attn.kappa.numel() for b in model.blocks)
        if not code or len(code) != n or set(code) - {"+", "-"}:
            raise SystemExit("fixedcode needs --code with exactly %d characters of + and -" % n)
        it = iter(code)
        for block in model.blocks:
            with torch.no_grad():
                block.attn.kappa.copy_(torch.tensor([1.0 if next(it) == "+" else -1.0
                                                     for _ in range(block.attn.kappa.numel())]))
            block.attn.kappa.requires_grad_(False)
            frozen.append(block.attn.kappa)
        return frozen
    for block in model.blocks:
        attn = block.attn
        if condition == "continuous":
            continue
        if condition == "binary":
            attn.effective_kappa = types.MethodType(lambda self: ste(self.kappa, torch.sign(self.kappa)), attn)
        elif condition == "late_binary":
            attn.effective_kappa = types.MethodType(
                lambda self: self.kappa if STEP["now"] < STEP["switch"]
                else ste(self.kappa, torch.sign(self.kappa)), attn)
        elif condition == "ternary":
            attn.effective_kappa = types.MethodType(
                lambda self: ste(self.kappa, torch.sign(self.kappa) * (self.kappa.abs() > tau).float()), attn)
        elif condition == "zero":
            with torch.no_grad():
                attn.kappa.zero_()
            attn.kappa.requires_grad_(False)
            frozen.append(attn.kappa)
        elif condition == "randsign":
            with torch.no_grad():
                signs = torch.randint(0, 2, attn.kappa.shape, generator=g).float() * 2 - 1
                attn.kappa.copy_(signs)
            attn.kappa.requires_grad_(False)
            frozen.append(attn.kappa)
    return frozen


def load_compact(path, seq_len, vocab_size=256):
    """lct.load_language_data's local-file branch, value for value, at a fraction of the memory.

    2026-09-21: seven of these runs committed ~2.1 GB each and, together with ollama's model servers,
    exhausted the machine's commit; one spike killed all seven at 4484 s and a shoofly arm died on a
    9.7 MB allocation. The original builds a 13-million-element Python list and int64 tensors for values
    that are all below 256. Here the training tokens stay uint8 and are widened per batch (see the
    training loop); validation is widened once, because lct.evaluate indexes it directly. Same text,
    same modulo, same view and split, so every batch the model sees is identical.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    arr = np.fromiter((ord(c) % vocab_size for c in text), dtype=np.uint8, count=len(text))
    del text
    tokens = torch.from_numpy(arr)
    n = len(tokens) // seq_len * seq_len
    tokens = tokens[:n].view(-1, seq_len)
    split = int(0.9 * len(tokens))
    return tokens[:split], tokens[split:].long()


def ternary_linear_forward(self, x):
    """BitNet b1.58's weight quantiser: W_q = RoundClip(W / mean|W|, -1, 1) * mean|W|, straight-through.

    Weights only (activations stay float), biases kept. The latent float weights keep learning through
    the STE; every forward pass sees only {-s, 0, +s} per matrix.
    """
    w = self.weight
    s = w.abs().mean().clamp(min=1e-5)
    wq = (w / s).round().clamp(-1, 1) * s
    return F.linear(x, ste(w, wq), self.bias)


def install_ternary_weights(model):
    """Every Linear inside the transformer blocks goes ternary; embeddings and the LM head stay float
    (the BitNet convention). Returns the patched layers so their sparsity can be read after training."""
    patched = []
    for block in model.blocks:
        for mod in block.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.forward = types.MethodType(ternary_linear_forward, mod)
                patched.append(mod)
    return patched


def weight_sparsity(patched):
    """Fraction of block weights that quantise to exactly zero -- free sparsity a kernel could skip."""
    if not patched:
        return None
    zeros = total = 0
    with torch.no_grad():
        for mod in patched:
            w = mod.weight
            q = (w / w.abs().mean().clamp(min=1e-5)).round().clamp(-1, 1)
            zeros += int((q == 0).sum())
            total += q.numel()
    return zeros / total


def regimes(model, condition, tau):
    out = {"neg": 0, "zero": 0, "pos": 0, "latent": []}
    for block in model.blocks:
        k = block.attn.kappa.detach()
        eff = block.attn.effective_kappa().detach()
        out["latent"].append([round(float(x), 4) for x in k])
        for v in eff:
            v = float(v)
            if v < -1e-6:
                out["neg"] += 1
            elif v > 1e-6:
                out["pos"] += 1
            else:
                out["zero"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_kappa", type=float, default=1e-2)
    ap.add_argument("--val_seqs", type=int, default=2500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "t1_runs"))
    ap.add_argument("--switch_frac", type=float, default=0.75,
                    help="late_binary: fraction of training spent continuous before switching to one bit")
    ap.add_argument("--posthoc", action="store_true",
                    help="after training, also evaluate the model with every curvature replaced by its sign")
    ap.add_argument("--code", default=None, help="fixedcode: the sign code, e.g. -+----++--+-")
    ap.add_argument("--sign_every", type=int, default=0,
                    help="record every head's sign(latent kappa) every N steps (read-only; 0 = off)")
    ap.add_argument("--weights", choices=("float", "ternary"), default="float",
                    help="ternary: every block Linear uses BitNet b1.58 absmean ternary weights (STE)")
    a = ap.parse_args()
    STEP["switch"] = int(a.switch_frac * a.steps)

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    seq_len = 64
    config = {"vocab_size": 256, "d_model": a.d_model, "n_heads": a.n_heads, "n_layers": a.n_layers,
              "d_ff": 4 * a.d_model, "max_seq_len": seq_len, "kappa_min": None, "kappa_max": None}
    model = lct.CurvatureTransformer(**lct._model_config(config))
    # the same small latent noise in every condition, so binary/ternary are not stuck at sign(0) = 0
    with torch.no_grad():
        g = torch.Generator().manual_seed(a.seed)
        for block in model.blocks:
            block.attn.kappa.copy_((torch.rand(block.attn.kappa.shape, generator=g) - 0.5) * 0.02)
    frozen = install(model, a.condition, a.tau, a.seed, code=a.code)
    patched = install_ternary_weights(model) if a.weights == "ternary" else []

    train, val = load_compact(DATA, seq_len, vocab_size=256)
    val = val[: a.val_seqs]
    kappa_params = [b.attn.kappa for b in model.blocks if b.attn.kappa.requires_grad]
    kappa_ids = {id(p) for p in kappa_params}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in kappa_ids]
    groups = [{"params": other, "lr": a.lr}]
    if kappa_params:
        groups.append({"params": kappa_params, "lr": a.lr_kappa})
    opt = torch.optim.AdamW(groups)

    g = torch.Generator().manual_seed(a.seed)
    n_train = len(train)
    model.train()
    t0 = time.time()
    log = []
    sign_log = []           # (step, "+-+-..."): when is each head's sign decided? (T2b)
    for step in range(1, a.steps + 1):
        STEP["now"] = step
        idx = torch.randint(0, n_train, (a.batch_size,), generator=g)
        batch = train[idx].long()          # uint8 storage, int64 for the model (load_compact)
        logits, _ = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if a.sign_every and (step % a.sign_every == 0 or step == 1):
            with torch.no_grad():
                sign_log.append([step, "".join("+" if float(v) > 0 else "-" if float(v) < 0 else "0"
                                               for b in model.blocks for v in b.attn.kappa)])
        if step % max(1, a.steps // 10) == 0 or step == 1:
            log.append({"step": step, "loss": float(loss), "elapsed": time.time() - t0})
            print("  step %5d  loss %.4f  %.1fs" % (step, float(loss), time.time() - t0), flush=True)

    model.eval()
    STEP["now"] = a.steps + 1           # evaluation happens after the switch, whatever the condition
    vloss, vppl = lct.evaluate(model, val, torch.device("cpu"), batch_size=32)
    reg = regimes(model, a.condition, a.tau)
    posthoc = {}
    if a.posthoc:
        # T0's operation at THIS scale: replace each trained curvature by its sign, at scale 1 and at
        # the model's own mean |kappa|, and re-evaluate the same weights. No retraining.
        saved = [b.attn.kappa.detach().clone() for b in model.blocks]
        mean_abs = float(torch.cat([k.flatten() for k in saved]).abs().mean())
        for label, scale in (("sign_1", 1.0), ("sign_meanabs", mean_abs)):
            with torch.no_grad():
                for b, k in zip(model.blocks, saved):
                    b.attn.kappa.copy_(torch.sign(k) * scale)
            _, p = lct.evaluate(model, val, torch.device("cpu"), batch_size=32)
            posthoc[label] = float(p)
        with torch.no_grad():
            for b, k in zip(model.blocks, saved):
                b.attn.kappa.copy_(k)
        posthoc["mean_abs_kappa"] = mean_abs
    os.makedirs(a.out_dir, exist_ok=True)
    rec = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "prereg": "PREREG_T1_20260921.md", "condition": a.condition, "seed": a.seed,
           "config": config, "steps": a.steps, "batch_size": a.batch_size, "tau": a.tau,
           "lr": a.lr, "lr_kappa": a.lr_kappa, "val_seqs": len(val),
           "val_loss": float(vloss), "val_ppl": float(vppl), "regimes": reg, "posthoc": posthoc,
           "switch_step": STEP["switch"] if a.condition == "late_binary" else None,
           "weights": a.weights, "weight_sparsity": weight_sparsity(patched), "sign_log": sign_log,
           "code": a.code,
           "train_seconds": time.time() - t0, "train_log": log,
           "n_params": sum(p.numel() for p in model.parameters())}
    path = os.path.join(a.out_dir, "%s_seed%d.json" % (a.condition, a.seed))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("DONE %s seed %d: val ppl %.4f  regimes neg/zero/pos %d/%d/%d  %.0fs  -> %s"
          % (a.condition, a.seed, vppl, reg["neg"], reg["zero"], reg["pos"], rec["train_seconds"], path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
