"""T2k: is the cosine-scoring gain robust to tuning the dot-product baseline's learning rate?

Uses this repository's model, condition installer and data loader unchanged, only moving the training loop
to CUDA. Mirrors src/ternary/t1_native_ternary.py main(). One run takes about 14 minutes on an RTX 4090 when
six share the card; more than six at once ran out of GPU memory.

PRE-REGISTRATION (written before any run, 2026-10-21):
  arms: zero (kappa=0, dot product) vs allsph (all 12 heads fixed kappa=+1, cosine * sqrt(d_k))
  lr:   3e-4 (paper), 6e-4, 1.2e-3, 2.4e-3   seeds: 0,1,2   d96 / 4 heads / 3 layers / 20k steps / batch 64
  H_robust: allsph beats zero at every lr AND best(allsph) beats best(zero) by >= 1%.
  H_lr:     best-vs-best gap < 0.5%, i.e. the paper's 2.4% is mostly a fixed-lr effect.
  Between: "narrowed".
ADDENDUM (before the arm ran): lr 4.8e-3 x 3 seeds added because the dot product's best mean sat at the grid
  edge (2.4e-3); hypotheses unchanged.

    python src/ternary/t2k_lr_sweep_gpu.py --arm zero --lr 2.4e-3 --seed 0 --out results/ternary_lane/t2k_lr_runs/zero_lr2.4e-3_seed0.json
"""
import argparse, json, os, sys, time
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import learnable_curvature_transformer as lct  # noqa
import t1_native_ternary as t1  # noqa

ap = argparse.ArgumentParser()
ap.add_argument('--arm', choices=['zero', 'allsph', 'allhyp'], required=True)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=20000)
ap.add_argument('--out', required=True)
a = ap.parse_args()

dev = torch.device('cuda')
torch.manual_seed(a.seed)
seq_len, d = 64, 96
config = {"vocab_size": 256, "d_model": d, "n_heads": 4, "n_layers": 3, "d_ff": 4 * d,
          "max_seq_len": seq_len, "kappa_min": None, "kappa_max": None}
model = lct.CurvatureTransformer(**lct._model_config(config))
with torch.no_grad():
    g = torch.Generator().manual_seed(a.seed)
    for block in model.blocks:
        block.attn.kappa.copy_((torch.rand(block.attn.kappa.shape, generator=g) - 0.5) * 0.02)
cond, code = ('zero', None) if a.arm == 'zero' else ('fixedcode', ('+' if a.arm == 'allsph' else '-') * 12)
t1.install(model, cond, 0.1, a.seed, code=code)
model.to(dev)

train, val = t1.load_compact(t1.DATA, seq_len, vocab_size=256)
val = val[:2500]
kappa_params = [b.attn.kappa for b in model.blocks if b.attn.kappa.requires_grad]
ids = {id(p) for p in kappa_params}
other = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
opt = torch.optim.AdamW([{"params": other, "lr": a.lr}] + ([{"params": kappa_params, "lr": 1e-2}] if kappa_params else []))

g = torch.Generator().manual_seed(a.seed)
n = len(train); t0 = time.time(); log = []; nan = False
model.train()
for step in range(1, a.steps + 1):
    t1.STEP["now"] = step
    idx = torch.randint(0, n, (64,), generator=g)
    batch = train[idx].long().to(dev)
    logits, _ = model(batch[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
    if not torch.isfinite(loss):
        nan = True; break
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 2000 == 0 or step == 1:
        log.append({"step": step, "loss": float(loss)})
model.eval()
if nan:
    vloss, vppl = float('nan'), float('nan')
else:
    vloss, vppl = lct.evaluate(model, val, dev, batch_size=32)
rec = dict(arm=a.arm, lr=a.lr, seed=a.seed, steps=a.steps, val_loss=float(vloss), val_ppl=float(vppl), diverged=nan,
           train_log=log, seconds=time.time() - t0)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(rec, open(a.out, 'w'), indent=1)
print(f"DONE {a.arm} lr={a.lr} seed={a.seed} ppl={vppl:.4f} {time.time()-t0:.0f}s", flush=True)
