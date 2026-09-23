from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR  # was os.path.dirname(SCRIPT_DIR) when this file lived in runs/
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import learnable_curvature_transformer as lct


def _load_config_and_data(
    checkpoint_path: str,
    data_path: Optional[str],
    device: torch.device,
    tokenizer: Optional[str],
    spm_model: Optional[str],
) -> Tuple[Dict, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    state = torch.load(checkpoint_path, map_location=device)
    config = dict(state["config"])
    selected_tokenizer = tokenizer or config.get("tokenizer", "char")
    selected_spm_model = spm_model or config.get("spm_model", "wikitext2_bpe.model")
    config["tokenizer"] = selected_tokenizer
    if selected_tokenizer == "bpe":
        config["spm_model"] = selected_spm_model

    reference_model = lct.CurvatureTransformer(**lct._model_config(config)).to(device)
    reference_model.load_state_dict(state["model"])
    fixed_vectors = lct._kappa_vectors_from_model(reference_model)

    vocab_size = int(config.get("vocab_size", 256))
    seq_len = int(config.get("max_seq_len", 64))
    if selected_tokenizer == "bpe":
        train_data, val_data, vocab_size = lct.load_language_data_bpe(
            selected_spm_model,
            seq_len,
            data_path=data_path,
        )
        config["vocab_size"] = vocab_size
    else:
        train_data, val_data = lct.load_language_data(data_path, seq_len, vocab_size=vocab_size)

    return config, fixed_vectors, train_data, val_data


def _random_vectors_from_reference(reference_vectors: List[torch.Tensor], seed: int) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    flat = torch.cat([vec.detach().cpu().flatten().float() for vec in reference_vectors])
    lo = float(flat.min().item())
    hi = float(flat.max().item())
    rng = np.random.default_rng(seed)
    vectors: List[torch.Tensor] = []
    for ref in reference_vectors:
        if hi > lo:
            values = rng.uniform(lo, hi, size=ref.numel()).astype(np.float32)
        else:
            values = np.full(ref.numel(), lo, dtype=np.float32)
        vectors.append(torch.tensor(values, dtype=ref.dtype).view_as(ref))
    meta = {
        "seed": seed,
        "distribution": "uniform_global_range",
        "range_min": lo,
        "range_max": hi,
    }
    return vectors, meta


def _ensure_output_dir(output_prefix: str):
    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def run_random_kappa_ablation(
    checkpoint_path: str,
    data_path: Optional[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_kappa: float,
    output_prefix: str,
    save_every: int,
    tokenizer: Optional[str],
    spm_model: Optional[str],
    seed: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    _ensure_output_dir(output_prefix)

    config, fixed_vectors, train_data, val_data = _load_config_and_data(
        checkpoint_path,
        data_path,
        device,
        tokenizer,
        spm_model,
    )
    zero_vectors = [torch.zeros_like(vec) for vec in fixed_vectors]
    random_vectors, random_meta = _random_vectors_from_reference(fixed_vectors, seed)

    print(f"Ablation data: Train={len(train_data)} Val={len(val_data)}")
    init_model = lct.CurvatureTransformer(**lct._model_config(config)).to(device)
    init_state = {name: tensor.detach().clone() for name, tensor in init_model.state_dict().items()}
    del init_model

    learned_model = lct.CurvatureTransformer(**lct._model_config(config)).to(device)
    fixed_model = lct.CurvatureTransformer(**lct._model_config(config)).to(device)
    zero_model = lct.CurvatureTransformer(**lct._model_config(config)).to(device)
    random_model = lct.CurvatureTransformer(**lct._model_config(config)).to(device)
    for model in (learned_model, fixed_model, zero_model, random_model):
        model.load_state_dict(init_state)

    lct._copy_kappa_vectors(fixed_model, fixed_vectors, trainable=False)
    lct._copy_kappa_vectors(zero_model, zero_vectors, trainable=False)
    lct._copy_kappa_vectors(random_model, random_vectors, trainable=False)

    learned_kappa_params = [block.attn.kappa for block in learned_model.blocks]
    learned_other = [param for name, param in learned_model.named_parameters() if "kappa" not in name]
    learned_opt = torch.optim.AdamW(
        [
            {"params": learned_other, "lr": lr},
            {"params": learned_kappa_params, "lr": lr_kappa},
        ],
        weight_decay=0.01,
    )

    fixed_opt = torch.optim.AdamW([p for p in fixed_model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    zero_opt = torch.optim.AdamW([p for p in zero_model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    random_opt = torch.optim.AdamW([p for p in random_model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)

    result = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": checkpoint_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lr_kappa": lr_kappa,
            "seed": seed,
            "config": config,
            "random_control": random_meta,
        },
        "learned_kappa": [],
        "fixed_kappa": [],
        "zero_kappa": [],
        "random_kappa": [],
        "reference_kappa": {
            f"layer_{idx}": [float(x) for x in vec.tolist()]
            for idx, vec in enumerate(fixed_vectors)
        },
        "random_reference_kappa": {
            f"layer_{idx}": [float(x) for x in vec.tolist()]
            for idx, vec in enumerate(random_vectors)
        },
    }

    print("\n" + "=" * 108)
    print(f"{'Ep':>4} | {'Learned':>8} {'Fixed':>8} {'Zero':>8} {'Random':>8} | {'Learned κ-range':>15}")
    print("=" * 108)

    for epoch in range(1, epochs + 1):
        shared_indices = torch.randperm(len(train_data))

        _ = lct.train_epoch(learned_model, train_data, learned_opt, device, batch_size, indices=shared_indices)
        _ = lct.train_epoch(fixed_model, train_data, fixed_opt, device, batch_size, indices=shared_indices)
        _ = lct.train_epoch(zero_model, train_data, zero_opt, device, batch_size, indices=shared_indices)
        _ = lct.train_epoch(random_model, train_data, random_opt, device, batch_size, indices=shared_indices)

        l_loss, l_ppl = lct.evaluate(learned_model, val_data, device, batch_size)
        f_loss, f_ppl = lct.evaluate(fixed_model, val_data, device, batch_size)
        z_loss, z_ppl = lct.evaluate(zero_model, val_data, device, batch_size)
        r_loss, r_ppl = lct.evaluate(random_model, val_data, device, batch_size)

        learned_curv = learned_model.get_all_curvatures()
        learned_kappas = [info["kappa"] for info in learned_curv.values()]
        learned_range = max(learned_kappas) - min(learned_kappas)

        result["learned_kappa"].append({
            "epoch": epoch,
            "loss": l_loss,
            "ppl": l_ppl,
            "kappas": {key: value["kappa"] for key, value in learned_curv.items()},
        })
        result["fixed_kappa"].append({"epoch": epoch, "loss": f_loss, "ppl": f_ppl})
        result["zero_kappa"].append({"epoch": epoch, "loss": z_loss, "ppl": z_ppl})
        result["random_kappa"].append({"epoch": epoch, "loss": r_loss, "ppl": r_ppl})

        print(f"{epoch:4d} | {l_ppl:8.3f} {f_ppl:8.3f} {z_ppl:8.3f} {r_ppl:8.3f} | {learned_range:15.4f}")

        if save_every > 0 and epoch % save_every == 0:
            with open(f"{output_prefix}_ablation_results.json", "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            torch.save({"config": config, "model": learned_model.state_dict(), "epoch": epoch}, f"{output_prefix}_ablation_learned.pt")
            torch.save({"config": config, "model": fixed_model.state_dict(), "epoch": epoch}, f"{output_prefix}_ablation_fixed.pt")
            torch.save({"config": config, "model": zero_model.state_dict(), "epoch": epoch}, f"{output_prefix}_ablation_zero.pt")
            torch.save({"config": config, "model": random_model.state_dict(), "epoch": epoch}, f"{output_prefix}_ablation_random.pt")

    with open(f"{output_prefix}_ablation_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    torch.save({"config": config, "model": learned_model.state_dict(), "epoch": epochs}, f"{output_prefix}_ablation_learned.pt")
    torch.save({"config": config, "model": fixed_model.state_dict(), "epoch": epochs}, f"{output_prefix}_ablation_fixed.pt")
    torch.save({"config": config, "model": zero_model.state_dict(), "epoch": epochs}, f"{output_prefix}_ablation_zero.pt")
    torch.save({"config": config, "model": random_model.state_dict(), "epoch": epochs}, f"{output_prefix}_ablation_random.pt")

    best_learned = min(row["ppl"] for row in result["learned_kappa"])
    best_fixed = min(row["ppl"] for row in result["fixed_kappa"])
    best_zero = min(row["ppl"] for row in result["zero_kappa"])
    best_random = min(row["ppl"] for row in result["random_kappa"])
    print(
        f"\nAblation best PPL: learned={best_learned:.3f} fixed={best_fixed:.3f} zero={best_zero:.3f} random={best_random:.3f}"
    )
    print(f"Ablation artifacts: {output_prefix}_ablation_results.json (+ four checkpoints)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Four-arm kappa ablation with random fixed control")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_kappa", type=float, default=1e-2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--output_prefix", type=str, default="runs/language_20260312_ablation_random/run")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default=None)
    parser.add_argument("--spm_model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_random_kappa_ablation(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        device=torch.device(args.device),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_kappa=args.lr_kappa,
        output_prefix=args.output_prefix,
        save_every=args.save_every,
        tokenizer=args.tokenizer,
        spm_model=args.spm_model,
        seed=args.seed,
    )
