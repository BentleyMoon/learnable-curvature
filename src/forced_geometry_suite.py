from __future__ import annotations

import argparse
import json
import os
import statistics
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


def _ensure_output_dir(output_prefix: str):
    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def _load_config_and_data(
    checkpoint_path: str,
    data_path: Optional[str],
    device: torch.device,
    tokenizer: Optional[str],
    spm_model: Optional[str],
) -> Tuple[Dict, torch.Tensor, torch.Tensor, Optional[object]]:
    state = torch.load(checkpoint_path, map_location=device)
    config = dict(state["config"])
    selected_tokenizer = tokenizer or config.get("tokenizer", "char")
    selected_spm_model = spm_model or config.get("spm_model", "wikitext2_bpe.model")
    config["tokenizer"] = selected_tokenizer
    if selected_tokenizer == "bpe":
        config["spm_model"] = selected_spm_model

    vocab_size = int(config.get("vocab_size", 256))
    seq_len = int(config.get("max_seq_len", 64))
    sp_processor = None
    if selected_tokenizer == "bpe":
        train_data, val_data, vocab_size = lct.load_language_data_bpe(
            selected_spm_model,
            seq_len,
            data_path=data_path,
        )
        config["vocab_size"] = vocab_size
        sp_processor = lct._load_sentencepiece_model(selected_spm_model)
    else:
        train_data, val_data = lct.load_language_data(data_path, seq_len, vocab_size=vocab_size)

    return config, train_data, val_data, sp_processor


def _constant_vectors(config: Dict, value: float) -> List[torch.Tensor]:
    n_layers = int(config["n_layers"])
    n_heads = int(config["n_heads"])
    vectors = []
    for _ in range(n_layers):
        vectors.append(torch.full((n_heads,), float(value), dtype=torch.float32))
    return vectors


def _attention_summary(analysis: Dict) -> Dict[str, float]:
    heads = list(analysis.get("heads", {}).values())
    if not heads:
        return {}
    keys = [
        "mean_abs_lag",
        "local4_mass",
        "long16_mass",
        "punct_mass",
        "word_start_mass",
        "positional_score",
        "syntactic_score",
    ]
    summary = {f"mean_{key}": float(statistics.fmean(head[key] for head in heads)) for key in keys}
    summary["head_count"] = len(heads)
    return summary


def _save_checkpoint(output_prefix: str, suffix: str, config: Dict, model: lct.CurvatureTransformer, epoch: int):
    torch.save({"config": config, "model": model.state_dict(), "epoch": epoch}, f"{output_prefix}_{suffix}.pt")


def run_forced_geometry_suite(
    checkpoint_path: str,
    data_path: Optional[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    output_prefix: str,
    save_every: int,
    tokenizer: Optional[str],
    spm_model: Optional[str],
    analysis_batches: int,
    analysis_batch_size: int,
    seed: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    _ensure_output_dir(output_prefix)

    config, train_data, val_data, sp_processor = _load_config_and_data(
        checkpoint_path,
        data_path,
        device,
        tokenizer,
        spm_model,
    )
    print(f"Forced-geometry data: Train={len(train_data)} Val={len(val_data)}")

    init_model = lct.CurvatureTransformer(**lct._model_config(config)).to(device)
    init_state = {name: tensor.detach().clone() for name, tensor in init_model.state_dict().items()}
    del init_model

    suite = {
        "hyperbolic_forced": {
            "kappa_value": -2.5,
            "model": lct.CurvatureTransformer(**lct._model_config(config)).to(device),
        },
        "euclidean_forced": {
            "kappa_value": 0.0,
            "model": lct.CurvatureTransformer(**lct._model_config(config)).to(device),
        },
        "spherical_forced": {
            "kappa_value": 3.0,
            "model": lct.CurvatureTransformer(**lct._model_config(config)).to(device),
        },
    }

    for entry in suite.values():
        model = entry["model"]
        model.load_state_dict(init_state)
        lct._copy_kappa_vectors(model, _constant_vectors(config, entry["kappa_value"]), trainable=False)
        entry["optimizer"] = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )

    results = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": checkpoint_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "analysis_batches": analysis_batches,
            "analysis_batch_size": analysis_batch_size,
            "config": config,
        },
        "hyperbolic_forced": [],
        "euclidean_forced": [],
        "spherical_forced": [],
        "attention_analysis": {},
    }

    print("\n" + "=" * 84)
    print(f"{'Ep':>4} | {'Hyperbolic':>11} {'Euclidean':>11} {'Spherical':>11}")
    print("=" * 84)

    for epoch in range(1, epochs + 1):
        shared_indices = torch.randperm(len(train_data))
        epoch_rows = {}
        for label, entry in suite.items():
            model = entry["model"]
            optimizer = entry["optimizer"]
            _ = lct.train_epoch(model, train_data, optimizer, device, batch_size, indices=shared_indices)
            loss, ppl = lct.evaluate(model, val_data, device, batch_size)
            epoch_rows[label] = {"epoch": epoch, "loss": loss, "ppl": ppl, "kappa_value": entry["kappa_value"]}
            results[label].append(epoch_rows[label])

        print(
            f"{epoch:4d} | {epoch_rows['hyperbolic_forced']['ppl']:11.3f}"
            f" {epoch_rows['euclidean_forced']['ppl']:11.3f} {epoch_rows['spherical_forced']['ppl']:11.3f}"
        )

        if save_every > 0 and epoch % save_every == 0:
            with open(f"{output_prefix}_results.json", "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)
            for label, entry in suite.items():
                _save_checkpoint(output_prefix, label, config, entry["model"], epoch)

    tokenizer_name = config.get("tokenizer", "char")
    for label, entry in suite.items():
        analysis = lct.analyze_attention_patterns(
            model=entry["model"],
            data=val_data,
            device=device,
            batch_size=analysis_batch_size,
            max_batches=analysis_batches,
            tokenizer=tokenizer_name,
            sp_processor=sp_processor,
        )
        results["attention_analysis"][label] = {
            "kappa_value": entry["kappa_value"],
            "summary": _attention_summary(analysis),
            "raw": analysis,
        }

    with open(f"{output_prefix}_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    for label, entry in suite.items():
        _save_checkpoint(output_prefix, label, config, entry["model"], epochs)

    best_scores = {
        label: min(row["ppl"] for row in results[label])
        for label in ("hyperbolic_forced", "euclidean_forced", "spherical_forced")
    }
    print(
        "\nForced geometry best PPL: "
        + ", ".join(f"{label}={score:.3f}" for label, score in best_scores.items())
    )
    print(f"Forced geometry artifacts: {output_prefix}_results.json (+ three checkpoints)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forced-geometry training suite")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--output_prefix", type=str, default="runs/language_20260312_forced_geometry/run")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default=None)
    parser.add_argument("--spm_model", type=str, default=None)
    parser.add_argument("--analysis_batches", type=int, default=64)
    parser.add_argument("--analysis_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_forced_geometry_suite(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        device=torch.device(args.device),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_prefix=args.output_prefix,
        save_every=args.save_every,
        tokenizer=args.tokenizer,
        spm_model=args.spm_model,
        analysis_batches=args.analysis_batches,
        analysis_batch_size=args.analysis_batch_size,
        seed=args.seed,
    )
