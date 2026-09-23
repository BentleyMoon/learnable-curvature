from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

import exact_geometry_transformer as egt


VARIANTS = [
    {"label": "standard_dot", "model_variant": "standard", "head_types": None},
    {"label": "exact_euclidean", "model_variant": "exact_euclidean", "head_types": None},
    {"label": "exact_mixed", "model_variant": "exact", "head_types": None},
]
TASKS = ["hierarchy", "periodic", "mixed"]


def _base_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "mode": "synthetic",
        "model_variant": "exact",
        "synthetic_task": "combined",
        "data": None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "lr": args.lr,
        "head_types": None,
        "init_c": args.init_c,
        "init_tau": args.init_tau,
        "learnable_c": args.learnable_c,
        "freeze_tau": args.freeze_tau,
        "tangent_clip": args.tangent_clip,
        "save_every": 0,
        "output_prefix": None,
        "seed": args.seed,
        "device": args.device,
    }


def write_markdown(path: str, payload: Dict[str, Any]):
    lines: List[str] = []
    lines.append("# Exact Geometry Control Matrix")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append("")
    lines.append("| Task | Variant | Final PPL | Best PPL |")
    lines.append("|---|---|---:|---:|")
    for run in payload["runs"]:
        summary = run["summary"]
        lines.append(
            f"| {run['task']} | {run['variant']} | {summary['final_val_ppl']:.3f} | {summary['best_val_ppl']:.3f} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the standard/exact-euclidean/exact-mixed synthetic matrix")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--train_samples", type=int, default=512)
    parser.add_argument("--val_samples", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init_c", type=float, default=1.0)
    parser.add_argument("--init_tau", type=float, default=1.0)
    parser.add_argument("--learnable_c", action="store_true")
    parser.add_argument("--freeze_tau", action="store_true")
    parser.add_argument("--tangent_clip", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda" if __import__('torch').cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="runs/control_matrix_20260312.json")
    parser.add_argument("--output_md", type=str, default="runs/control_matrix_20260312.md")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": TASKS,
        "variants": [variant["label"] for variant in VARIANTS],
        "runs": [],
    }

    for task in TASKS:
        for variant in VARIANTS:
            run_args = _base_args(args)
            run_args["synthetic_task"] = task
            run_args["model_variant"] = variant["model_variant"]
            run_args["head_types"] = variant["head_types"]
            print(f"\n=== task={task} variant={variant['label']} ===")
            result = egt.run_experiment(SimpleNamespace(**run_args))
            payload["runs"].append(
                {
                    "task": task,
                    "variant": variant["label"],
                    "config": result["config"],
                    "data_meta": result["data_meta"],
                    "summary": result["summary"],
                }
            )

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_markdown(args.output_md, payload)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
