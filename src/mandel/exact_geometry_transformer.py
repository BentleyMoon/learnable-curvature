from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo src/
from learnable_curvature_transformer import (
    StandardTransformer,
    evaluate,
    load_language_data,
    train_epoch,
)


class HeadType:
    EUC = "E"
    HYP = "H"
    SPH = "S"
    ALL = {EUC, HYP, SPH}


SYNTHETIC_TASKS = ("hierarchy", "periodic", "mixed", "combined")


def softplus_inverse(value: float, eps: float = 1e-6) -> float:
    value = max(float(value), eps)
    return math.log(math.expm1(value))


def default_head_types(n_heads: int) -> List[str]:
    layouts = {
        4: [HeadType.HYP, HeadType.HYP, HeadType.EUC, HeadType.SPH],
        8: [
            HeadType.HYP,
            HeadType.HYP,
            HeadType.HYP,
            HeadType.EUC,
            HeadType.SPH,
            HeadType.SPH,
            HeadType.SPH,
            HeadType.EUC,
        ],
    }
    if n_heads not in layouts:
        raise ValueError(f"No default exact-geometry layout for n_heads={n_heads}. Provide --head_types.")
    return list(layouts[n_heads])


def all_euclidean_head_types(n_heads: int) -> List[str]:
    return [HeadType.EUC for _ in range(n_heads)]


def parse_head_types(spec: Optional[str], n_heads: int) -> List[str]:
    if spec is None:
        return default_head_types(n_heads)

    raw = spec.replace(" ", "")
    if "," in raw:
        head_types = [item for item in raw.split(",") if item]
    else:
        head_types = list(raw)

    if len(head_types) != n_heads:
        raise ValueError(f"Expected {n_heads} head labels, got {len(head_types)} from {spec!r}.")
    for label in head_types:
        if label not in HeadType.ALL:
            raise ValueError(f"Unknown head label {label!r}. Use only E, H, S.")
    return head_types


def is_exact_variant(model_variant: str) -> bool:
    return model_variant in {"exact", "exact_euclidean"}


def is_temperature_variant(model_variant: str) -> bool:
    return model_variant == "temperature"


def has_attention_state(model_variant: str) -> bool:
    return is_exact_variant(model_variant) or is_temperature_variant(model_variant)


def resolve_head_types(model_variant: str, head_types_spec: Optional[str], n_heads: int) -> Optional[List[str]]:
    if model_variant in {"standard", "temperature"}:
        return None
    if model_variant == "exact_euclidean":
        return all_euclidean_head_types(n_heads)
    return parse_head_types(head_types_spec, n_heads)


def comparison_label(model_variant: str, head_types: Optional[Sequence[str]]) -> str:
    if model_variant == "standard":
        return "standard_dot"
    if model_variant == "temperature":
        return "temperature_attention"
    if model_variant == "exact_euclidean":
        return "exact_euclidean"
    return f"exact_{''.join(head_types or [])}"


def clip_tangent(u: torch.Tensor, max_norm: float, eps: float = 1e-5) -> torch.Tensor:
    norm_u = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm_u, max=1.0)
    return u * scale


def pairwise_euclidean_sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1).unsqueeze(-2)
    dist2 = x_sq + y_sq - 2.0 * torch.matmul(x, y.transpose(-2, -1))
    return dist2.clamp_min(0.0)


def _hierarchy_sequence(seq_len: int, vocab_size: int) -> torch.Tensor:
    seq: List[int] = []
    depth = 0
    max_depth = 4
    for _ in range(seq_len):
        draw = torch.rand(()).item()
        if depth < max_depth and draw < 0.28:
            seq.append(10 + depth)
            depth += 1
        elif depth > 0 and draw < 0.52:
            depth -= 1
            seq.append(20 + depth)
        else:
            seq.append(40 + depth * 6 + int(torch.randint(0, 6, ()).item()))
    return torch.tensor(seq, dtype=torch.long) % vocab_size


def _periodic_sequence(seq_len: int, vocab_size: int, base_start: int = 80, noise_start: int = 120) -> torch.Tensor:
    period = int(torch.randint(3, 8, ()).item())
    base = torch.randint(base_start, base_start + 24, (period,), dtype=torch.long)
    seq = base.repeat(seq_len // period + 1)[:seq_len].clone()
    noise_mask = torch.rand(seq_len) < 0.1
    if noise_mask.any():
        seq[noise_mask] = torch.randint(noise_start, noise_start + 16, (int(noise_mask.sum().item()),), dtype=torch.long)
    return seq % vocab_size


def _mixed_sequence(seq_len: int, vocab_size: int) -> torch.Tensor:
    left_len = seq_len // 2
    right_len = seq_len - left_len
    left = _hierarchy_sequence(left_len, vocab_size)
    right = _periodic_sequence(right_len, vocab_size, base_start=150, noise_start=190)
    return torch.cat([left, right], dim=0) % vocab_size


def generate_task_synthetic_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    task: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if task not in SYNTHETIC_TASKS:
        raise ValueError(f"Unknown synthetic task {task!r}.")

    counts = {name: 0 for name in ("hierarchy", "periodic", "mixed")}
    samples: List[torch.Tensor] = []
    task_order = ["hierarchy", "periodic", "mixed"]
    for _ in range(n_samples):
        selected_task = task
        if task == "combined":
            selected_task = task_order[int(torch.randint(0, len(task_order), ()).item())]

        counts[selected_task] += 1
        if selected_task == "hierarchy":
            sample = _hierarchy_sequence(seq_len, vocab_size)
        elif selected_task == "periodic":
            sample = _periodic_sequence(seq_len, vocab_size)
        else:
            sample = _mixed_sequence(seq_len, vocab_size)
        samples.append(sample)

    payload = torch.stack(samples)
    meta = {
        "mode": "synthetic",
        "synthetic_task": task,
        "component_counts": counts,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "n_samples": n_samples,
    }
    return payload, meta


class LorentzOps:
    def expmap0(self, u: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        sqrt_c = torch.sqrt(c)
        norm_u = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(eps)
        theta = sqrt_c * norm_u
        time = torch.cosh(theta) / sqrt_c
        space = torch.sinh(theta) * u / theta.clamp_min(eps)
        return torch.cat([time, space], dim=-1)

    def pairwise_sqdist(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x0, xv = x[..., :1], x[..., 1:]
        y0, yv = y[..., :1], y[..., 1:]
        lorentz_dot = -torch.matmul(x0, y0.transpose(-2, -1)) + torch.matmul(xv, yv.transpose(-2, -1))
        z = (-c * lorentz_dot).clamp_min(1.0 + eps)
        dist = torch.acosh(z) / torch.sqrt(c)
        return dist * dist


class SphereOps:
    def expmap_north(self, u: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        radius = 1.0 / torch.sqrt(c)
        norm_u = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(eps)
        theta = norm_u / radius
        north = radius * torch.cos(theta)
        tangent = radius * torch.sin(theta) * u / norm_u
        return torch.cat([north, tangent], dim=-1)

    def pairwise_sqdist(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        radius_sq = 1.0 / c
        cosine_angle = (torch.matmul(x, y.transpose(-2, -1)) / radius_sq).clamp(-1.0 + eps, 1.0 - eps)
        dist = torch.acos(cosine_angle) / torch.sqrt(c)
        return dist * dist


class ExactGeometricMultiHeadAttention(nn.Module):
    """Exact metric attention over Q/K only; V and aggregation remain Euclidean."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_types: Sequence[str],
        dropout: float = 0.1,
        init_c: float = 1.0,
        init_tau: float = 1.0,
        learnable_c: bool = False,
        freeze_tau: bool = False,
        tangent_clip: float = 1.5,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if len(head_types) != n_heads:
            raise ValueError("head_types must have length n_heads.")
        if any(label not in HeadType.ALL for label in head_types):
            raise ValueError("head_types may only contain E, H, S.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.head_types = tuple(head_types)
        self.tangent_clip = float(tangent_clip)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        raw_c = torch.full((n_heads,), softplus_inverse(init_c), dtype=torch.float32)
        raw_tau = torch.full((n_heads,), softplus_inverse(init_tau), dtype=torch.float32)
        self.raw_c = nn.Parameter(raw_c, requires_grad=learnable_c)
        self.raw_tau = nn.Parameter(raw_tau, requires_grad=(not freeze_tau))

        self.dropout = nn.Dropout(dropout)
        self.lorentz_ops = LorentzOps()
        self.sphere_ops = SphereOps()

    def curvature(self) -> torch.Tensor:
        return F.softplus(self.raw_c) + 1e-4

    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_tau) + 1e-4

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        curvatures = self.curvature()
        temperatures = self.temperature()
        score_list = []
        for head_idx, head_type in enumerate(self.head_types):
            qh = q[:, head_idx]
            kh = k[:, head_idx]
            tau = temperatures[head_idx]

            if head_type == HeadType.EUC:
                dist2 = pairwise_euclidean_sqdist(qh, kh)
            elif head_type == HeadType.HYP:
                c = curvatures[head_idx]
                qh_geom = self.lorentz_ops.expmap0(clip_tangent(qh, self.tangent_clip), c)
                kh_geom = self.lorentz_ops.expmap0(clip_tangent(kh, self.tangent_clip), c)
                dist2 = self.lorentz_ops.pairwise_sqdist(qh_geom, kh_geom, c)
            else:
                c = curvatures[head_idx]
                qh_geom = self.sphere_ops.expmap_north(clip_tangent(qh, self.tangent_clip), c)
                kh_geom = self.sphere_ops.expmap_north(clip_tangent(kh, self.tangent_clip), c)
                dist2 = self.sphere_ops.pairwise_sqdist(qh_geom, kh_geom, c)

            score_list.append((-dist2 / tau).unsqueeze(1))

        scores = torch.cat(score_list, dim=1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out), attn

    def get_geometry_state(self) -> Dict[str, Dict[str, Optional[float]]]:
        curvatures = self.curvature().detach().cpu().tolist()
        temperatures = self.temperature().detach().cpu().tolist()
        result: Dict[str, Dict[str, Optional[float]]] = {}
        for idx, head_type in enumerate(self.head_types):
            result[f"head_{idx}"] = {
                "type": head_type,
                "curvature": None if head_type == HeadType.EUC else float(curvatures[idx]),
                "temperature": float(temperatures[idx]),
            }
        return result


class TemperatureMultiHeadAttention(nn.Module):
    """Standard dot-product attention with a learnable per-head temperature."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.log_tau = nn.Parameter(torch.zeros(n_heads))
        self.dropout = nn.Dropout(dropout)

    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp_min(1e-4)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        dot = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        tau = self.temperature().view(1, self.n_heads, 1, 1)
        scores = dot / tau
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out), attn

    def get_temperature_state(self) -> Dict[str, Dict[str, float]]:
        temperatures = self.temperature().detach().cpu().tolist()
        return {
            f"head_{idx}": {
                "temperature": float(value),
            }
            for idx, value in enumerate(temperatures)
        }


class ExactGeometryBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        head_types: Sequence[str],
        dropout: float = 0.1,
        init_c: float = 1.0,
        init_tau: float = 1.0,
        learnable_c: bool = False,
        freeze_tau: bool = False,
        tangent_clip: float = 1.5,
    ):
        super().__init__()
        self.attn = ExactGeometricMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            head_types=head_types,
            dropout=dropout,
            init_c=init_c,
            init_tau=init_tau,
            learnable_c=learnable_c,
            freeze_tau=freeze_tau,
            tangent_clip=tangent_clip,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.ln2(x))
        return x, attn_weights


class TemperatureBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = TemperatureMultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.ln2(x))
        return x, attn_weights


class ExactGeometryTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        head_types: Optional[Sequence[str]] = None,
        init_c: float = 1.0,
        init_tau: float = 1.0,
        learnable_c: bool = False,
        freeze_tau: bool = False,
        tangent_clip: float = 1.5,
    ):
        super().__init__()
        resolved_head_types = list(head_types) if head_types is not None else default_head_types(n_heads)
        if len(resolved_head_types) != n_heads:
            raise ValueError("head_types must match n_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_types = tuple(resolved_head_types)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                ExactGeometryBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    head_types=self.head_types,
                    dropout=dropout,
                    init_c=init_c,
                    init_tau=init_tau,
                    learnable_c=learnable_c,
                    freeze_tau=freeze_tau,
                    tangent_clip=tangent_clip,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len = idx.shape
        device = idx.device
        tok_emb = self.token_emb(idx)
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

        all_attn = []
        for block in self.blocks:
            x, attn = block(x, mask)
            all_attn.append(attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, all_attn

    def get_all_geometry(self) -> Dict[str, Dict[str, Optional[float]]]:
        result: Dict[str, Dict[str, Optional[float]]] = {}
        for layer_idx, block in enumerate(self.blocks):
            layer_state = block.attn.get_geometry_state()
            for head_name, info in layer_state.items():
                result[f"layer_{layer_idx}/{head_name}"] = info
        return result

    def geometry_summary(self) -> str:
        geometry = self.get_all_geometry()
        hyp = sum(1 for info in geometry.values() if info["type"] == HeadType.HYP)
        sph = sum(1 for info in geometry.values() if info["type"] == HeadType.SPH)
        euc = sum(1 for info in geometry.values() if info["type"] == HeadType.EUC)
        lines = ["Exact Geometry Layout:"]
        lines.append(f"  Head layout per layer: {' '.join(self.head_types)}")
        lines.append(f"  Hyperbolic heads: {hyp}")
        lines.append(f"  Euclidean heads:  {euc}")
        lines.append(f"  Spherical heads:  {sph}")
        for key, info in sorted(geometry.items()):
            c_str = "n/a" if info["curvature"] is None else f"{info['curvature']:.4f}"
            lines.append(f"  {key:<18} type={info['type']} c={c_str} tau={info['temperature']:.4f}")
        return "\n".join(lines)

    def count_parameters(self) -> Dict[str, float]:
        total = sum(param.numel() for param in self.parameters())
        geometry_params = sum(block.attn.raw_c.numel() + block.attn.raw_tau.numel() for block in self.blocks)
        return {
            "total": total,
            "geometry_params": geometry_params,
            "geometry_fraction": geometry_params / total,
        }


class TemperatureTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TemperatureBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len = idx.shape
        device = idx.device
        tok_emb = self.token_emb(idx)
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

        all_attn = []
        for block in self.blocks:
            x, attn = block(x, mask)
            all_attn.append(attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, all_attn

    def get_all_temperatures(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for layer_idx, block in enumerate(self.blocks):
            layer_state = block.attn.get_temperature_state()
            for head_name, info in layer_state.items():
                result[f"layer_{layer_idx}/{head_name}"] = info
        return result

    def temperature_summary(self) -> str:
        temperatures = self.get_all_temperatures()
        lines = ["Temperature Attention Layout:"]
        for key, info in sorted(temperatures.items()):
            lines.append(f"  {key:<18} tau={info['temperature']:.4f}")
        return "\n".join(lines)

    def count_parameters(self) -> Dict[str, float]:
        total = sum(param.numel() for param in self.parameters())
        temperature_params = sum(block.attn.log_tau.numel() for block in self.blocks)
        return {
            "total": total,
            "temperature_params": temperature_params,
            "temperature_fraction": temperature_params / total,
        }


def build_model(args: argparse.Namespace, vocab_size: int, head_types: Optional[Sequence[str]]) -> nn.Module:
    if args.model_variant == "standard":
        return StandardTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dropout=args.dropout,
        )

    if args.model_variant == "temperature":
        return TemperatureTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dropout=args.dropout,
        )

    return ExactGeometryTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        head_types=head_types,
        init_c=args.init_c,
        init_tau=args.init_tau,
        learnable_c=args.learnable_c,
        freeze_tau=args.freeze_tau,
        tangent_clip=args.tangent_clip,
    )


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.lr_tau is None:
        return torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr, weight_decay=0.01)

    tau_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "raw_tau" in name or "log_tau" in name:
            tau_params.append(param)
        else:
            other_params.append(param)

    if not tau_params:
        return torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr, weight_decay=0.01)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": args.lr})
    groups.append({"params": tau_params, "lr": args.lr_tau})
    return torch.optim.AdamW(groups, weight_decay=0.01)


def attention_state_key(model_variant: str) -> Optional[str]:
    if is_exact_variant(model_variant):
        return "geometry"
    if is_temperature_variant(model_variant):
        return "temperature"
    return None


def extract_attention_state(model_variant: str, model: nn.Module) -> Optional[Dict[str, Any]]:
    if is_exact_variant(model_variant):
        return model.get_all_geometry()
    if is_temperature_variant(model_variant):
        return model.get_all_temperatures()
    return None


def print_attention_summary(model_variant: str, model: nn.Module):
    if is_exact_variant(model_variant):
        print(model.geometry_summary())
    elif is_temperature_variant(model_variant):
        print(model.temperature_summary())


def save_artifacts(
    output_prefix: str,
    config: Dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    results: Dict[str, Any],
    epoch: int,
):
    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(f"{output_prefix}_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    torch.save(
        {
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "results": results,
            "epoch": epoch,
        },
        f"{output_prefix}.pt",
    )


def load_data(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
    if args.mode == "synthetic":
        train_data, train_meta = generate_task_synthetic_data(args.train_samples, args.seq_len, args.vocab_size, args.synthetic_task)
        val_data, val_meta = generate_task_synthetic_data(args.val_samples, args.seq_len, args.vocab_size, args.synthetic_task)
        data_meta = {
            "mode": "synthetic",
            "synthetic_task": args.synthetic_task,
            "train": train_meta,
            "val": val_meta,
        }
        return train_data, val_data, args.vocab_size, data_meta

    train_data, val_data = load_language_data(args.data, args.seq_len, vocab_size=args.vocab_size)
    data_meta = {
        "mode": "language",
        "data_path": args.data,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "train_sequences": int(len(train_data)),
        "val_sequences": int(len(val_data)),
    }
    return train_data, val_data, args.vocab_size, data_meta


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    head_types = resolve_head_types(args.model_variant, args.head_types, args.n_heads)
    label = comparison_label(args.model_variant, head_types)

    train_data, val_data, vocab_size, data_meta = load_data(args)
    model = build_model(args, vocab_size=vocab_size, head_types=head_types).to(device)
    optimizer = build_optimizer(model, args)

    config: Dict[str, Any] = {
        "mode": args.mode,
        "comparison_label": label,
        "model_variant": args.model_variant,
        "vocab_size": vocab_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "lr": args.lr,
        "lr_tau": args.lr_tau,
        "head_types": head_types,
        "init_c": args.init_c,
        "init_tau": args.init_tau,
        "learnable_c": args.learnable_c,
        "freeze_tau": args.freeze_tau,
        "tangent_clip": args.tangent_clip,
        "seed": args.seed,
        "synthetic_task": args.synthetic_task if args.mode == "synthetic" else None,
    }

    params = model.count_parameters() if hasattr(model, "count_parameters") else {"total": sum(param.numel() for param in model.parameters())}
    print(f"Train: {len(train_data)} sequences, Val: {len(val_data)} sequences")
    print(f"Model variant: {args.model_variant}")
    print(f"Comparison label: {label}")
    print(f"Parameters: {params}")
    if is_exact_variant(args.model_variant):
        print(f"Head layout: {' '.join(head_types or [])}")
    if is_temperature_variant(args.model_variant):
        print("Per-head learnable temperature enabled.")
    if args.mode == "synthetic":
        print(f"Synthetic task: {args.synthetic_task}")

    results: Dict[str, Any] = {
        "config": config,
        "data_meta": data_meta,
        "metrics": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_data, optimizer, device, args.batch_size)
        val_loss, val_ppl = evaluate(model, val_data, device, args.batch_size)
        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_ppl": float(val_ppl),
        }
        state_key = attention_state_key(args.model_variant)
        if state_key is not None:
            row[state_key] = extract_attention_state(args.model_variant, model)
        results["metrics"].append(row)
        print(f"{epoch:4d} | train={train_loss:.4f} val={val_loss:.4f} ppl={val_ppl:.3f}")

        if args.output_prefix and args.save_every > 0 and epoch % args.save_every == 0:
            save_artifacts(args.output_prefix, config, model, optimizer, results, epoch)
            print(f"Saved checkpoint at epoch {epoch}")

    best_row = min(results["metrics"], key=lambda item: item["val_ppl"])
    final_row = results["metrics"][-1]
    summary: Dict[str, Any] = {
        "comparison_label": label,
        "best_epoch": int(best_row["epoch"]),
        "best_val_ppl": float(best_row["val_ppl"]),
        "final_epoch": int(final_row["epoch"]),
        "final_val_loss": float(final_row["val_loss"]),
        "final_val_ppl": float(final_row["val_ppl"]),
    }
    state_key = attention_state_key(args.model_variant)
    if state_key is not None:
        summary[f"final_{state_key}"] = extract_attention_state(args.model_variant, model)
    results["summary"] = summary

    if has_attention_state(args.model_variant):
        print()
        print_attention_summary(args.model_variant, model)

    if args.output_prefix:
        save_artifacts(args.output_prefix, config, model, optimizer, results, args.epochs)
        print(f"Artifacts saved to {args.output_prefix}_results.json and {args.output_prefix}.pt")

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact geometric attention experiments")
    parser.add_argument("--mode", choices=["synthetic", "language"], default="synthetic")
    parser.add_argument("--model_variant", choices=["exact", "exact_euclidean", "temperature", "standard"], default="exact")
    parser.add_argument("--synthetic_task", choices=list(SYNTHETIC_TASKS), default="combined")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--train_samples", type=int, default=8000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_tau", type=float, default=None, help="Optional learning rate override for temperature parameters")
    parser.add_argument("--head_types", type=str, default=None, help="Examples: HHES or H,H,E,S")
    parser.add_argument("--init_c", type=float, default=1.0)
    parser.add_argument("--init_tau", type=float, default=1.0)
    parser.add_argument("--learnable_c", action="store_true")
    parser.add_argument("--freeze_tau", action="store_true")
    parser.add_argument("--tangent_clip", type=float, default=1.5)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
