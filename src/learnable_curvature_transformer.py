"""
Learnable Curvature Transformer
================================

A transformer where each attention head learns its own curvature Îº,
discovering whether hierarchical (hyperbolic), periodic (spherical),
or unstructured (Euclidean) geometry best serves its function.

Requirements:
    pip install torch numpy

Usage:
    # Train on synthetic data (validation):
    python learnable_curvature_transformer.py --mode synthetic
    
    # Train on WikiText-2 (requires torchtext or manual download):
    python learnable_curvature_transformer.py --mode language --data wikitext2.txt
    
    # Analyze learned curvatures:
    python learnable_curvature_transformer.py --mode analyze --checkpoint model.pt

Paper: "Learnable Curvature: Discovering Optimal Geometry for Neural Computation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import json
import os
import numpy as np
from typing import Optional, Tuple, Dict, List


def _softplus_inverse(value: float, eps: float = 1e-8) -> float:
    """Inverse of softplus for positive scalar initialization."""
    safe = max(value - eps, eps)
    return math.log(math.expm1(safe))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CURVATURE-PARAMETERIZED DISTANCE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class CurvatureDistance(nn.Module):
    """
    Differentiable distance function parameterized by learnable curvature Îº.
    
    Îº < 0 â†’ hyperbolic-like (amplifies distances for high-norm points)
    Îº = 0 â†’ Euclidean (standard dot-product attention)
    Îº > 0 â†’ spherical-like (angular similarity dominates)
    
    Uses smooth blending so gradients flow through Îº at all values.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """
        Compute curvature-parameterized attention scores.
        
        Args:
            q: queries  [batch, heads, seq_len, head_dim]
            k: keys     [batch, heads, seq_len, head_dim]
            kappa: per-head curvature [heads] or [1]
            
        Returns:
            scores: [batch, heads, seq_len, seq_len]
        """
        # Standard dot-product component (Euclidean)
        d_k = q.size(-1)
        dot_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Curvature modulation
        # kappa shape: [heads, 1, 1] for broadcasting
        kappa = kappa.view(-1, 1, 1)
        
        # Hyperbolic component: amplify based on query/key norms
        # High-norm vectors get more extreme attention (boundary effect)
        q_norm = q.norm(dim=-1, keepdim=True)  # [B, H, S, 1]
        k_norm = k.norm(dim=-1, keepdim=True)  # [B, H, S, 1]
        norm_product = q_norm * k_norm.transpose(-2, -1)  # [B, H, S, S]
        
        # Spherical component: angular similarity
        # Normalize and compute cosine similarity
        q_normalized = F.normalize(q, dim=-1)
        k_normalized = F.normalize(k, dim=-1)
        cos_sim = torch.matmul(q_normalized, k_normalized.transpose(-2, -1))
        
        # Blend based on sign of Îº:
        #   Îº < 0: scores amplified by norm product (hyperbolic effect)
        #   Îº > 0: scores dominated by angular similarity (spherical effect)
        #   Îº = 0: pure dot product (Euclidean)
        
        # Use tanh for smooth sign decomposition
        hyp_weight = torch.tanh(-kappa * 3.0).clamp(min=0)   # ~1 when Îº<<0
        sph_weight = torch.tanh(kappa * 3.0).clamp(min=0)    # ~1 when Îº>>0
        euc_weight = 1.0 - hyp_weight - sph_weight + 1e-8    # ~1 when Îºâ‰ˆ0
        
        # Hyperbolic scores: dot product amplified by norms
        hyp_scores = dot_scores * (1.0 + 0.5 * norm_product / (d_k ** 0.5))
        
        # Spherical scores: angular similarity (ignore magnitudes)
        sph_scores = cos_sim * (d_k ** 0.5)  # rescale to match dot_scores magnitude
        
        # Blend
        scores = euc_weight * dot_scores + hyp_weight * hyp_scores + sph_weight * sph_scores
        
        return scores


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CURVATURE-ADAPTIVE ATTENTION HEAD
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class CurvatureMultiHeadAttention(nn.Module):
    """
    Multi-head attention where each head has a learnable curvature Îº.
    
    During training, Îº values evolve via gradient descent.
    After training, inspecting Îº reveals the geometric structure
    the model discovered in the data:
    
        Îº < 0 â†’ this head learned to process hierarchical structure
        Îº â‰ˆ 0 â†’ this head does standard attention (no geometric preference)
        Îº > 0 â†’ this head learned to process periodic/angular structure
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        kappa_min: Optional[float] = None,
        kappa_max: Optional[float] = None,
        learnable_tau: bool = False,
        init_tau: float = 1.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # THE KEY INNOVATION: per-head learnable curvature
        # Initialized at 0 (Euclidean) â€” the model discovers geometry
        self.kappa = nn.Parameter(torch.zeros(n_heads))
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.learnable_tau = learnable_tau
        self.tau_eps = 1e-4
        if learnable_tau:
            tau_init = _softplus_inverse(init_tau, eps=self.tau_eps)
            self.log_tau = nn.Parameter(torch.full((n_heads,), tau_init))
        else:
            self.register_parameter("log_tau", None)
        
        self.curvature_dist = CurvatureDistance()
        self.dropout = nn.Dropout(dropout)

    def effective_kappa(self) -> torch.Tensor:
        """Kappa values used in forward pass (optionally clamped)."""
        if self.kappa_min is not None or self.kappa_max is not None:
            return torch.clamp(self.kappa, min=self.kappa_min, max=self.kappa_max)
        return self.kappa

    def effective_tau(self) -> torch.Tensor:
        """Positive per-head attention temperature."""
        if self.log_tau is None:
            return torch.ones_like(self.kappa)
        return F.softplus(self.log_tau) + self.tau_eps
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, 1, 1, seq_len] or None
            
        Returns:
            output: [batch, seq_len, d_model]
            attn_weights: [batch, n_heads, seq_len, seq_len]
        """
        B, S, D = x.shape
        
        # Project to Q, K, V
        q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        
        # Curvature-parameterized attention scores
        scores = self.curvature_dist(q, k, self.effective_kappa())
        scores = scores / self.effective_tau().view(-1, 1, 1)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax + dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.W_o(out)
        
        return out, attn
    
    def get_curvatures(self) -> Dict[str, float]:
        """Return learned curvatures with regime labels"""
        kappa_values = self.effective_kappa().detach().cpu().numpy()
        tau_values = self.effective_tau().detach().cpu().numpy()
        result = {}
        for i, k in enumerate(kappa_values):
            regime = "H" if k < -0.15 else ("S" if k > 0.15 else "E")
            result[f"head_{i}"] = {
                "kappa": float(k),
                "regime": regime,
                "tau": float(tau_values[i]),
            }
        return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  TRANSFORMER BLOCK
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        kappa_min: Optional[float] = None,
        kappa_max: Optional[float] = None,
        learnable_tau: bool = False,
        init_tau: float = 1.0,
    ):
        super().__init__()
        self.attn = CurvatureMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            learnable_tau=learnable_tau,
            init_tau=init_tau,
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
    
    def forward(self, x, mask=None):
        # Pre-norm architecture
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.ln2(x))
        return x, attn_weights


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CURVATURE-ADAPTIVE TRANSFORMER (full model)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class CurvatureTransformer(nn.Module):
    """
    A transformer language model where every attention head
    has a learnable curvature parameter Îº.
    
    Architecture: standard GPT-style decoder-only transformer
    Innovation: CurvatureMultiHeadAttention replaces standard MHA
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        kappa_min: Optional[float] = None,
        kappa_max: Optional[float] = None,
        learnable_tau: bool = False,
        init_tau: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.learnable_tau = learnable_tau
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                kappa_min=kappa_min,
                kappa_max=kappa_max,
                learnable_tau=learnable_tau,
                init_tau=init_tau,
            )
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
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
        """
        Args:
            idx: token indices [batch, seq_len]
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            all_attn: list of attention weights per layer
        """
        B, S = idx.shape
        device = idx.device
        
        # Embeddings
        tok_emb = self.token_emb(idx)
        pos = torch.arange(S, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=device)).unsqueeze(0).unsqueeze(0)
        
        # Transformer blocks
        all_attn = []
        for block in self.blocks:
            x, attn = block(x, mask)
            all_attn.append(attn)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, all_attn
    
    def get_all_curvatures(self) -> Dict[str, Dict]:
        """Return all learned curvatures across all layers"""
        result = {}
        for layer_idx, block in enumerate(self.blocks):
            curvatures = block.attn.get_curvatures()
            for head_name, info in curvatures.items():
                key = f"layer_{layer_idx}/{head_name}"
                result[key] = info
        return result
    
    def curvature_summary(self) -> str:
        """Human-readable summary of discovered geometry"""
        curvatures = self.get_all_curvatures()
        lines = ["Discovered Geometry:", "=" * 50]
        
        for key, info in sorted(curvatures.items()):
            k = info['kappa']
            r = info['regime']
            bar_len = int(abs(k) * 20)
            if k < 0:
                bar = "<" + "=" * bar_len
            elif k > 0:
                bar = "=" * bar_len + ">"
            else:
                bar = "."
            lines.append(f"  {key:<25} k={k:+.4f} [{r}] {bar}")
        
        # Statistics
        kappas = [info['kappa'] for info in curvatures.values()]
        n_hyp = sum(1 for k in kappas if k < -0.15)
        n_sph = sum(1 for k in kappas if k > 0.15)
        n_euc = len(kappas) - n_hyp - n_sph
        
        lines.append("=" * 50)
        lines.append(f"  Hyperbolic heads: {n_hyp}/{len(kappas)}")
        lines.append(f"  Spherical heads:  {n_sph}/{len(kappas)}")
        lines.append(f"  Euclidean heads:  {n_euc}/{len(kappas)}")
        lines.append(f"  Mean kappa: {sum(kappas)/len(kappas):+.4f}")
        
        return "\n".join(lines)
    
    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        kappa_params = sum(block.attn.kappa.numel() for block in self.blocks)
        tau_params = sum(
            0 if block.attn.log_tau is None else block.attn.log_tau.numel()
            for block in self.blocks
        )
        return {
            "total": total,
            "kappa_params": kappa_params,
            "tau_params": tau_params,
            "kappa_fraction": kappa_params / total,
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  BASELINE: STANDARD TRANSFORMER (for comparison)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class StandardMultiHeadAttention(nn.Module):
    """Standard dot-product attention (no curvature)"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, S, D = x.shape
        q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out), attn


class StandardTransformer(nn.Module):
    """Baseline transformer without learnable curvature"""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.ModuleDict({
                'attn': StandardMultiHeadAttention(d_model, n_heads, dropout),
                'ff': nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                   nn.Linear(d_ff, d_model), nn.Dropout(dropout)),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
            })
            self.blocks.append(block)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, idx):
        B, S = idx.shape
        x = self.dropout(self.token_emb(idx) + self.pos_emb(torch.arange(S, device=idx.device).unsqueeze(0)))
        mask = torch.tril(torch.ones(S, S, device=idx.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            attn_out, _ = block['attn'](block['ln1'](x), mask)
            x = x + self.dropout(attn_out)
            x = x + block['ff'](block['ln2'](x))
        return self.lm_head(self.ln_f(x)), []


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SYNTHETIC DATA (for validation without real corpus)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def generate_synthetic_data(n_samples=10000, seq_len=64, vocab_size=256):
    """
    Generate synthetic sequences with known geometric structure:
    - Hierarchical: nested bracket languages  (should induce Îº < 0)
    - Periodic: repeating patterns            (should induce Îº > 0)
    - Mixed: both interleaved                 (should induce product geometry)
    """
    data = []
    
    for _ in range(n_samples):
        task = torch.randint(0, 3, (1,)).item()
        
        if task == 0:
            # Hierarchical: nested structure like ((a)(b(c)))
            seq = []
            depth = 0
            for i in range(seq_len):
                if depth < 4 and torch.rand(1).item() < 0.3:
                    seq.append(10 + depth)  # open bracket at depth
                    depth += 1
                elif depth > 0 and torch.rand(1).item() < 0.3:
                    depth -= 1
                    seq.append(20 + depth)  # close bracket at depth
                else:
                    seq.append(30 + depth * 5 + torch.randint(0, 5, (1,)).item())
            data.append(torch.tensor(seq[:seq_len]))
        
        elif task == 1:
            # Periodic: repeating pattern with noise
            period = torch.randint(3, 8, (1,)).item()
            base = torch.randint(50, 80, (period,))
            seq = base.repeat(seq_len // period + 1)[:seq_len]
            # Add noise
            noise_mask = torch.rand(seq_len) < 0.1
            seq[noise_mask] = torch.randint(80, 100, (noise_mask.sum(),))
            data.append(seq)
        
        else:
            # Mixed: first half hierarchical, second half periodic
            half = seq_len // 2
            hier = torch.randint(100, 130, (half,))
            period = torch.randint(2, 5, (1,)).item()
            periodic = torch.randint(130, 160, (period,)).repeat(half // period + 1)[:half]
            data.append(torch.cat([hier, periodic]))
    
    return torch.stack(data)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  TRAINING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def train_epoch(model, data, optimizer, device, batch_size=32, indices=None):
    model.train()
    total_loss = 0
    n_batches = 0
    
    if indices is None:
        indices = torch.randperm(len(data))
    for i in range(0, len(data) - batch_size, batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = data[batch_idx].to(device)
        
        # Input: all tokens except last. Target: all tokens except first.
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, data, device, batch_size=32):
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(data) - batch_size, batch_size):
            batch = data[i:i+batch_size].to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, math.exp(avg_loss)  # loss, perplexity


def load_language_data(data_path: Optional[str], seq_len: int, vocab_size: int = 256):
    """Load character-level language data and return train/val sequence tensors."""
    if data_path and os.path.exists(data_path):
        print(f"Loading {data_path}...")
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        tokens = torch.tensor([ord(c) % vocab_size for c in text], dtype=torch.long)
        n = len(tokens) // seq_len * seq_len
        tokens = tokens[:n].view(-1, seq_len)
        split = int(0.9 * len(tokens))
        return tokens[:split], tokens[split:]

    print("Loading WikiText-2 from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_text = '\n'.join(item['text'] for item in ds['train'] if item['text'].strip())
    val_text = '\n'.join(item['text'] for item in ds['validation'] if item['text'].strip())

    train_tokens = torch.tensor([ord(c) % vocab_size for c in train_text], dtype=torch.long)
    val_tokens = torch.tensor([ord(c) % vocab_size for c in val_text], dtype=torch.long)

    n_train = len(train_tokens) // seq_len * seq_len
    n_val = len(val_tokens) // seq_len * seq_len
    train_data = train_tokens[:n_train].view(-1, seq_len)
    val_data = val_tokens[:n_val].view(-1, seq_len)
    return train_data, val_data


def _load_sentencepiece_model(spm_model_path: str):
    """Load a SentencePiece model from disk."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("sentencepiece is required for --tokenizer bpe. Install with: pip install sentencepiece") from exc

    if not os.path.exists(spm_model_path):
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model_path}")

    sp = spm.SentencePieceProcessor()
    loaded = sp.Load(spm_model_path)
    if not loaded:
        raise RuntimeError(f"Failed to load SentencePiece model: {spm_model_path}")
    return sp


def load_language_data_bpe(spm_model_path: str, seq_len: int, data_path: Optional[str] = None):
    """Load WikiText-2 (or local text) with BPE tokenization."""
    sp = _load_sentencepiece_model(spm_model_path)
    vocab_size = int(sp.GetPieceSize())

    if data_path and os.path.exists(data_path):
        print(f"Loading {data_path} with BPE tokenizer...")
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        train_text = text
        val_text = text[int(len(text) * 0.9):]
    else:
        print("Loading WikiText-2 from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(item["text"] for item in ds["train"] if item["text"].strip())
        val_text = "\n".join(item["text"] for item in ds["validation"] if item["text"].strip())

    train_ids = sp.Encode(train_text)
    val_ids = sp.Encode(val_text)

    def _chunk(ids: List[int], chunk_len: int):
        n = len(ids) // chunk_len * chunk_len
        if n <= 0:
            raise RuntimeError(f"Tokenized sequence is too short for seq_len={chunk_len}.")
        return torch.tensor(ids[:n], dtype=torch.long).view(-1, chunk_len)

    train_data = _chunk(train_ids, seq_len)
    val_data = _chunk(val_ids, seq_len)
    return train_data, val_data, vocab_size


def _pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze_attention_patterns(
    model: CurvatureTransformer,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
    max_batches: int = 64,
    tokenizer: str = "char",
    sp_processor=None,
) -> Dict:
    """
    Analyze head-level attention behavior and correlate it with learned curvature.

    Character-level signals are proxy metrics.
    BPE mode additionally reports cross-word and sentence-crossing attention mass.
    """
    model.eval()
    layers = model.n_layers
    heads = model.n_heads
    seq_len = data.size(1) - 1

    q_idx = torch.arange(seq_len, device=device)
    k_idx = torch.arange(seq_len, device=device)
    lag = q_idx.unsqueeze(1) - k_idx.unsqueeze(0)  # [Q, K]
    abs_lag = lag.abs().float()
    self_mask = (lag == 0).float()
    local4_mask = (abs_lag <= 4).float()
    long16_mask = (abs_lag >= 16).float()

    punct_chars = ".,;:!?()[]{}\"'"
    punct_ids = {ord(c) % 256 for c in punct_chars}
    bracket_ids = {ord(c) % 256 for c in "()[]{}"}
    newline_id = ord("\n") % 256
    space_id = ord(" ") % 256

    lower_a = ord("a") % 256
    lower_z = ord("z") % 256
    upper_a = ord("A") % 256
    upper_z = ord("Z") % 256

    sums = {
        "mean_abs_lag": torch.zeros(layers, heads, dtype=torch.float64),
        "local4_mass": torch.zeros(layers, heads, dtype=torch.float64),
        "self_mass": torch.zeros(layers, heads, dtype=torch.float64),
        "long16_mass": torch.zeros(layers, heads, dtype=torch.float64),
        "punct_mass": torch.zeros(layers, heads, dtype=torch.float64),
        "bracket_mass": torch.zeros(layers, heads, dtype=torch.float64),
        "newline_mass": torch.zeros(layers, heads, dtype=torch.float64),
        "word_start_mass": torch.zeros(layers, heads, dtype=torch.float64),
        "cross_word_mass": torch.zeros(layers, heads, dtype=torch.float64),
        "sentence_crossing_mass": torch.zeros(layers, heads, dtype=torch.float64),
    }
    lag_hist = torch.zeros(layers, heads, seq_len, dtype=torch.float64)

    total_queries = 0.0
    max_items = min(len(data), batch_size * max_batches)
    n_batches = 0

    if tokenizer == "bpe" and sp_processor is None:
        raise RuntimeError("BPE attention analysis requires a loaded SentencePiece processor.")

    with torch.no_grad():
        for start in range(0, max_items, batch_size):
            batch = data[start:start + batch_size].to(device)
            if batch.size(0) == 0:
                continue

            x = batch[:, :-1]
            _, attn_layers = model(x)
            bsz = x.size(0)
            queries_in_batch = float(bsz * seq_len)
            total_queries += queries_in_batch
            n_batches += 1

            if tokenizer == "bpe":
                punct_pieces = {".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "\"", "'"}
                bracket_pieces = {"(", ")", "[", "]", "{", "}"}

                unique_ids = torch.unique(x).tolist()
                piece_props = {}
                bos_id = int(sp_processor.bos_id())
                for token_id in unique_ids:
                    piece = sp_processor.IdToPiece(int(token_id))
                    core = piece.lstrip("▁")
                    is_word_start = piece.startswith("▁") or (bos_id >= 0 and int(token_id) == bos_id)
                    is_newline = ("\n" in core) or (core == "<0x0A>") or ("\r" in core)
                    is_sentence_end = (core in {".", "!", "?"}) or is_newline
                    piece_props[int(token_id)] = {
                        "punct": core in punct_pieces,
                        "bracket": core in bracket_pieces,
                        "newline": is_newline,
                        "word_start": is_word_start,
                        "sentence_end": is_sentence_end,
                    }

                punct_key = torch.zeros_like(x, dtype=torch.bool)
                bracket_key = torch.zeros_like(x, dtype=torch.bool)
                newline_key = torch.zeros_like(x, dtype=torch.bool)
                word_start_key = torch.zeros_like(x, dtype=torch.bool)
                sentence_end_key = torch.zeros_like(x, dtype=torch.bool)
                for token_id, props in piece_props.items():
                    token_mask = (x == token_id)
                    if props["punct"]:
                        punct_key |= token_mask
                    if props["bracket"]:
                        bracket_key |= token_mask
                    if props["newline"]:
                        newline_key |= token_mask
                    if props["word_start"]:
                        word_start_key |= token_mask
                    if props["sentence_end"]:
                        sentence_end_key |= token_mask

                cross_word_key = torch.zeros((bsz, seq_len, seq_len), dtype=torch.float32, device=device)
                sentence_cross_key = torch.zeros((bsz, seq_len, seq_len), dtype=torch.float32, device=device)
                for bi in range(bsz):
                    token_row = x[bi].tolist()
                    word_ids = []
                    sentence_ids = []
                    current_word = -1
                    current_sentence = 0
                    for token_id in token_row:
                        props = piece_props[int(token_id)]
                        if props["word_start"]:
                            current_word += 1
                        if current_word < 0:
                            current_word = 0
                        word_ids.append(current_word)
                        sentence_ids.append(current_sentence)
                        if props["sentence_end"]:
                            current_sentence += 1

                    word_vec = torch.tensor(word_ids, device=device)
                    sentence_vec = torch.tensor(sentence_ids, device=device)
                    cross_word_key[bi] = (word_vec.unsqueeze(1) != word_vec.unsqueeze(0)).float()
                    sentence_cross_key[bi] = (sentence_vec.unsqueeze(1) != sentence_vec.unsqueeze(0)).float()

                cross_word_key = cross_word_key.unsqueeze(1)
                sentence_cross_key = sentence_cross_key.unsqueeze(1)
            else:
                punct_key = torch.zeros_like(x, dtype=torch.bool)
                for pid in punct_ids:
                    punct_key |= (x == pid)
                bracket_key = torch.zeros_like(x, dtype=torch.bool)
                for bid in bracket_ids:
                    bracket_key |= (x == bid)
                newline_key = (x == newline_id)

                is_alpha = ((x >= lower_a) & (x <= lower_z)) | ((x >= upper_a) & (x <= upper_z))
                prev_is_boundary = torch.zeros_like(x, dtype=torch.bool)
                prev_is_boundary[:, 1:] = (x[:, :-1] == space_id) | (x[:, :-1] == newline_id)
                word_start_key = is_alpha & prev_is_boundary
                cross_word_key = torch.zeros((bsz, 1, seq_len, seq_len), dtype=torch.float32, device=device)
                sentence_cross_key = torch.zeros((bsz, 1, seq_len, seq_len), dtype=torch.float32, device=device)

            punct_key = punct_key.unsqueeze(1).unsqueeze(2).float()
            bracket_key = bracket_key.unsqueeze(1).unsqueeze(2).float()
            newline_key = newline_key.unsqueeze(1).unsqueeze(2).float()
            word_start_key = word_start_key.unsqueeze(1).unsqueeze(2).float()

            for layer_idx, attn in enumerate(attn_layers):
                sums["mean_abs_lag"][layer_idx] += (attn * abs_lag).sum(dim=(0, 2, 3)).cpu().double()
                sums["local4_mass"][layer_idx] += (attn * local4_mask).sum(dim=(0, 2, 3)).cpu().double()
                sums["self_mass"][layer_idx] += (attn * self_mask).sum(dim=(0, 2, 3)).cpu().double()
                sums["long16_mass"][layer_idx] += (attn * long16_mask).sum(dim=(0, 2, 3)).cpu().double()
                sums["punct_mass"][layer_idx] += (attn * punct_key).sum(dim=(0, 2, 3)).cpu().double()
                sums["bracket_mass"][layer_idx] += (attn * bracket_key).sum(dim=(0, 2, 3)).cpu().double()
                sums["newline_mass"][layer_idx] += (attn * newline_key).sum(dim=(0, 2, 3)).cpu().double()
                sums["word_start_mass"][layer_idx] += (attn * word_start_key).sum(dim=(0, 2, 3)).cpu().double()
                sums["cross_word_mass"][layer_idx] += (attn * cross_word_key).sum(dim=(0, 2, 3)).cpu().double()
                sums["sentence_crossing_mass"][layer_idx] += (attn * sentence_cross_key).sum(dim=(0, 2, 3)).cpu().double()

                for lag_val in range(seq_len):
                    lag_mask = (lag == lag_val).float()
                    lag_hist[layer_idx, :, lag_val] += (attn * lag_mask).sum(dim=(0, 2, 3)).cpu().double()

    if total_queries == 0:
        raise RuntimeError("No batches available for attention analysis.")

    kappas = model.get_all_curvatures()
    per_head = {}

    all_kappa = []
    all_positional = []
    all_syntactic = []
    all_local4 = []
    all_long16 = []
    all_punct = []
    all_abslag = []
    all_cross_word = []
    all_sentence_cross = []

    for layer_idx in range(layers):
        for head_idx in range(heads):
            key = f"layer_{layer_idx}/head_{head_idx}"
            kappa = float(kappas[key]["kappa"])

            mean_abs_lag = float(sums["mean_abs_lag"][layer_idx, head_idx] / total_queries)
            local4_mass = float(sums["local4_mass"][layer_idx, head_idx] / total_queries)
            self_mass = float(sums["self_mass"][layer_idx, head_idx] / total_queries)
            long16_mass = float(sums["long16_mass"][layer_idx, head_idx] / total_queries)
            punct_mass = float(sums["punct_mass"][layer_idx, head_idx] / total_queries)
            bracket_mass = float(sums["bracket_mass"][layer_idx, head_idx] / total_queries)
            newline_mass = float(sums["newline_mass"][layer_idx, head_idx] / total_queries)
            word_start_mass = float(sums["word_start_mass"][layer_idx, head_idx] / total_queries)
            cross_word_mass = float(sums["cross_word_mass"][layer_idx, head_idx] / total_queries)
            sentence_crossing_mass = float(sums["sentence_crossing_mass"][layer_idx, head_idx] / total_queries)

            lag_dist = lag_hist[layer_idx, head_idx].numpy()
            lag_total = lag_dist.sum()
            if lag_total > 0:
                lag_dist = lag_dist / lag_total
            lag_peak = int(np.argmax(lag_dist))
            lag_peak_mass = float(lag_dist[lag_peak])
            valid = lag_dist[lag_dist > 0]
            lag_entropy = float(-np.sum(valid * np.log(valid)) / np.log(len(lag_dist)))
            if len(lag_dist) > 2:
                fft_power = np.abs(np.fft.rfft(lag_dist)) ** 2
                periodicity_strength = float(np.max(fft_power[1:]) / (np.sum(fft_power[1:]) + 1e-12))
            else:
                periodicity_strength = 0.0

            if tokenizer == "bpe":
                positional_score = (
                    0.45 * local4_mass +
                    0.25 * lag_peak_mass +
                    0.20 * periodicity_strength +
                    0.10 * self_mass
                )
                syntactic_score = (
                    0.20 * punct_mass +
                    0.15 * bracket_mass +
                    0.20 * word_start_mass +
                    0.20 * long16_mass +
                    0.15 * cross_word_mass +
                    0.10 * sentence_crossing_mass
                )
            else:
                positional_score = (
                    0.40 * local4_mass +
                    0.30 * lag_peak_mass +
                    0.20 * periodicity_strength +
                    0.10 * self_mass
                )
                syntactic_score = (
                    0.35 * punct_mass +
                    0.20 * bracket_mass +
                    0.20 * word_start_mass +
                    0.25 * long16_mass
                )

            if positional_score - syntactic_score > 0.02:
                proxy_label = "positional-proxy"
            elif syntactic_score - positional_score > 0.02:
                proxy_label = "syntactic-proxy"
            else:
                proxy_label = "mixed-proxy"

            per_head[key] = {
                "kappa": kappa,
                "regime": kappas[key]["regime"],
                "mean_abs_lag": mean_abs_lag,
                "local4_mass": local4_mass,
                "self_mass": self_mass,
                "long16_mass": long16_mass,
                "punct_mass": punct_mass,
                "bracket_mass": bracket_mass,
                "newline_mass": newline_mass,
                "word_start_mass": word_start_mass,
                "cross_word_mass": cross_word_mass,
                "sentence_crossing_mass": sentence_crossing_mass,
                "lag_peak": lag_peak,
                "lag_peak_mass": lag_peak_mass,
                "lag_entropy": lag_entropy,
                "periodicity_strength": periodicity_strength,
                "positional_score": positional_score,
                "syntactic_score": syntactic_score,
                "function_proxy": proxy_label,
            }

            all_kappa.append(kappa)
            all_positional.append(positional_score)
            all_syntactic.append(syntactic_score)
            all_local4.append(local4_mass)
            all_long16.append(long16_mass)
            all_punct.append(punct_mass)
            all_abslag.append(mean_abs_lag)
            all_cross_word.append(cross_word_mass)
            all_sentence_cross.append(sentence_crossing_mass)

    correlations = {
        "kappa_vs_positional_score": _pearson_corr(all_kappa, all_positional),
        "kappa_vs_syntactic_score": _pearson_corr(all_kappa, all_syntactic),
        "kappa_vs_local4_mass": _pearson_corr(all_kappa, all_local4),
        "kappa_vs_long16_mass": _pearson_corr(all_kappa, all_long16),
        "kappa_vs_punct_mass": _pearson_corr(all_kappa, all_punct),
        "kappa_vs_mean_abs_lag": _pearson_corr(all_kappa, all_abslag),
        "kappa_vs_cross_word_mass": _pearson_corr(all_kappa, all_cross_word),
        "kappa_vs_sentence_crossing_mass": _pearson_corr(all_kappa, all_sentence_cross),
    }

    tokenization_note = "character-level ord(c)%256"
    notes = "syntactic/positional assignments are proxy metrics"
    if tokenizer == "bpe":
        tokenization_note = "sentencepiece-bpe"
        notes = "includes BPE-aware cross-word and sentence-boundary metrics"

    return {
        "meta": {
            "batches_analyzed": n_batches,
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "tokenization": tokenization_note,
            "notes": notes,
        },
        "correlations": correlations,
        "heads": per_head,
    }


def _copy_kappa_vectors(model: CurvatureTransformer, vectors: List[torch.Tensor], trainable: bool):
    """Copy per-layer kappa vectors into a model and optionally freeze them."""
    if len(vectors) != len(model.blocks):
        raise ValueError("Kappa vector count does not match number of layers.")
    for layer_idx, block in enumerate(model.blocks):
        target = block.attn.kappa
        vec = vectors[layer_idx].to(target.device, dtype=target.dtype)
        if vec.numel() != target.numel():
            raise ValueError(f"Kappa vector size mismatch at layer {layer_idx}.")
        with torch.no_grad():
            target.copy_(vec)
        target.requires_grad = trainable


def _kappa_vectors_from_model(model: CurvatureTransformer) -> List[torch.Tensor]:
    """Extract effective kappas (post-clamp if enabled) per layer."""
    vectors = []
    for block in model.blocks:
        vectors.append(block.attn.effective_kappa().detach().clone().cpu())
    return vectors


def _model_config(config: Dict) -> Dict:
    """Extract model constructor args from a training config dict."""
    return {
        "vocab_size": int(config["vocab_size"]),
        "d_model": int(config["d_model"]),
        "n_heads": int(config["n_heads"]),
        "n_layers": int(config["n_layers"]),
        "d_ff": int(config["d_ff"]),
        "max_seq_len": int(config["max_seq_len"]),
        "kappa_min": config.get("kappa_min", None),
        "kappa_max": config.get("kappa_max", None),
        "learnable_tau": bool(config.get("learnable_tau", False)),
        "init_tau": float(config.get("init_tau", 1.0)),
    }


def run_kappa_ablation(
    checkpoint_path: str,
    data_path: Optional[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_kappa: float,
    output_prefix: str,
    save_every: int = 0,
    kappa_min: Optional[float] = None,
    kappa_max: Optional[float] = None,
    tokenizer: Optional[str] = None,
    spm_model: Optional[str] = None,
):
    """
    Ablation:
    1) learned_kappa: train from scratch with learnable kappa
    2) fixed_kappa: train from scratch with kappa fixed to values discovered in checkpoint
    3) zero_kappa: train from scratch with kappa fixed to zero (euclidean control)
    """
    state = torch.load(checkpoint_path, map_location=device)
    config = dict(state["config"])
    if kappa_min is not None:
        config["kappa_min"] = kappa_min
    if kappa_max is not None:
        config["kappa_max"] = kappa_max
    config.setdefault("kappa_min", None)
    config.setdefault("kappa_max", None)
    selected_tokenizer = tokenizer or config.get("tokenizer", "char")
    selected_spm_model = spm_model or config.get("spm_model", "wikitext2_bpe.model")
    config["tokenizer"] = selected_tokenizer
    if selected_tokenizer == "bpe":
        config["spm_model"] = selected_spm_model

    reference_model = CurvatureTransformer(**_model_config(config)).to(device)
    reference_model.load_state_dict(state["model"])
    fixed_vectors = _kappa_vectors_from_model(reference_model)
    zero_vectors = [torch.zeros_like(v) for v in fixed_vectors]

    vocab_size = int(config.get("vocab_size", 256))
    seq_len = int(config.get("max_seq_len", 64))
    if selected_tokenizer == "bpe":
        train_data, val_data, vocab_size = load_language_data_bpe(
            selected_spm_model,
            seq_len,
            data_path=data_path,
        )
        config["vocab_size"] = vocab_size
    else:
        train_data, val_data = load_language_data(data_path, seq_len, vocab_size=vocab_size)
    print(f"Ablation data: Train={len(train_data)} Val={len(val_data)}")

    init_model = CurvatureTransformer(**_model_config(config)).to(device)
    init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}
    del init_model

    learned_model = CurvatureTransformer(**_model_config(config)).to(device)
    fixed_model = CurvatureTransformer(**_model_config(config)).to(device)
    zero_model = CurvatureTransformer(**_model_config(config)).to(device)
    learned_model.load_state_dict(init_state)
    fixed_model.load_state_dict(init_state)
    zero_model.load_state_dict(init_state)

    _copy_kappa_vectors(fixed_model, fixed_vectors, trainable=False)
    _copy_kappa_vectors(zero_model, zero_vectors, trainable=False)

    learned_kappa_params = [block.attn.kappa for block in learned_model.blocks]
    learned_other = [p for n, p in learned_model.named_parameters() if 'kappa' not in n]
    learned_opt = torch.optim.AdamW([
        {"params": learned_other, "lr": lr},
        {"params": learned_kappa_params, "lr": lr_kappa},
    ], weight_decay=0.01)

    fixed_opt = torch.optim.AdamW(
        [p for p in fixed_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    zero_opt = torch.optim.AdamW(
        [p for p in zero_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    result = {
        "meta": {
            "checkpoint": checkpoint_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lr_kappa": lr_kappa,
            "config": config,
        },
        "learned_kappa": [],
        "fixed_kappa": [],
        "zero_kappa": [],
        "reference_kappa": {
            f"layer_{i}": [float(x) for x in v.tolist()]
            for i, v in enumerate(fixed_vectors)
        },
    }

    print(f"\n{'='*92}")
    print(f"{'Ep':>4} | {'Learned PPL':>11} {'Fixed PPL':>10} {'Zero PPL':>9} | {'Learned κ-range':>15}")
    print(f"{'='*92}")

    for ep in range(1, epochs + 1):
        shared_indices = torch.randperm(len(train_data))

        _ = train_epoch(learned_model, train_data, learned_opt, device, batch_size, indices=shared_indices)
        _ = train_epoch(fixed_model, train_data, fixed_opt, device, batch_size, indices=shared_indices)
        _ = train_epoch(zero_model, train_data, zero_opt, device, batch_size, indices=shared_indices)

        l_loss, l_ppl = evaluate(learned_model, val_data, device, batch_size)
        f_loss, f_ppl = evaluate(fixed_model, val_data, device, batch_size)
        z_loss, z_ppl = evaluate(zero_model, val_data, device, batch_size)

        l_curv = learned_model.get_all_curvatures()
        l_kappas = [v["kappa"] for v in l_curv.values()]
        l_range = max(l_kappas) - min(l_kappas)

        result["learned_kappa"].append({
            "epoch": ep,
            "loss": l_loss,
            "ppl": l_ppl,
            "kappas": {k: v["kappa"] for k, v in l_curv.items()},
        })
        result["fixed_kappa"].append({"epoch": ep, "loss": f_loss, "ppl": f_ppl})
        result["zero_kappa"].append({"epoch": ep, "loss": z_loss, "ppl": z_ppl})

        print(f"{ep:4d} | {l_ppl:11.3f} {f_ppl:10.3f} {z_ppl:9.3f} | {l_range:15.4f}")

        if save_every > 0 and ep % save_every == 0:
            with open(f"{output_prefix}_ablation_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            torch.save({"config": config, "model": learned_model.state_dict(), "epoch": ep}, f"{output_prefix}_ablation_learned.pt")
            torch.save({"config": config, "model": fixed_model.state_dict(), "epoch": ep}, f"{output_prefix}_ablation_fixed.pt")
            torch.save({"config": config, "model": zero_model.state_dict(), "epoch": ep}, f"{output_prefix}_ablation_zero.pt")

    with open(f"{output_prefix}_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    torch.save({"config": config, "model": learned_model.state_dict(), "epoch": epochs}, f"{output_prefix}_ablation_learned.pt")
    torch.save({"config": config, "model": fixed_model.state_dict(), "epoch": epochs}, f"{output_prefix}_ablation_fixed.pt")
    torch.save({"config": config, "model": zero_model.state_dict(), "epoch": epochs}, f"{output_prefix}_ablation_zero.pt")

    best_l = min(row["ppl"] for row in result["learned_kappa"])
    best_f = min(row["ppl"] for row in result["fixed_kappa"])
    best_z = min(row["ppl"] for row in result["zero_kappa"])
    print(f"\nAblation best PPL: learned={best_l:.3f} fixed={best_f:.3f} zero={best_z:.3f}")
    print(f"Ablation artifacts: {output_prefix}_ablation_results.json (+ three checkpoints)")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def save_training_artifacts(
    output_prefix: str,
    config: Dict,
    model: CurvatureTransformer,
    baseline: Optional[StandardTransformer],
    optimizer: torch.optim.Optimizer,
    baseline_optimizer: Optional[torch.optim.Optimizer],
    results: Dict,
    epoch: int,
):
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results_path = f"{output_prefix}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    kappa_path = f"{output_prefix}_kappa_trajectory.json"
    with open(kappa_path, 'w', encoding='utf-8') as f:
        json.dump({
            "epochs": [row["epoch"] for row in results.get("curvature_model", [])],
            "kappas_per_epoch": results.get("curvatures", []),
            "temperatures_per_epoch": results.get("temperatures", []),
            "final_curvatures": model.get_all_curvatures(),
        }, f, indent=2)

    model_path = f"{output_prefix}.pt"
    checkpoint = {
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "results": results,
        "epoch": epoch,
    }
    if baseline is not None:
        checkpoint["baseline_model"] = baseline.state_dict()
    if baseline_optimizer is not None:
        checkpoint["baseline_optimizer"] = baseline_optimizer.state_dict()
    torch.save(checkpoint, model_path)
    return results_path, kappa_path, model_path


def main():
    parser = argparse.ArgumentParser(description="Curvature-Adaptive Transformer")
    parser.add_argument("--mode", choices=["synthetic", "language", "analyze", "analyze_attention", "ablation"], default="synthetic")
    parser.add_argument("--data", type=str, default=None, help="Path to text data")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default=None, help="Tokenization mode (default: checkpoint setting or char)")
    parser.add_argument("--spm_model", type=str, default=None, help="SentencePiece model path for --tokenizer bpe")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline model training/eval")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20, help="Epochs to run (additional epochs when --resume is set)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_kappa", type=float, default=1e-2, help="Learning rate for curvature")
    parser.add_argument("--learnable_tau", action="store_true", help="Learn a positive per-head attention temperature alongside kappa")
    parser.add_argument("--lr_tau", type=float, default=1e-2, help="Learning rate for attention temperature")
    parser.add_argument("--init_tau", type=float, default=1.0, help="Initial attention temperature when --learnable_tau is enabled")
    parser.add_argument("--kappa_min", type=float, default=None, help="Optional lower bound for kappa during forward")
    parser.add_argument("--kappa_max", type=float, default=None, help="Optional upper bound for kappa during forward")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 disables)")
    parser.add_argument("--output_prefix", type=str, default="curvature_transformer")
    parser.add_argument("--analysis_batches", type=int, default=64)
    parser.add_argument("--analysis_batch_size", type=int, default=16)
    parser.add_argument("--analysis_output", type=str, default="attention_geometry_analysis.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    if args.mode in {"analyze", "analyze_attention"}:
        if not args.checkpoint:
            print("Provide --checkpoint for analysis mode")
            return
        state = torch.load(args.checkpoint, map_location=device)
        config = state["config"]
        selected_tokenizer = args.tokenizer or config.get("tokenizer", "char")
        selected_spm_model = args.spm_model or config.get("spm_model", "wikitext2_bpe.model")
        config["tokenizer"] = selected_tokenizer
        if selected_tokenizer == "bpe":
            config["spm_model"] = selected_spm_model

        model = CurvatureTransformer(**_model_config(config)).to(device)
        model.load_state_dict(state["model"])

        if args.mode == "analyze":
            print(model.curvature_summary())
            return

        seq_len = int(config.get("max_seq_len", args.seq_len))
        sp_processor = None
        if selected_tokenizer == "bpe":
            _, val_data, _ = load_language_data_bpe(selected_spm_model, seq_len, data_path=args.data)
            sp_processor = _load_sentencepiece_model(selected_spm_model)
        else:
            vocab_size = int(config.get("vocab_size", 256))
            _, val_data = load_language_data(args.data, seq_len, vocab_size=vocab_size)
        print(f"Analyzing attention on {min(len(val_data), args.analysis_batches * args.analysis_batch_size)} sequences...")
        analysis = analyze_attention_patterns(
            model=model,
            data=val_data,
            device=device,
            batch_size=args.analysis_batch_size,
            max_batches=args.analysis_batches,
            tokenizer=selected_tokenizer,
            sp_processor=sp_processor,
        )
        analysis["checkpoint"] = args.checkpoint
        with open(args.analysis_output, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"Attention analysis saved to {args.analysis_output}")

        print("\nCorrelation summary:")
        for name, value in analysis["correlations"].items():
            if value is None:
                print(f"  {name:<30} n/a")
            else:
                print(f"  {name:<30} {value:+.4f}")
        return

    if args.mode == "ablation":
        if not args.checkpoint:
            print("Provide --checkpoint for ablation mode")
            return
        run_kappa_ablation(
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_kappa=args.lr_kappa,
            output_prefix=args.output_prefix,
            save_every=args.save_every,
            kappa_min=args.kappa_min,
            kappa_max=args.kappa_max,
            tokenizer=args.tokenizer,
            spm_model=args.spm_model,
        )
        return

    checkpoint_state = None
    config = dict(
        vocab_size=256,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_len,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        learnable_tau=args.learnable_tau,
        init_tau=args.init_tau,
    )

    if args.resume:
        if not args.checkpoint:
            print("--resume requires --checkpoint")
            return
        checkpoint_state = torch.load(args.checkpoint, map_location=device)
        config = checkpoint_state["config"]
        if args.kappa_min is not None:
            config["kappa_min"] = args.kappa_min
        if args.kappa_max is not None:
            config["kappa_max"] = args.kappa_max
        config.setdefault("kappa_min", None)
        config.setdefault("kappa_max", None)
        config.setdefault("learnable_tau", False)
        config.setdefault("init_tau", 1.0)
        print(f"Resuming from checkpoint: {args.checkpoint}")
        print(f"Checkpoint config: d_model={config['d_model']} n_heads={config['n_heads']} n_layers={config['n_layers']} seq_len={config['max_seq_len']}")

    selected_tokenizer = args.tokenizer or config.get("tokenizer", "char")
    selected_spm_model = args.spm_model or config.get("spm_model", "wikitext2_bpe.model")
    config["tokenizer"] = selected_tokenizer
    if selected_tokenizer == "bpe":
        config["spm_model"] = selected_spm_model

    seq_len = int(config.get("max_seq_len", args.seq_len))
    vocab_size = int(config.get("vocab_size", 256))

    if args.mode == "synthetic":
        print("Generating synthetic data...")
        train_data = generate_synthetic_data(8000, seq_len, vocab_size)
        val_data = generate_synthetic_data(1000, seq_len, vocab_size)
    else:
        if selected_tokenizer == "bpe":
            train_data, val_data, vocab_size = load_language_data_bpe(
                selected_spm_model,
                seq_len,
                data_path=args.data,
            )
            config["vocab_size"] = vocab_size
        else:
            train_data, val_data = load_language_data(args.data, seq_len, vocab_size=vocab_size)

    print(f"Train: {len(train_data)} sequences, Val: {len(val_data)} sequences")

    model = CurvatureTransformer(**_model_config(config)).to(device)
    baseline_config = {
        "vocab_size": config["vocab_size"],
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "d_ff": config["d_ff"],
        "max_seq_len": config["max_seq_len"],
    }
    baseline = None if args.skip_baseline else StandardTransformer(**baseline_config).to(device)

    params = model.count_parameters()
    print(
        f"\nCurvature model: {params['total']:,} params "
        f"({params['kappa_params']} kappa params, {params['tau_params']} tau params)"
    )
    if args.skip_baseline:
        print("Baseline model: skipped")

    kappa_params = [block.attn.kappa for block in model.blocks]
    tau_params = [
        block.attn.log_tau
        for block in model.blocks
        if block.attn.log_tau is not None
    ]
    other_params = [
        p for n, p in model.named_parameters()
        if 'kappa' not in n and 'log_tau' not in n
    ]
    optim_groups = [
        {"params": other_params, "lr": args.lr},
        {"params": kappa_params, "lr": args.lr_kappa},
    ]
    if tau_params:
        optim_groups.append({"params": tau_params, "lr": args.lr_tau})

    optimizer = torch.optim.AdamW(optim_groups, weight_decay=0.01)

    baseline_optimizer = None
    if baseline is not None:
        baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=args.lr, weight_decay=0.01)

    results = {"curvature_model": [], "baseline": [], "curvatures": [], "temperatures": []}
    start_epoch = 0
    if checkpoint_state is not None:
        model.load_state_dict(checkpoint_state["model"])
        if "optimizer" in checkpoint_state:
            optimizer.load_state_dict(checkpoint_state["optimizer"])
        if baseline is not None and "baseline_model" in checkpoint_state:
            baseline.load_state_dict(checkpoint_state["baseline_model"])
        if baseline is not None and baseline_optimizer is not None and "baseline_optimizer" in checkpoint_state:
            baseline_optimizer.load_state_dict(checkpoint_state["baseline_optimizer"])
        if "results" in checkpoint_state:
            results = checkpoint_state["results"]
        start_epoch = int(checkpoint_state.get("epoch", len(results.get("curvature_model", []))))

    print(f"\n{'='*60}")
    print(f"{'Ep':>4} | {'k-Loss':>8} {'k-PPL':>8} | {'Base-Loss':>9} {'Base-PPL':>9} | Curvatures")
    print(f"{'='*60}")

    last_k_val_loss = float("nan")
    last_k_ppl = float("nan")
    last_b_val_loss = float("nan")
    last_b_ppl = float("nan")

    for local_epoch in range(args.epochs):
        epoch_num = start_epoch + local_epoch + 1

        _ = train_epoch(model, train_data, optimizer, device, args.batch_size)
        if baseline is not None and baseline_optimizer is not None:
            _ = train_epoch(baseline, train_data, baseline_optimizer, device, args.batch_size)

        last_k_val_loss, last_k_ppl = evaluate(model, val_data, device, args.batch_size)
        if baseline is not None:
            last_b_val_loss, last_b_ppl = evaluate(baseline, val_data, device, args.batch_size)
        else:
            last_b_val_loss, last_b_ppl = float("nan"), float("nan")

        curvatures = model.get_all_curvatures()
        temperatures = {k: v["tau"] for k, v in curvatures.items()}
        kappa_str = " ".join(f"{v['kappa']:+.2f}" for v in curvatures.values())
        tau_range_str = ""
        if config.get("learnable_tau", False):
            tau_values = list(temperatures.values())
            tau_range_str = f" | tau=[{min(tau_values):.2f},{max(tau_values):.2f}]"

        if baseline is None:
            base_loss_str = "      n/a"
            base_ppl_str = "      n/a"
        else:
            base_loss_str = f"{last_b_val_loss:9.4f}"
            base_ppl_str = f"{last_b_ppl:9.2f}"
        print(
            f"{epoch_num:4d} | {last_k_val_loss:8.4f} {last_k_ppl:8.2f} | "
            f"{base_loss_str} {base_ppl_str} | [{kappa_str}]{tau_range_str}"
        )

        results["curvature_model"].append({"epoch": epoch_num, "loss": last_k_val_loss, "ppl": last_k_ppl})
        results["baseline"].append({
            "epoch": epoch_num,
            "loss": None if baseline is None else last_b_val_loss,
            "ppl": None if baseline is None else last_b_ppl,
        })
        results["curvatures"].append({k: v["kappa"] for k, v in curvatures.items()})
        results["temperatures"].append(temperatures)

        if args.save_every > 0 and epoch_num % args.save_every == 0:
            save_training_artifacts(
                output_prefix=args.output_prefix,
                config=config,
                model=model,
                baseline=baseline,
                optimizer=optimizer,
                baseline_optimizer=baseline_optimizer,
                results=results,
                epoch=epoch_num,
            )
            print(f"Saved checkpoint at epoch {epoch_num}")

    print(f"\n{model.curvature_summary()}")

    if baseline is not None and not math.isnan(last_b_ppl):
        improvement = (last_b_ppl - last_k_ppl) / last_b_ppl * 100
        print(f"\nFinal perplexity: Curvature={last_k_ppl:.2f}, Baseline={last_b_ppl:.2f}")
        print(f"Improvement: {improvement:+.1f}%")
    else:
        print(f"\nFinal perplexity: Curvature={last_k_ppl:.2f}")

    results_path, kappa_path, model_path = save_training_artifacts(
        output_prefix=args.output_prefix,
        config=config,
        model=model,
        baseline=baseline,
        optimizer=optimizer,
        baseline_optimizer=baseline_optimizer,
        results=results,
        epoch=start_epoch + args.epochs,
    )
    print(f"\nResults saved to {results_path}")
    print(f"Kappa trajectory saved to {kappa_path}")
    print(f"Model saved to {model_path}")

    curvatures = model.get_all_curvatures()
    kappas = [v["kappa"] for v in curvatures.values()]
    kappa_range = max(kappas) - min(kappas)
    tau_values = [v["tau"] for v in curvatures.values()]

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    print(f"kappa range across heads: {kappa_range:.4f}")
    if config.get("learnable_tau", False):
        print(f"tau range across heads: {min(tau_values):.4f} to {max(tau_values):.4f}")
    if kappa_range > 0.3:
        print("-> HEADS DIVERGED. Different heads found different geometry.")
        print("   This is evidence for mixed-curvature structure in the data.")
    elif kappa_range > 0.1:
        print("-> MODERATE DIVERGENCE. Some geometric signal detected.")
    elif max(abs(k) for k in kappas) > 0.15:
        print("-> UNIFORM SHIFT. All heads prefer same geometry.")
    else:
        print("-> NEAR ZERO. Curvature did not move significantly.")
if __name__ == "__main__":
    main()

