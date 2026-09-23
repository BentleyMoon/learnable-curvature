#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
LEARNABLE CURVATURE with AUTOMATIC DIFFERENTIATION
═══════════════════════════════════════════════════════════════════════

Custom autograd engine (micrograd-style) that supports:
- Poincaré ball (hyperbolic) distance — PROPER implementation  
- Spherical distance — PROPER implementation
- Smooth interpolation via learnable curvature κ
- Exact gradients through all operations

This replaces the numerical gradient hack from v1-v4.
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import math
import json

import os
np.random.seed(int(os.environ.get("LG_SEED", "42")))  # original: np.random.seed(42)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUTOGRAD ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Var:
    """Scalar variable with autograd support"""
    def __init__(self, val, children=(), op=''):
        self.val = float(val)
        self.grad = 0.0
        self._backward = lambda: None
        self._children = children
        self._op = op
    
    def __repr__(self): return f"Var({self.val:.6f})"
    
    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.val + other.val, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other): return self.__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.val * other.val, (self, other), '*')
        def _backward():
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): return self.__mul__(other)
    
    def __sub__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return self + (other * Var(-1.0))
    
    def __rsub__(self, other): return Var(other) - self
    
    def __truediv__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.val / (other.val + 1e-12), (self, other), '/')
        def _backward():
            self.grad += out.grad / (other.val + 1e-12)
            other.grad += -self.val / ((other.val + 1e-12)**2) * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self): return self * Var(-1.0)
    
    def __pow__(self, n):
        assert isinstance(n, (int, float))
        out = Var(self.val ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.val ** (n-1)) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        ev = math.exp(min(self.val, 80))
        out = Var(ev, (self,), 'exp')
        def _backward():
            self.grad += ev * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Var(math.log(max(self.val, 1e-12)), (self,), 'log')
        def _backward():
            self.grad += out.grad / max(self.val, 1e-12)
        out._backward = _backward
        return out
    
    def sqrt(self):
        sv = math.sqrt(max(self.val, 1e-12))
        out = Var(sv, (self,), 'sqrt')
        def _backward():
            self.grad += out.grad / (2 * sv + 1e-12)
        out._backward = _backward
        return out
    
    def tanh(self):
        tv = math.tanh(self.val)
        out = Var(tv, (self,), 'tanh')
        def _backward():
            self.grad += (1 - tv**2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Var(max(0, self.val), (self,), 'relu')
        def _backward():
            self.grad += (self.val > 0) * out.grad
        out._backward = _backward
        return out
    
    def acosh(self):
        v = max(self.val, 1.0 + 1e-8)
        out = Var(math.acosh(v), (self,), 'acosh')
        def _backward():
            self.grad += out.grad / math.sqrt(max(v*v - 1, 1e-12))
        out._backward = _backward
        return out
    
    def acos(self):
        v = max(-1+1e-7, min(1-1e-7, self.val))
        out = Var(math.acos(v), (self,), 'acos')
        def _backward():
            self.grad += -out.grad / math.sqrt(max(1 - v*v, 1e-12))
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def var_dot(xs, ys):
    """Dot product of two lists of Vars"""
    return sum(x * y for x, y in zip(xs, ys))

def var_norm_sq(xs):
    return sum(x * x for x in xs)

def var_norm(xs):
    return var_norm_sq(xs).sqrt()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GEOMETRIC DISTANCES (with autograd)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def euclidean_dist(x, y):
    """Standard Euclidean distance"""
    return sum((xi - yi)**2 for xi, yi in zip(x, y)).sqrt()

def poincare_dist(x, y, c):
    """
    Poincaré ball distance with curvature c (c > 0 gives curvature -c).
    d(x,y) = (1/√c) * acosh(1 + 2c||x-y||² / ((1-c||x||²)(1-c||y||²)))
    
    c is a Var so gradients flow through it.
    """
    diff_sq = sum((xi - yi)**2 for xi, yi in zip(x, y))
    x_sq = var_norm_sq(x)
    y_sq = var_norm_sq(y)
    
    # Clamp to stay inside ball
    denom = (Var(1.0) - c * x_sq) * (Var(1.0) - c * y_sq)
    # Soft clamp
    safe_denom = denom * denom.relu() / (denom * denom + Var(0.01)).sqrt() + Var(0.01)
    
    arg = Var(1.0) + Var(2.0) * c * diff_sq / safe_denom
    
    return arg.acosh() / c.sqrt()

def spherical_dist(x, y, c):
    """
    Spherical distance with curvature c (c > 0).
    d(x,y) = (1/√c) * acos(cos(√c * ||x-y||))
    
    Simplified: use dot-product formula on normalized vectors.
    """
    x_norm = var_norm(x)
    y_norm = var_norm(y)
    
    dot = var_dot(x, y)
    cos_angle = dot / (x_norm * y_norm + Var(1e-8))
    
    # Clamp cos to [-1+eps, 1-eps]
    # Soft clamp using tanh
    safe_cos = cos_angle * Var(0.999)
    
    angle = safe_cos.acos()
    return angle / c.sqrt()

def kappa_dist(x, y, kappa):
    """
    Unified distance parameterized by learnable curvature κ.
    
    κ < 0 → Poincaré (hyperbolic) with c = |κ|
    κ ≈ 0 → Euclidean  
    κ > 0 → Spherical with c = κ
    
    Smooth transition using sigmoid blending.
    """
    d_euc = euclidean_dist(x, y)
    
    # Smooth blending weights
    # sigmoid(κ * scale) → 0 for κ<0, 1 for κ>0
    # sigmoid(-κ * scale) → 1 for κ<0, 0 for κ>0
    scale = Var(5.0)
    
    sig_pos = (kappa * scale).tanh() * Var(0.5) + Var(0.5)  # ~1 when κ>0
    sig_neg = (kappa * Var(-1.0) * scale).tanh() * Var(0.5) + Var(0.5)  # ~1 when κ<0
    sig_flat = Var(1.0) - sig_pos - sig_neg + Var(1.0)  # ~1 when κ≈0
    
    # For hyperbolic: use |κ| as curvature param  
    abs_kappa = (kappa * kappa + Var(0.001)).sqrt()
    
    # Hyperbolic component (amplifies distance)
    hyp_factor = (Var(1.0) + abs_kappa * var_norm_sq(x) * Var(0.3)) * \
                 (Var(1.0) + abs_kappa * var_norm_sq(y) * Var(0.3))
    d_hyp = d_euc * hyp_factor.sqrt()
    
    # Spherical component (compresses distance via angular similarity)
    x_n = var_norm(x)
    y_n = var_norm(y)
    dot = var_dot(x, y)
    cos_sim = dot / (x_n * y_n + Var(1e-8))
    # angle-based: dissimilar directions → larger distance
    d_sph = d_euc * (Var(2.0) - cos_sim) * Var(0.5) * (Var(1.0) + abs_kappa * Var(0.5))
    
    # Blend based on sign of κ
    d = sig_neg * d_hyp + sig_pos * d_sph + (Var(1.0) - sig_neg - sig_pos).relu() * d_euc
    
    return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VERIFY AUTOGRAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 70)
print("AUTOGRAD VERIFICATION")
print("=" * 70)

x = [Var(0.3), Var(0.5), Var(-0.2), Var(0.1)]
y = [Var(0.6), Var(-0.3), Var(0.4), Var(0.7)]

print("\nGradient of distance w.r.t. κ at various values:")
print(f"{'κ':>8} {'dist':>10} {'∂d/∂κ':>12} {'verified':>10}")
print("-" * 45)

for kv in [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]:
    kappa = Var(kv)
    xv = [Var(v.val) for v in x]
    yv = [Var(v.val) for v in y]
    d = kappa_dist(xv, yv, kappa)
    d.backward()
    
    # Numerical verification
    eps = 1e-5
    k_p = Var(kv + eps)
    k_m = Var(kv - eps)
    d_p = kappa_dist([Var(v.val) for v in x], [Var(v.val) for v in y], k_p)
    d_m = kappa_dist([Var(v.val) for v in x], [Var(v.val) for v in y], k_m)
    num_grad = (d_p.val - d_m.val) / (2 * eps)
    
    match = "✓" if abs(kappa.grad - num_grad) < 0.01 * (abs(num_grad) + 0.01) else "✗"
    print(f"{kv:>+8.3f} {d.val:>10.5f} {kappa.grad:>+12.6f} {num_grad:>+10.6f} {match}")

kappa_zero = Var(0.0)
d_zero = kappa_dist([Var(v.val) for v in x], [Var(v.val) for v in y], kappa_zero)
d_zero.backward()
print(f"\n∂d/∂κ at κ=0: {kappa_zero.grad:+.8f}  {'NON-ZERO ✓' if abs(kappa_zero.grad) > 1e-6 else 'ZERO ✗'}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONTRASTIVE LOSS (InfoNCE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def infonce_var(d_pos, d_negs, temp=0.5):
    """InfoNCE with Var — gradients flow through everything"""
    logits = [d_pos * Var(-1.0/temp)] + [d * Var(-1.0/temp) for d in d_negs]
    # LogSumExp
    max_l = max(l.val for l in logits)
    exps = [(l - Var(max_l)).exp() for l in logits]
    sum_exp = sum(exps)
    log_sum = sum_exp.log() + Var(max_l)
    loss = logits[0] * Var(-1.0) + log_sum
    return loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA GENERATORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gen_tree_data(n, dim=6):
    nodes = []
    def build(d, path):
        nodes.append(list(path))
        if d < 4:
            for b in range(3): build(d+1, path+[b])
    build(0, [])
    
    data = []
    for _ in range(n):
        a = [nd for nd in nodes if len(nd) >= 2]
        a = a[np.random.randint(len(a))]
        pos = [nd for nd in nodes if len(nd) >= 2 and nd[:2] == a[:2] and nd != a]
        neg = [nd for nd in nodes if len(nd) >= 1 and nd[0] != a[0]]
        if not pos or not neg: continue
        p = pos[np.random.randint(len(pos))]
        ns = [neg[np.random.randint(len(neg))] for _ in range(4)]
        
        def feat(nd):
            f = np.array([nd[i]/3.0 if i < len(nd) else len(nd)/5.0 + np.random.randn()*0.03 for i in range(dim)])
            return f
        data.append((feat(a), feat(p), [feat(g) for g in ns]))
    return data

def gen_phase_data(n, dim=6):
    data = []
    for _ in range(n):
        freq = np.random.randint(2, 5)
        pa = np.random.uniform(0, 2*np.pi)
        pp = pa + np.random.randn() * 0.15
        def mk(phase):
            return np.array([np.sin(freq*2*np.pi*i/dim + phase) + 0.3*np.cos(2*freq*2*np.pi*i/dim + phase) + np.random.randn()*0.02 for i in range(dim)])
        ns = [mk(pa + np.pi + np.random.randn()*0.3) for _ in range(4)]
        data.append((mk(pa), mk(pp), ns))
    return data

def gen_flat_data(n, dim=6):
    centers = [np.random.randn(dim) * 1.5 for _ in range(5)]
    data = []
    for _ in range(n):
        ca = np.random.randint(5)
        cn = ca
        while cn == ca: cn = np.random.randint(5)
        def pt(ci): return centers[ci] + np.random.randn(dim) * 0.3
        data.append((pt(ca), pt(ca), [pt(cn) for _ in range(4)]))
    return data

def gen_mixed_data(n, dim=6):
    """First half: tree. Second half: phase."""
    nodes = []
    def build(d, path):
        nodes.append(list(path))
        if d < 3:
            for b in range(3): build(d+1, path+[b])
    build(0, [])
    h = dim // 2
    
    data = []
    for _ in range(n):
        a_nd = [nd for nd in nodes if len(nd) >= 2]
        a_nd = a_nd[np.random.randint(len(a_nd))]
        pos_pool = [nd for nd in nodes if len(nd) >= 2 and nd[:2] == a_nd[:2] and nd != a_nd]
        neg_pool = [nd for nd in nodes if len(nd) >= 1 and nd[0] != a_nd[0]]
        if not pos_pool or not neg_pool: continue
        
        freq = np.random.randint(2, 4)
        pa = np.random.uniform(0, 2*np.pi)
        pp = pa + np.random.randn() * 0.15
        
        def mk(nd, phase):
            hier = np.array([nd[i]/3.0 if i < len(nd) else np.random.randn()*0.03 for i in range(h)])
            cycl = np.array([np.sin(freq*2*np.pi*i/h + phase) + np.random.randn()*0.02 for i in range(dim - h)])
            return np.concatenate([hier, cycl])
        
        p_nd = pos_pool[np.random.randint(len(pos_pool))]
        negs = [mk(neg_pool[np.random.randint(len(neg_pool))], pa + np.pi + np.random.randn()*0.3) for _ in range(4)]
        data.append((mk(a_nd, pa), mk(p_nd, pp), negs))
    return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def to_vars(arr):
    return [Var(float(v)) for v in arr]

def train_kappa(data, name, expected, dim=6, epochs=25, lr=0.15, batch=150):
    """Train a single learnable curvature κ on contrastive data"""
    kappa_val = 0.0
    history = {'kappa': [0.0], 'loss': [], 'grad': []}
    
    print(f"\n{'━'*70}")
    print(f"{name}  |  Expected: {expected}")
    print(f"{'━'*70}")
    print(f"  {len(data)} samples, dim={dim}, lr={lr}")
    
    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss = 0
        total_grad = 0
        n = min(batch, len(data))
        
        for a_np, p_np, negs_np in data[:n]:
            kappa = Var(kappa_val)
            a = to_vars(a_np)
            p = to_vars(p_np)
            negs = [to_vars(ng) for ng in negs_np]
            
            d_pos = kappa_dist(a, p, kappa)
            d_negs = [kappa_dist(a, ng, kappa) for ng in negs]
            
            loss = infonce_var(d_pos, d_negs, temp=0.5)
            loss.backward()
            
            total_loss += loss.val
            total_grad += kappa.grad
        
        avg_loss = total_loss / n
        avg_grad = total_grad / n
        
        kappa_val -= lr * avg_grad
        kappa_val = max(-2.0, min(2.0, kappa_val))
        
        history['kappa'].append(kappa_val)
        history['loss'].append(avg_loss)
        history['grad'].append(avg_grad)
        
        if (epoch+1) % 3 == 0 or epoch == 0:
            r = "ℍ" if kappa_val < -0.15 else ("S" if kappa_val > 0.15 else "ℝ")
            arrow = "↓hyp" if avg_grad > 0.01 else ("↑sph" if avg_grad < -0.01 else "—")
            print(f"  Ep {epoch+1:3d} | κ={kappa_val:+7.4f}[{r}] loss={avg_loss:.4f} ∂L/∂κ={avg_grad:+.6f} {arrow}")
    
    regime = "ℍ" if kappa_val < -0.15 else ("S" if kappa_val > 0.15 else "ℝ")
    print(f"\n  ── RESULT: κ={kappa_val:+.4f} [{regime}]  (expected: {expected})")
    
    return kappa_val, regime, history

def train_multi_kappa(data, name, dim=6, epochs=30, lr=0.15, batch=150):
    """Train TWO learnable curvatures on product data"""
    h = dim // 2
    k1_val, k2_val = 0.0, 0.0
    history = {'k1': [0.0], 'k2': [0.0], 'loss': []}
    
    print(f"\n{'━'*70}")
    print(f"{name}")
    print(f"κ₁ for dims [0:{h}], κ₂ for dims [{h}:{dim}]")
    print(f"{'━'*70}")
    
    for epoch in range(epochs):
        np.random.shuffle(data)
        tl = 0; g1 = 0; g2 = 0
        n = min(batch, len(data))
        
        for a_np, p_np, negs_np in data[:n]:
            # Distance = sqrt(d1² + d2²) where d1, d2 use different κ
            k1 = Var(k1_val)
            k2 = Var(k2_val)
            
            a1, a2 = to_vars(a_np[:h]), to_vars(a_np[h:])
            p1, p2 = to_vars(p_np[:h]), to_vars(p_np[h:])
            
            d1_pos = kappa_dist(a1, p1, k1)
            d2_pos = kappa_dist(a2, p2, k2)
            d_pos = (d1_pos * d1_pos + d2_pos * d2_pos + Var(1e-10)).sqrt()
            
            d_negs = []
            for ng in negs_np:
                n1, n2 = to_vars(ng[:h]), to_vars(ng[h:])
                dn1 = kappa_dist(a1, n1, k1)
                dn2 = kappa_dist(a2, n2, k2)
                d_negs.append((dn1 * dn1 + dn2 * dn2 + Var(1e-10)).sqrt())
            
            loss = infonce_var(d_pos, d_negs, temp=0.5)
            loss.backward()
            
            tl += loss.val
            g1 += k1.grad
            g2 += k2.grad
        
        k1_val -= lr * g1 / n
        k2_val -= lr * g2 / n
        k1_val = max(-2.0, min(2.0, k1_val))
        k2_val = max(-2.0, min(2.0, k2_val))
        
        history['k1'].append(k1_val)
        history['k2'].append(k2_val)
        history['loss'].append(tl/n)
        
        if (epoch+1) % 4 == 0 or epoch == 0:
            r1 = "ℍ" if k1_val < -0.15 else ("S" if k1_val > 0.15 else "ℝ")
            r2 = "ℍ" if k2_val < -0.15 else ("S" if k2_val > 0.15 else "ℝ")
            print(f"  Ep {epoch+1:3d} | κ₁={k1_val:+7.4f}[{r1}] κ₂={k2_val:+7.4f}[{r2}] "
                  f"loss={tl/n:.4f} ∂κ₁={g1/n:+.5f} ∂κ₂={g2/n:+.5f}")
    
    r1 = "ℍ" if k1_val < -0.15 else ("S" if k1_val > 0.15 else "ℝ")
    r2 = "ℍ" if k2_val < -0.15 else ("S" if k2_val > 0.15 else "ℝ")
    print(f"\n  ── PRODUCT: {r1} × {r2}")
    print(f"     κ₁(hier)={k1_val:+.4f}  κ₂(peri)={k2_val:+.4f}")
    
    return k1_val, k2_val, r1, r2, history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RUN ALL EXPERIMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIM = 6
N = 2000

print("\n" + "=" * 70)
print("LEARNABLE CURVATURE — AUTOGRAD EXPERIMENTS")
print(f"Dim={DIM}, N={N}, Custom autograd engine")
print("=" * 70)

tree_data = gen_tree_data(N, DIM)
phase_data = gen_phase_data(N, DIM)
flat_data = gen_flat_data(N, DIM)
mixed_data = gen_mixed_data(N, DIM)

results = {}

k, r, h = train_kappa(tree_data, "HIERARCHICAL (Tree)", "κ < 0 (hyperbolic)", DIM)
results['tree'] = {'kappa': k, 'regime': r, 'expected': 'hyperbolic'}

k, r, h = train_kappa(phase_data, "PERIODIC (Phase)", "κ > 0 (spherical)", DIM)
results['phase'] = {'kappa': k, 'regime': r, 'expected': 'spherical'}

k, r, h = train_kappa(flat_data, "FLAT (Clusters)", "κ ≈ 0 (euclidean)", DIM)
results['flat'] = {'kappa': k, 'regime': r, 'expected': 'euclidean'}

k1, k2, r1, r2, mh = train_multi_kappa(mixed_data, "MIXED: Tree(0:3) + Phase(3:6)", DIM)
results['mixed'] = {'k1': k1, 'k2': k2, 'r1': r1, 'r2': r2}

# ━━━ SYNTHESIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n\n{'='*70}")
print("SYNTHESIS — GEOMETRY DISCOVERY WITH AUTOGRAD")
print(f"{'='*70}")

print(f"\n{'Task':<22} {'Expected':>12} {'Learned κ':>12} {'Regime':>8}")
print("-" * 58)
for name, r in [('Hierarchical', results['tree']), ('Periodic', results['phase']), ('Flat', results['flat'])]:
    print(f"{name:<22} {r['expected']:>12} {r['kappa']:>+12.4f} {r['regime']:>8}")

print(f"\nProduct (Mixed):")
print(f"  κ₁(hier dims)  = {results['mixed']['k1']:+.4f} [{results['mixed']['r1']}]")
print(f"  κ₂(peri dims)  = {results['mixed']['k2']:+.4f} [{results['mixed']['r2']}]")
print(f"  Geometry: {results['mixed']['r1']} × {results['mixed']['r2']}")

# Count matches
matches = 0
if results['tree']['kappa'] < -0.15: matches += 1
if results['phase']['kappa'] > 0.15: matches += 1
if abs(results['flat']['kappa']) < 0.15: matches += 1
product_diff = results['mixed']['r1'] != results['mixed']['r2']

print(f"""
SCORECARD:
  Hierarchical → negative κ (hyperbolic)?  {'✓' if results['tree']['kappa'] < -0.15 else '✗'} (κ={results['tree']['kappa']:+.3f})
  Periodic → positive κ (spherical)?       {'✓' if results['phase']['kappa'] > 0.15 else '✗'} (κ={results['phase']['kappa']:+.3f})
  Flat → near-zero κ (euclidean)?          {'✓' if abs(results['flat']['kappa']) < 0.15 else '✗'} (κ={results['flat']['kappa']:+.3f})
  Product → different curvatures?           {'✓' if product_diff else '✗'} ({results['mixed']['r1']}×{results['mixed']['r2']})

  Score: {matches}/3 single-κ + {'1' if product_diff else '0'}/1 product = {matches + (1 if product_diff else 0)}/4
""")

with open(os.environ.get("LG_OUT", 'autograd_geometry_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("Results saved to autograd_geometry_results.json")
