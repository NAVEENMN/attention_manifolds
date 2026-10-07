#!/usr/bin/env python3
"""
Prototype: Curve Attention with Query-Dependent Value Sampling (v5)

Uses InKAN's stable B-spline basis evaluation (updated piecewise form).

Usage:
  python prototype_curve_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from inkan.basis import _bspline_basis

# Compile B-spline basis for 20x speedup
_bspline_basis_fast = torch.compile(_bspline_basis)


class CurveAttention(nn.Module):
    """Attention with query-dependent value curve sampling.

    V values are B-spline control points (head_dim per head).
    After standard AV blending, each query predicts WHERE
    to sample the blended V curve, extracting query-specific info.

    Uses InKAN's _bspline_basis (stable piecewise polynomial).
    """

    def __init__(self, head_dim, n_sample_points=64, grid_range=(0., 1.)):
        super().__init__()
        self.head_dim = head_dim
        self.R = n_sample_points
        self.grid_range = grid_range

        # B-spline grid (head_dim control points)
        n_spans = head_dim - 3
        h = (grid_range[1] - grid_range[0]) / n_spans
        self.inv_h = 1.0 / h
        gs = torch.arange(head_dim, dtype=torch.float32) * h + grid_range[0] - 3 * h
        self.register_buffer('grid_starts', gs)

        # Query → sample locations predictor
        self.tau_proj = nn.Linear(head_dim, n_sample_points)

        # Initialize for identity: sample at points where the B-spline
        # sampling matrix B is well-conditioned, so pinv(B) @ B ≈ I.
        # Use evenly spaced points across the FULL support of all bases,
        # extending slightly beyond the grid range to cover edge bases.
        a, b = grid_range
        # Full support of uniform cubic B-spline with these grid_starts
        # extends from grid_starts[0] + h (first basis center) to
        # grid_starts[-1] + h (last basis center), i.e. roughly [-2h, 1+2h]
        # But we clamp to a sensible range
        support_lo = a - 0.5 * h  # slightly before grid start
        support_hi = b + 0.5 * h  # slightly after grid end
        greville = torch.linspace(support_lo, support_hi, head_dim)
        greville_01 = ((greville - a) / (b - a)).clamp(1e-6, 1 - 1e-6)

        # If R == head_dim, use Greville points directly
        # If R != head_dim, use linspace within Greville range
        if n_sample_points == head_dim:
            targets_01 = greville_01
        else:
            targets_01 = torch.linspace(
                greville_01[0].item(), greville_01[-1].item(), n_sample_points
            ).clamp(1e-4, 1 - 1e-4)

        # Initialize tau_proj: zero weights (query-independent at init) + Greville bias
        nn.init.zeros_(self.tau_proj.weight)
        self.tau_proj.bias.data = torch.log(targets_01 / (1 - targets_01))

        # Project R readings back to head_dim
        self.sample_proj = nn.Linear(n_sample_points, head_dim)

        # Initialize sample_proj as pseudo-inverse of B(τ_init)
        # so that sample(eval_curve(coeffs)) ≈ coeffs at initialization
        with torch.no_grad():
            tau_init = torch.sigmoid(self.tau_proj.bias.data)
            tau_mapped = tau_init * (b - a) + a
            tau_flat = tau_mapped.reshape(-1, 1)
            B_init = _bspline_basis(tau_flat, self.grid_starts, self.inv_h, head_dim)
            B_init = B_init.squeeze(1)  # [R, head_dim]
            # Pseudo-inverse: sample_proj.weight = pinv(B_init)
            # pinv(B_init): [head_dim, R] so that pinv(B) @ B ≈ I_{head_dim}
            B_pinv = torch.linalg.pinv(B_init)  # [head_dim, R]
            self.sample_proj.weight.data = B_pinv
            self.sample_proj.bias.data.zero_()

    def eval_curve(self, coeffs, tau_01):
        """Evaluate B-spline curves at points tau. Fully differentiable via InKAN.

        coeffs: [..., head_dim] — control points
        tau_01: [..., R] — sample locations in (0, 1)
        Returns: [..., R]
        """
        a, b = self.grid_range
        tau = tau_01 * (b - a) + a

        # InKAN expects [batch, features] → [batch, features, n_bases]
        orig_shape = tau.shape
        # Reshape to [N, 1] for InKAN
        tau_flat = tau.reshape(-1, 1)
        B = _bspline_basis_fast(tau_flat, self.grid_starts, self.inv_h, self.head_dim)
        # B: [N, 1, head_dim] → [N, head_dim]
        B = B.squeeze(1)
        B = B.reshape(*orig_shape, self.head_dim)  # [..., R, head_dim]

        # Evaluate: coeffs [..., head_dim] × B [..., R, head_dim] → [..., R]
        return (B * coeffs.unsqueeze(-2)).sum(-1)

    def forward(self, q, k, v, mask=None):
        """q, k, v: [batch, heads, seq, head_dim]"""
        dtype = q.dtype

        # Standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)

        # Blend V coefficients (standard AV)
        blended = torch.matmul(weights, v)  # [B, H, Sq, head_dim]

        # Query-dependent sampling locations
        tau = torch.sigmoid(self.tau_proj(q.to(self.tau_proj.weight.dtype)))

        # Sample blended curve
        sampled = self.eval_curve(blended.float(), tau.float())

        # Project back to head_dim
        output = self.sample_proj(sampled.to(self.sample_proj.weight.dtype))
        return output.to(dtype)


class FixedSampleAttention(nn.Module):
    """Control: learned but query-independent sampling locations."""

    def __init__(self, head_dim, n_sample_points=64, grid_range=(0., 1.)):
        super().__init__()
        self.head_dim = head_dim
        self.R = n_sample_points
        self.grid_range = grid_range

        n_spans = head_dim - 3
        h = (grid_range[1] - grid_range[0]) / n_spans
        self.inv_h = 1.0 / h
        gs = torch.arange(head_dim, dtype=torch.float32) * h + grid_range[0] - 3 * h
        self.register_buffer('grid_starts', gs)

        lo = grid_range[0] + 1.0 * h
        hi = grid_range[1] - 1.0 * h
        targets = torch.linspace(lo, hi, n_sample_points)
        targets_01 = (targets - grid_range[0]) / (grid_range[1] - grid_range[0])
        targets_01 = targets_01.clamp(1e-4, 1 - 1e-4)
        self.tau_logit = nn.Parameter(torch.log(targets_01 / (1 - targets_01)))

        self.sample_proj = nn.Linear(n_sample_points, head_dim)

    def eval_curve(self, coeffs, tau_01):
        a, b = self.grid_range
        tau = tau_01 * (b - a) + a
        orig_shape = tau.shape
        tau_flat = tau.reshape(-1, 1)
        B = _bspline_basis_fast(tau_flat, self.grid_starts, self.inv_h, self.head_dim)
        B = B.squeeze(1).reshape(*orig_shape, self.head_dim)
        return (B * coeffs.unsqueeze(-2)).sum(-1)

    def forward(self, q, k, v, mask=None):
        dtype = q.dtype
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        blended = torch.matmul(weights, v)

        tau = torch.sigmoid(self.tau_logit).expand(*blended.shape[:-1], self.R)
        sampled = self.eval_curve(blended.float(), tau)
        output = self.sample_proj(sampled.to(self.sample_proj.weight.dtype))
        return output.to(dtype)


# ─── Tests ───

def test_basis_stability():
    print("=" * 60)
    print("  InKAN Basis Stability")
    print("=" * 60)

    for n_bases in [8, 16, 32, 64, 128]:
        h = 1.0 / (n_bases - 3)
        inv_h = 1.0 / h
        gs = torch.arange(n_bases, dtype=torch.float32) * h - 3 * h
        t = torch.linspace(0, 1, 1000).unsqueeze(0)
        B = _bspline_basis(t, gs, inv_h, n_bases).squeeze(0)

        sum_err = (B.sum(dim=-1) - 1.0).abs().max().item()
        neg = (B < -1e-7).sum().item()
        print(f"  n_bases={n_bases:>4d}: sum_err={sum_err:.2e}, neg={neg}")
        assert sum_err < 1e-5, f"Partition of unity failed at n_bases={n_bases}: {sum_err}"
        assert neg == 0, f"Negative basis values at n_bases={n_bases}: {neg}"


def test_gradients():
    print("\n" + "=" * 60)
    print("  Gradient Flow")
    print("=" * 60)

    ca = CurveAttention(64, 64)
    q = torch.randn(1, 1, 2, 64)
    k = torch.randn(1, 1, 2, 64)
    v = torch.randn(1, 1, 2, 64)

    out = ca(q, k, v)
    out.sum().backward()

    for name, p in ca.named_parameters():
        g = p.grad
        has_grad = g is not None and g.abs().max() > 0
        print(f"  {name:<25s}: {'OK' if has_grad else 'NONE'} (norm={g.norm():.4f})" if has_grad else f"  {name:<25s}: NONE")
        assert has_grad, f"No gradient for {name}"


def test_query_dependent():
    print("\n" + "=" * 60)
    print("  Query-Dependent Reading")
    print("=" * 60)

    ca = CurveAttention(64, 64)
    q1 = torch.randn(1, 1, 1, 64)
    q2 = torch.randn(1, 1, 1, 64)
    k = torch.randn(1, 1, 1, 64)
    v = torch.randn(1, 1, 1, 64)

    w = F.softmax(torch.matmul(q1, k.transpose(-2, -1)) / math.sqrt(64), dim=-1)
    std = torch.matmul(w, v)
    w2 = F.softmax(torch.matmul(q2, k.transpose(-2, -1)) / math.sqrt(64), dim=-1)
    std2 = torch.matmul(w2, v)
    std_diff = (std - std2).abs().max().item()
    print(f"  Standard: diff = {std_diff:.6f}")
    assert std_diff < 1e-5, "Standard attention should give same V to both queries"

    out1 = ca(q1, k, v)
    out2 = ca(q2, k, v)
    curve_diff = (out1 - out2).abs().max().item()
    print(f"  Curve (at init, zero weights): diff = {curve_diff:.6f} (same — identity init)")

    # After perturbing tau_proj weights, queries should produce different outputs
    ca.tau_proj.weight.data.uniform_(-0.1, 0.1)
    out1 = ca(q1, k, v)
    out2 = ca(q2, k, v)
    curve_diff_perturbed = (out1 - out2).abs().max().item()
    print(f"  Curve (perturbed weights):     diff = {curve_diff_perturbed:.6f} (different)")
    assert curve_diff_perturbed > 1e-3, "Curve attention should give different output after weight perturbation"


def test_precision():
    print("\n" + "=" * 60)
    print("  Precision Tests")
    print("=" * 60)

    for dtype, name in [(torch.float32, "float32"), (torch.float16, "float16"), (torch.bfloat16, "bfloat16")]:
        ca = CurveAttention(64, 64).to(dtype)
        q = torch.randn(1, 1, 2, 64, dtype=dtype)
        k = torch.randn(1, 1, 2, 64, dtype=dtype)
        v = torch.randn(1, 1, 2, 64, dtype=dtype)
        out = ca(q, k, v)
        assert out.isfinite().all(), f"{name}: non-finite output"
        assert out.dtype == dtype, f"{name}: wrong output dtype {out.dtype}"
        print(f"  {name}: OK (output dtype={out.dtype})")


def test_sampling_conditioning():
    print("\n" + "=" * 60)
    print("  Sampling Matrix Conditioning")
    print("=" * 60)

    for hd in [64, 128]:
        ca = CurveAttention(hd, hd)
        with torch.no_grad():
            q = torch.zeros(1, 1, 1, hd)
            tau_01 = torch.sigmoid(ca.tau_proj(q)).squeeze()
            a, b = ca.grid_range
            tau = tau_01 * (b - a) + a
            tau_flat = tau.reshape(-1, 1)
            B = _bspline_basis(tau_flat, ca.grid_starts, ca.inv_h, hd).squeeze(1)
            svs = torch.linalg.svdvals(B)
            cond = svs[0] / svs[-1] if svs[-1] > 0 else float('inf')
            dead = (B.sum(dim=0) < 1e-6).sum().item()
            rank = (svs > 1e-6).sum().item()
            print(f"  head_dim={hd}: cond={cond:.2e}, rank={rank}/{hd}, "
                  f"dead_bases={dead}, tau=[{tau_01.min():.4f}, {tau_01.max():.4f}]")


def test_identity_init():
    """Verify that CurveAttention starts as identity (output ≈ standard AV)."""
    print("\n" + "=" * 60)
    print("  Identity Initialization Test")
    print("=" * 60)

    for hd in [64, 128]:
        ca = CurveAttention(hd, hd)
        torch.manual_seed(42)
        q = torch.randn(2, 4, 8, hd)  # [batch, heads, seq, head_dim]
        k = torch.randn(2, 4, 8, hd)
        v = torch.randn(2, 4, 8, hd)

        # Standard attention output
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        weights = F.softmax(scores, dim=-1)
        standard_out = torch.matmul(weights, v)  # [B, H, S, hd]

        # Curve attention output (should match at init)
        with torch.no_grad():
            curve_out = ca(q, k, v)

        mse = F.mse_loss(curve_out, standard_out).item()
        cos = F.cosine_similarity(
            curve_out.flatten(0, 2), standard_out.flatten(0, 2), dim=-1
        ).mean().item()
        max_err = (curve_out - standard_out).abs().max().item()

        print(f"  head_dim={hd}: MSE={mse:.6e}, cos={cos:.6f}, max_err={max_err:.6e}")
        assert cos > 0.99, f"Identity init failed at head_dim={hd}: cos={cos}"
        assert mse < 0.01, f"Identity init MSE too high at head_dim={hd}: {mse}"


if __name__ == "__main__":
    test_basis_stability()
    test_gradients()
    test_identity_init()
    test_query_dependent()
    test_precision()
    test_sampling_conditioning()

    print(f"\n{'=' * 60}")
    print(f"  All tests passed.")
    ca = CurveAttention(64, 64)
    n = sum(p.numel() for p in ca.parameters())
    print(f"  Params per layer: {n:,}, 16 layers: {n*16:,}")
    print(f"  Using InKAN stable piecewise B-spline basis")
    print(f"  Identity-initialized: starts as standard attention.")
