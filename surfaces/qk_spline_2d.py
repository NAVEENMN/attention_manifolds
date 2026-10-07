"""Experimental 2D Q/K logit splines for HF Llama, Transformers 4.57.6.

No InKAN dependency. V and RoPE's ordinary dot-product branch are unchanged.
This prototype deliberately disables KV caching and uses eager attention.
"""
import contextlib
import math
import types

import torch
from torch import nn
from torch.nn import functional as F


def cubic_basis(t, n_basis):
    """Cardinal cubic basis on [0,1], including both endpoints.

    Basis a has support ((a-3)/(m-3), (a+1)/(m-3)).
    This is a uniform extended knot grid, not an open/clamped knot vector.
    """
    if n_basis < 4:
        raise ValueError('n_basis must be >= 4')
    if not t.is_floating_point():
        raise TypeError('t must be floating point')
    u = t.unsqueeze(-1) * (n_basis - 3) - torch.arange(
        n_basis, device=t.device, dtype=t.dtype) + 3
    z = u.clamp(0, 4)
    p0 = z.pow(3) / 6
    y = z - 1
    p1 = (-3 * y.pow(3) + 3 * y.square() + 3 * y + 1) / 6
    y = 3 - z
    p2 = (-3 * y.pow(3) + 3 * y.square() + 3 * y + 1) / 6
    p3 = (4 - z).pow(3) / 6
    b = torch.where(z < 1, p0, torch.where(z < 2, p1, torch.where(z < 3, p2, p3)))
    return torch.where((u >= 0) & (u <= 4), b, torch.zeros_like(b))


class QKSpline2D(nn.Module):
    """One surface per feature coordinate, shared across heads within a layer.

    delta_score[i,j] = sum_d B(q[i,d]) @ C[d] @ B(k[j,d]) / sqrt(D).
    Inputs here are pre-RoPE projected Q/K, K already expanded for GQA.
    """
    def __init__(self, head_dim, n_basis=8, radius=3.0):
        super().__init__()
        if head_dim < 1 or n_basis < 4 or radius <= 0:
            raise ValueError('Invalid spline dimensions or radius')
        self.head_dim, self.n_basis = head_dim, n_basis
        self.coefficients = nn.Parameter(torch.zeros(head_dim, n_basis, n_basis))
        for kind in ('q', 'k'):
            self.register_buffer(kind + '_mean', torch.zeros(head_dim))
            self.register_buffer(kind + '_radius', torch.full((head_dim,), radius))
        self._edit = None

    def coordinates(self, x, kind):
        # Fixed calibrated affine map; clipping preserves compact basis support
        # in normalized coordinates but boundary supports include clipped tails.
        mean = getattr(self, kind + '_mean')
        radius = getattr(self, kind + '_radius')
        return (0.5 + (x.float() - mean) / (2 * radius)).clamp(0, 1)

    def basis(self, x, kind):
        return cubic_basis(self.coordinates(x, kind), self.n_basis)

    def forward(self, q, k):
        bq, bk = self.basis(q, 'q'), self.basis(k, 'k')
        # [B,H,T,D,M] -> [B,H,T,D,M]; never create [B,H,T,S,D,M,M].
        features = torch.einsum('bhtda,dac->bhtdc', bq, self.coefficients)
        scores = features.flatten(-2) @ bk.flatten(-2).transpose(-1, -2)
        if self._edit is not None:
            h, d, a, b, delta = self._edit
            if h >= q.shape[1]:
                raise ValueError('Edit head outside query-head range')
            patch = bq[:, h, :, d, a].unsqueeze(-1) * bk[:, h, :, d, b].unsqueeze(-2)
            # Runtime edit is head-specific although trained C is head-shared.
            head_mask = F.one_hot(torch.tensor(h, device=q.device), q.shape[1]).to(scores.dtype)
            scores = scores + delta * patch.unsqueeze(1) * head_mask[None, :, None, None]
        return scores / math.sqrt(self.head_dim)

    @contextlib.contextmanager
    def local_edit(self, head, coordinate, q_index, k_index, delta):
        """Add one tensor-product basis patch to one head, then restore exactly."""
        if head < 0 or not 0 <= coordinate < self.head_dim:
            raise ValueError('Invalid head or coordinate')
        if not (0 <= q_index < self.n_basis and 0 <= k_index < self.n_basis):
            raise ValueError('Invalid basis index')
        if not math.isfinite(delta):
            raise ValueError('delta must be finite')
        previous = self._edit
        self._edit = (head, coordinate, q_index, k_index, float(delta))
        try:
            yield self
        finally:
            self._edit = previous


def _forward(attn, hidden_states, position_embeddings, attention_mask=None,
             past_key_values=None, cache_position=None, **kwargs):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    if past_key_values is not None or kwargs.get('past_key_value') is not None:
        raise ValueError('2D prototype requires use_cache=False; KV caching is unsupported')
    if kwargs.get('use_cache', False):
        raise ValueError('2D prototype requires use_cache=False')
    if hidden_states.dtype != torch.float32:
        raise ValueError('This reference implementation requires float32; disable autocast')
    shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
    q0 = attn.q_proj(hidden_states).view(shape).transpose(1, 2)
    k0 = attn.k_proj(hidden_states).view(shape).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q0, k0, *position_embeddings)
    k = repeat_kv(k, attn.num_key_value_groups)
    k0 = repeat_kv(k0, attn.num_key_value_groups)
    v = repeat_kv(v, attn.num_key_value_groups)
    scores = (q @ k.transpose(-1, -2)) * attn.scaling
    scores = scores + attn.qk_spline(q0, k0)
    # Keep causality even if a caller supplies no mask. Padding still requires
    # the standard HF 4D additive mask; the CLI uses unpadded sequences.
    nq, nk = scores.shape[-2:]
    if nq != nk:
        raise ValueError('Only full-sequence, no-cache attention is supported')
    causal = torch.ones(nq, nk, dtype=torch.bool, device=scores.device).triu(1)
    scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
    if attention_mask is not None:
        if attention_mask.ndim != 4 or attention_mask.dtype == torch.bool:
            raise ValueError('Expected a 4D additive attention mask from eager Llama')
        scores = scores + attention_mask[:, :, :, :nk]
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    weights = F.dropout(weights, p=attn.attention_dropout, training=attn.training)
    out = (weights @ v).transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1)
    return attn.o_proj(out), weights


def install(model, n_basis=8, radius=3.0, layers=None):
    """Freeze Llama backbone; install zero-initialized surfaces in selected layers."""
    if model.config.model_type != 'llama':
        raise ValueError('Only HF Llama is supported by this adapter')
    if model.config._attn_implementation != 'eager':
        raise ValueError("Load with attn_implementation='eager'")
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError('Load the backbone in float32')
    selected = list(range(len(model.model.layers))) if layers is None else list(layers)
    if not selected or len(set(selected)) != len(selected) or any(
            i < 0 or i >= len(model.model.layers) for i in selected):
        raise ValueError('Invalid layer selection')
    if any(hasattr(layer.self_attn, 'qk_spline') for layer in model.model.layers):
        raise ValueError('Adapter already installed')
    model.requires_grad_(False)
    model.config.use_cache = False
    for i in selected:
        attn = model.model.layers[i].self_attn
        attn.qk_spline = QKSpline2D(attn.head_dim, n_basis, radius).to(attn.q_proj.weight.device)
        attn.forward = types.MethodType(_forward, attn)
    return model


def adapters(model):
    return {i: layer.self_attn.qk_spline for i, layer in enumerate(model.model.layers)
            if hasattr(layer.self_attn, 'qk_spline')}


@torch.no_grad()
def calibrate(model, batches, std_multiplier=3.0):
    """Fixed per-coordinate domains from pre-RoPE Q/K moments across heads/tokens.

    Run before training. Zero surfaces mean calibration does not change logits.
    Domains are mean +/- std_multiplier*std, with a 1e-3 std floor.
    """
    if std_multiplier <= 0:
        raise ValueError('std_multiplier must be positive')
    if any(torch.count_nonzero(a.coefficients).item() for a in adapters(model).values()):
        raise ValueError('Calibrate before training: changing domains changes learned surfaces')
    stats, hooks = {}, []
    def make_hook(index, kind, dim):
        def hook(module, args, output):
            x = output.detach().reshape(-1, dim).double()
            key = (index, kind)
            s, ss, n = stats.get(key, (0, 0, 0))
            stats[key] = (s + x.sum(0), ss + x.square().sum(0), n + len(x))
        return hook
    was_training = model.training
    model.eval()
    try:
        for i, adapter in adapters(model).items():
            attn = model.model.layers[i].self_attn
            for kind in ('q', 'k'):
                hooks.append(getattr(attn, kind + '_proj').register_forward_hook(
                    make_hook(i, kind, adapter.head_dim)))
        for batch in batches:
            model(input_ids=batch, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    if not stats:
        raise ValueError('No calibration data')
    summary = {}
    for (i, kind), (s, ss, n) in stats.items():
        mean = s / n
        std = (ss / n - mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
        adapter = adapters(model)[i]
        getattr(adapter, kind + '_mean').copy_(mean.float())
        getattr(adapter, kind + '_radius').copy_((std_multiplier * std).float())
        summary[f'{i}.{kind}'] = {'samples': n, 'mean_std': std.mean().item()}
    return summary


def save_adapter(path, model, metadata):
    state = adapters(model)
    if not state:
        raise ValueError('No adapters installed')
    torch.save({'format': 'qk-logit-spline-2d-v1', 'n_basis': next(iter(state.values())).n_basis,
                'layers': list(state), 'state': {i: {k: v.detach().cpu() for k, v in a.state_dict().items()}
                for i, a in state.items()}, 'metadata': metadata}, path)


def load_adapter(path, model):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'qk-logit-spline-2d-v1':
        raise ValueError('Not a 2D logit-spline checkpoint; old 1D checkpoints are incompatible')
    install(model, n_basis=checkpoint['n_basis'], layers=checkpoint['layers'])
    for i, state in checkpoint['state'].items():
        adapters(model)[int(i)].load_state_dict(state, strict=True)
    return checkpoint['metadata']
