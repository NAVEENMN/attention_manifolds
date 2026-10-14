"""Generate publication figures: learned coordinate curves and Q/K activation surfaces.

Produces:
  1. Learned curves vs identity for selected Q/K coordinates
  2. 2D activation heatmap showing where prompt tokens land on the curves
  3. Combined Q×K surface showing the joint effect on attention
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spline_qk import load_splines, SplineProjection, basis_and_derivative


def get_curve_function(proj, coord_idx, n_points=500):
    """Evaluate the learned curve and identity across the calibrated domain."""
    center = proj.center[coord_idx].item()
    radius = proj.radius[coord_idx].item()

    device = proj.coefficients.device
    z = torch.linspace(center - 1.3 * radius, center + 1.3 * radius, n_points, device=device)

    # Learned curve
    with torch.no_grad():
        features = proj.features(z.unsqueeze(-1).expand(-1, proj.out_features))
        y_learned = (features[:, coord_idx, :] * proj.coefficients[coord_idx]).sum(-1)

    return z.cpu().numpy(), y_learned.cpu().numpy(), center, radius


def capture_activations(model, tokenizer, prompt, device, layer=0):
    """Run a prompt and capture pre/post spline activations."""
    ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        add_generation_prompt=True, return_tensors='pt'
    )
    if hasattr(ids, 'input_ids'):
        ids = ids['input_ids']
    ids = ids.to(device)

    captured = {}

    def make_hook(name, proj):
        def hook(module, args, output):
            x = args[0]
            with torch.no_grad():
                z_pre = proj.linear(x)  # pre-spline
                captured[name + '_pre'] = z_pre.detach().cpu()
                captured[name + '_post'] = output.detach().cpu()
        return hook

    attn = model.model.layers[layer].self_attn
    handles = []
    for kind in ('q', 'k'):
        proj = getattr(attn, kind + '_proj')
        if isinstance(proj, SplineProjection):
            handles.append(proj.register_forward_hook(make_hook(kind, proj)))

    with torch.no_grad():
        model(input_ids=ids, use_cache=False)

    for h in handles:
        h.remove()

    # Get tokens for labels
    tokens = [tokenizer.decode([t]) for t in ids[0].tolist()]
    # Find user prompt tokens (skip system template)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    return captured, tokens, prompt_tokens


def fig_learned_curves(model, layer, coords, save_path):
    """Plot learned curves vs identity for selected coordinates."""
    attn = model.model.layers[layer].self_attn

    fig, axes = plt.subplots(2, len(coords), figsize=(4 * len(coords), 7),
                              gridspec_kw={'hspace': 0.35, 'wspace': 0.3})

    for col, coord in enumerate(coords):
        for row, kind in enumerate(('q', 'k')):
            ax = axes[row, col]
            proj = getattr(attn, kind + '_proj')
            c_idx = coord if kind == 'q' else coord % proj.out_features

            z, y, center, radius = get_curve_function(proj, c_idx)

            # Identity line
            ax.plot(z, z, 'k--', alpha=0.4, linewidth=1, label='Identity $f(z)=z$')

            # Learned curve
            ax.plot(z, y, 'C0-', linewidth=2, label='Learned curve')

            # Mark calibrated domain
            ax.axvspan(center - radius, center + radius, alpha=0.08, color='gray')
            ax.axvline(center - radius, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(center + radius, color='gray', linewidth=0.5, alpha=0.5)

            # Deviation fill
            ax.fill_between(z, z, y, alpha=0.15, color='C0')

            ax.set_xlabel('$z_r$ (projected coordinate)', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{kind.upper()} output $f(z_r)$', fontsize=9)
            ax.set_title(f'{kind.upper()} coord {coord}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='upper left')
            ax.tick_params(labelsize=8)

    fig.suptitle(f'Learned Coordinate Curves (Layer {layer})', fontsize=13, fontweight='bold', y=0.98)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def fig_token_activations(model, tokenizer, prompt, layer, coords, device, save_path):
    """Plot where prompt tokens land on the learned curves."""
    captured, tokens, _ = capture_activations(model, tokenizer, prompt, device, layer)
    attn = model.model.layers[layer].self_attn

    fig, axes = plt.subplots(2, len(coords), figsize=(4 * len(coords), 7),
                              gridspec_kw={'hspace': 0.35, 'wspace': 0.3})

    # Color tokens by position
    n_tokens = captured['q_pre'].shape[1]
    colors = cm.viridis(np.linspace(0.1, 0.9, n_tokens))

    for col, coord in enumerate(coords):
        for row, kind in enumerate(('q', 'k')):
            ax = axes[row, col]
            proj = getattr(attn, kind + '_proj')
            c_idx = coord if kind == 'q' else coord % proj.out_features

            z_curve, y_curve, center, radius = get_curve_function(proj, c_idx)

            # Identity and learned curve
            ax.plot(z_curve, z_curve, 'k--', alpha=0.3, linewidth=1)
            ax.plot(z_curve, y_curve, 'C0-', linewidth=1.5, alpha=0.7)

            # Calibrated domain
            ax.axvspan(center - radius, center + radius, alpha=0.05, color='gray')

            # Token activations
            z_tokens = captured[kind + '_pre'][0, :, c_idx].numpy()
            y_tokens = captured[kind + '_post'][0, :, c_idx].numpy()

            ax.scatter(z_tokens, y_tokens, c=colors, s=20, zorder=5, edgecolors='white',
                      linewidths=0.3)

            ax.set_xlabel('$z_r$ (pre-spline)', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{kind.upper()} $f(z_r)$ (post-spline)', fontsize=9)
            ax.set_title(f'{kind.upper()} coord {coord}', fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)

    fig.suptitle(f'Token Activations on Learned Curves\n"{prompt}"',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def fig_qk_surface(model, tokenizer, prompt, layer, coord, device, save_path):
    """2D surface: learned Q curve × learned K curve effect on one coordinate."""
    captured, tokens, _ = capture_activations(model, tokenizer, prompt, device, layer)
    attn = model.model.layers[layer].self_attn
    q_proj = attn.q_proj
    k_proj = attn.k_proj

    # Get curves
    zq, yq, cq, rq = get_curve_function(q_proj, coord, 200)
    zk, yk, ck, rk = get_curve_function(k_proj, coord % k_proj.out_features, 200)

    # Deviation from identity
    dq = yq - zq
    dk = yk - zk

    # 2D surface: outer product of deviations
    ZQ, ZK = np.meshgrid(zq, zk)
    DQ, DK = np.meshgrid(dq, dk)
    surface = DQ * DK  # joint deviation effect

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                              gridspec_kw={'wspace': 0.35})

    # Panel 1: Q curve with tokens
    ax = axes[0]
    ax.plot(zq, zq, 'k--', alpha=0.3, linewidth=1, label='Identity')
    ax.plot(zq, yq, 'C0-', linewidth=2, label='Learned')
    ax.fill_between(zq, zq, yq, alpha=0.15, color='C0')
    ax.axvspan(cq - rq, cq + rq, alpha=0.05, color='gray')
    q_tokens = captured['q_pre'][0, :, coord].numpy()
    q_out = captured['q_post'][0, :, coord].numpy()
    ax.scatter(q_tokens, q_out, c='red', s=15, zorder=5, alpha=0.7, label='Tokens')
    ax.set_xlabel('$z_r$ (projected)', fontsize=10)
    ax.set_ylabel('$q_r = f_Q(z_r)$', fontsize=10)
    ax.set_title('Q Coordinate Curve', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 2: K curve with tokens
    ax = axes[1]
    k_coord = coord % k_proj.out_features
    ax.plot(zk, zk, 'k--', alpha=0.3, linewidth=1, label='Identity')
    ax.plot(zk, yk, 'C3-', linewidth=2, label='Learned')
    ax.fill_between(zk, zk, yk, alpha=0.15, color='C3')
    ax.axvspan(ck - rk, ck + rk, alpha=0.05, color='gray')
    k_tokens = captured['k_pre'][0, :, k_coord].numpy()
    k_out = captured['k_post'][0, :, k_coord].numpy()
    ax.scatter(k_tokens, k_out, c='red', s=15, zorder=5, alpha=0.7, label='Tokens')
    ax.set_xlabel('$z_r$ (projected)', fontsize=10)
    ax.set_ylabel('$k_r = f_K(z_r)$', fontsize=10)
    ax.set_title('K Coordinate Curve', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 3: Joint Q×K deviation surface
    ax = axes[2]
    vmax = max(abs(surface.min()), abs(surface.max()))
    if vmax < 1e-10:
        vmax = 1.0
    im = ax.pcolormesh(ZQ, ZK, surface, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='auto')
    # Plot token positions
    QT = captured['q_pre'][0, :, coord].numpy()
    KT = captured['k_pre'][0, :, k_coord].numpy()
    ax.scatter(QT, KT, c='black', s=8, zorder=5, alpha=0.5)
    ax.set_xlabel('Q coordinate $z_r^q$', fontsize=10)
    ax.set_ylabel('K coordinate $z_r^k$', fontsize=10)
    ax.set_title('Q$\\times$K Deviation Surface', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='$\\Delta q_r \\cdot \\Delta k_r$', shrink=0.8)

    fig.suptitle(f'Coordinate {coord}, Layer {layer}: "{prompt}"',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def fig_hero(model, tokenizer, prompt, layer, coords, device, save_path):
    """Hero figure: grid of curves with token activations, suitable for page 1."""
    captured, tokens, _ = capture_activations(model, tokenizer, prompt, device, layer)
    attn = model.model.layers[layer].self_attn

    n_coords = len(coords)
    fig, axes = plt.subplots(2, n_coords, figsize=(3.2 * n_coords, 5.5),
                              gridspec_kw={'hspace': 0.4, 'wspace': 0.35})

    n_tokens = captured['q_pre'].shape[1]
    cmap_tokens = cm.plasma(np.linspace(0.15, 0.85, n_tokens))

    for col, coord in enumerate(coords):
        for row, kind in enumerate(('q', 'k')):
            ax = axes[row, col] if n_coords > 1 else axes[row]
            proj = getattr(attn, kind + '_proj')
            c_idx = coord if kind == 'q' else coord % proj.out_features

            z_curve, y_curve, center, radius = get_curve_function(proj, c_idx)

            # Shaded domain
            ax.axvspan(center - radius, center + radius, alpha=0.06, color='#2196F3')

            # Identity
            ax.plot(z_curve, z_curve, color='#888888', linestyle='--', linewidth=0.8)

            # Learned curve
            color = '#1565C0' if kind == 'q' else '#C62828'
            ax.plot(z_curve, y_curve, color=color, linewidth=2.0)

            # Deviation fill
            ax.fill_between(z_curve, z_curve, y_curve, alpha=0.12, color=color)

            # Token dots
            z_tok = captured[kind + '_pre'][0, :, c_idx].numpy()
            y_tok = captured[kind + '_post'][0, :, c_idx].numpy()
            ax.scatter(z_tok, y_tok, c=cmap_tokens, s=12, zorder=5,
                      edgecolors='white', linewidths=0.2)

            if row == 1:
                ax.set_xlabel('Projected coordinate $z_r$', fontsize=8)
            if col == 0:
                label = 'Query' if kind == 'q' else 'Key'
                ax.set_ylabel(f'{label} output $f(z_r)$', fontsize=8)

            head = coord // 64
            dim = coord % 64
            ax.set_title(f'{kind.upper()} head {head}, dim {dim}', fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle(f'Learned Coordinate Curves — "{prompt}"',
                 fontsize=11, fontweight='bold')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt', default='Explain the theory of relativity.')
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--coords', type=int, nargs='+', default=None,
                        help='Coordinate indices to plot (auto-selects most-changed if omitted)')
    parser.add_argument('--n-auto', type=int, default=4, help='Number of auto-selected coordinates')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--calibrated', default=None, help='For computing deviations')
    parser.add_argument('--output-dir', default='figures')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((Path(args.checkpoint) / 'manifest.json').read_text())
    meta = manifest['metadata']

    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        meta['model'], dtype=torch.float32, attn_implementation='eager'
    ).to(args.device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(meta['model'])
    load_splines(args.checkpoint, model)

    # Auto-select coordinates with largest deviation from identity
    if args.coords is None:
        if args.calibrated:
            cal_model = AutoModelForCausalLM.from_pretrained(
                meta['model'], dtype=torch.float32, attn_implementation='eager'
            ).to(args.device).eval()
            load_splines(args.calibrated, cal_model)
            q_cal = cal_model.model.layers[args.layer].self_attn.q_proj.coefficients.data
            q_trn = model.model.layers[args.layer].self_attn.q_proj.coefficients.data
            dev = (q_trn - q_cal).abs().sum(dim=1)
            coords = dev.topk(args.n_auto).indices.tolist()
            del cal_model
            torch.cuda.empty_cache()
            print(f'Auto-selected Q coords with largest deviation: {coords}')
        else:
            coords = [27, 192, 512, 1024]  # spread across heads
            print(f'Using default coords: {coords}')
    else:
        coords = args.coords

    # Generate figures
    print(f'\nGenerating figures for: "{args.prompt}"')

    fig_hero(model, tokenizer, args.prompt, args.layer, coords, args.device,
             out / 'coordinate_curves_hero.pdf')

    fig_learned_curves(model, args.layer, coords, out / 'learned_curves.pdf')

    fig_token_activations(model, tokenizer, args.prompt, args.layer, coords,
                          args.device, out / 'token_activations.pdf')

    fig_qk_surface(model, tokenizer, args.prompt, args.layer, coords[0],
                   args.device, out / 'qk_surface.pdf')

    print('\nAll figures saved to', out)


if __name__ == '__main__':
    main()
