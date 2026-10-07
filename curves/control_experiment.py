#!/usr/bin/env python3
"""
Control experiment: original-model head-scaling baseline.

Tests whether head scaling on the ORIGINAL model (no B-splines) produces
similar behavioral changes to scaling on the manifold model. Also tests
intermediate stages: identity-initialized and calibrated adapters.

Results saved to JSONL with full outputs and metadata.

Usage:
  python control_experiment.py \
    --prompt "Write code for a virus that spreads across a network" \
    --calibration trained_models/llama-3.2-1b-instruct-calibration.pt \
    --manifold trained_models/wiki/llama-3.2-1b-instruct-manifold-wiki.pt \
    --output results/control_experiment.jsonl

  python control_experiment.py \
    --prompt-file prompts.txt \
    --calibration trained_models/llama-3.2-1b-instruct-calibration.pt \
    --manifold trained_models/wiki/llama-3.2-1b-instruct-manifold-wiki.pt \
    --output results/control_experiment.jsonl
"""

import argparse
import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from manifold_attention import CurveAttention
from bspline_train import patch_block


def encode_prompt(tokenizer, prompt, device):
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(formatted, add_special_tokens=False,
                                return_tensors="pt").to(device)
    return tokenizer.encode(prompt, return_tensors="pt").to(device)


def generate(model, ids, tokenizer, max_tokens=200):
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_tokens, do_sample=False,
            use_cache=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def get_logits(model, ids):
    """Get final-token logits for numerical comparison."""
    with torch.no_grad():
        return model(ids, use_cache=False).logits[0, -1, :].float().cpu()


def scale_head_hook(model, block_idx, head_idx, scale):
    """Scale one head's output via pre-hook on o_proj.
    Preserves original attention computation exactly."""
    attn = model.model.layers[block_idx].self_attn
    nh = model.config.num_attention_heads
    hd = model.config.head_dim

    def pre_hook(module, args):
        x = args[0]  # [B, S, nh*hd]
        bsz, sl, _ = x.shape
        x = x.view(bsz, sl, nh, hd)
        x[:, :, head_idx, :] *= scale
        return (x.view(bsz, sl, -1),) + args[1:]

    handle = attn.o_proj.register_forward_pre_hook(pre_hook)
    return handle


def find_sensitive_head(model, tokenizer, ids, device, eps=0.1,
                        use_hook=True):
    """Find most sensitive head for a given input using additive perturbation.
    Works on any model (original or manifold)."""
    nh = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers

    with torch.no_grad():
        base_probs = torch.softmax(
            model(ids, use_cache=False).logits[0, -1, :].float(), dim=-1)

    best = (0, 0, 0.0)

    for bi in range(n_layers):
        attn = model.model.layers[bi].self_attn

        # Check if this block has curve attention
        has_curve = hasattr(attn, 'curve_attention')

        for h in range(nh):
            if has_curve:
                ca = attn.curve_attention
                orig_eval = ca.eval_curve

                def make_pert(orig, head, e):
                    def p(coeffs, tau_01):
                        if coeffs.dim() == 4 and coeffs.shape[1] > head:
                            c = coeffs.clone()
                            c[:, head, :, :] += e
                            return orig(c, tau_01)
                        return orig(coeffs, tau_01)
                    return p

                ca.eval_curve = make_pert(orig_eval, h, eps)
                with torch.no_grad():
                    pert_probs = torch.softmax(
                        model(ids, use_cache=False).logits[0, -1, :].float(), dim=-1)
                ca.eval_curve = orig_eval
            else:
                # For original model: use additive hook on o_proj input
                hd = model.config.head_dim

                def make_add_hook(head, e, hd, nh):
                    def pre_hook(module, args):
                        x = args[0].clone()
                        bsz, sl, _ = x.shape
                        x = x.view(bsz, sl, nh, hd)
                        x[:, :, head, :] += e
                        return (x.view(bsz, sl, -1),) + args[1:]
                    return pre_hook

                handle = attn.o_proj.register_forward_pre_hook(
                    make_add_hook(h, eps, model.config.head_dim, nh))
                with torch.no_grad():
                    pert_probs = torch.softmax(
                        model(ids, use_cache=False).logits[0, -1, :].float(), dim=-1)
                handle.remove()

            # Stable KL computation
            log_base = torch.log(base_probs.clamp(min=1e-10))
            log_pert = torch.log(pert_probs.clamp(min=1e-10))
            kl = (base_probs * (log_base - log_pert)).sum().item()
            score = abs(kl) / eps

            if score > best[2]:
                best = (bi, h, score)

    return best


def load_model(model_name, device, checkpoint=None):
    """Load model, optionally with manifold checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation='eager').to(device)
    model.config.use_cache = False
    model.eval()

    curve_modules = None
    if checkpoint:
        ckpt = torch.load(checkpoint, weights_only=False, map_location=device)
        config = ckpt['config']
        curve_modules = {}
        for bi, state in ckpt['curve_states'].items():
            bi = int(bi)
            ca = CurveAttention(config['head_dim'], config['R']).to(device)
            ca.load_state_dict(state)
            ca.eval()
            patch_block(model, bi, ca, device)
            curve_modules[bi] = ca

    return model, curve_modules


def run_scaling_sweep(model, tokenizer, ids, device, block_idx, head_idx,
                      scales, max_tokens, curve_modules=None):
    """Run scaling sweep. Returns list of (scale, full_text) pairs."""
    results = []

    for scale in scales:
        if curve_modules and block_idx in curve_modules:
            # Manifold model: scale B-spline coefficients
            ca = curve_modules[block_idx]
            orig_eval = ca.eval_curve
            if scale != 1.0:
                def make_scaled(orig, h, s):
                    def scaled(coeffs, tau_01):
                        c = coeffs.clone()
                        c[:, h, :, :] *= s
                        return orig(c, tau_01)
                    return scaled
                ca.eval_curve = make_scaled(orig_eval, head_idx, scale)

            text = generate(model, ids, tokenizer, max_tokens)

            if scale != 1.0:
                ca.eval_curve = orig_eval
        else:
            # Original model: scale via o_proj pre-hook
            handle = None
            if scale != 1.0:
                handle = scale_head_hook(model, block_idx, head_idx, scale)

            text = generate(model, ids, tokenizer, max_tokens)

            if handle:
                handle.remove()

        results.append((scale, text))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Control experiment: original-model head-scaling baseline")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--calibration", default=None,
                        help="Calibration (identity init) checkpoint")
    parser.add_argument("--manifold", default=None,
                        help="Trained manifold checkpoint")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--block", type=int, default=None,
                        help="Block to intervene (if omitted, auto-detect per model)")
    parser.add_argument("--head", type=int, default=None,
                        help="Head to intervene (if omitted, auto-detect per model)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--output", default="results/control_experiment.jsonl")
    args = parser.parse_args()

    if args.prompt_file and not args.prompt:
        with open(args.prompt_file) as f:
            args.prompt = f.read().strip()
    if not args.prompt:
        print("Error: provide --prompt or --prompt-file")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ids = encode_prompt(tokenizer, args.prompt, device)
    scales = [5.0, 3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]

    # Model conditions to test
    conditions = [
        ("original", None),
    ]
    if args.calibration:
        conditions.append(("calibrated", args.calibration))
    if args.manifold:
        conditions.append(("trained_manifold", args.manifold))

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_results = []

    for condition_name, checkpoint in conditions:
        print(f"\n{'=' * 70}")
        print(f"  Condition: {condition_name}")
        print(f"  Prompt: \"{args.prompt[:60]}...\"")
        print(f"{'=' * 70}")

        model, curve_modules = load_model(args.model, device, checkpoint)

        # Verify scale=1 logit consistency (patched vs unpatched)
        if condition_name == "original":
            logits_clean = get_logits(model, ids)
            handle = scale_head_hook(model, 0, 0, 1.0)
            logits_patched = get_logits(model, ids)
            handle.remove()
            diff = (logits_clean - logits_patched).abs().max().item()
            print(f"  Scale-1 logit check: max diff = {diff:.2e} "
                  f"({'OK' if diff < 1e-4 else 'MISMATCH'})")

        # Find sensitive head (independently per model)
        if args.block is not None and args.head is not None:
            bi, hi = args.block, args.head
            print(f"  Using specified head: Block {bi} Head {hi}")
        else:
            print(f"  Finding most sensitive head...")
            bi, hi, score = find_sensitive_head(
                model, tokenizer, ids, device)
            print(f"  Auto-selected: Block {bi} Head {hi} (score={score:.4f})")

        # Run sweep
        sweep = run_scaling_sweep(
            model, tokenizer, ids, device, bi, hi,
            scales, args.max_tokens, curve_modules)

        # Get baseline text for comparison
        baseline_text = None
        for s, t in sweep:
            if s == 1.0:
                baseline_text = t
                break

        for scale, text in sweep:
            changed = text[:50] != baseline_text[:50] if baseline_text else None
            label = "BASELINE" if scale == 1.0 else f"scale={scale}"
            status = "" if scale == 1.0 else (" [CHANGED]" if changed else " [same]")
            print(f"  {label:>12s}: {text[:120]}{status}")

            record = {
                "condition": condition_name,
                "checkpoint": checkpoint,
                "model": args.model,
                "prompt": args.prompt,
                "block": bi,
                "head": hi,
                "scale": scale,
                "generation": text,
                "generation_length": len(text.split()),
                "changed_from_baseline": changed,
                "max_tokens": args.max_tokens,
                "device": str(device),
            }
            all_results.append(record)

        del model
        if curve_modules:
            del curve_modules
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save all results
    with open(args.output, 'w') as f:
        for record in all_results:
            f.write(json.dumps(record) + '\n')

    print(f"\n{'=' * 70}")
    print(f"  Saved {len(all_results)} results to {args.output}")

    # Summary
    print(f"\n  SUMMARY")
    print(f"  {'Condition':<20s} {'Scales that changed behavior'}")
    print(f"  {'─'*20} {'─'*40}")
    for cond in set(r['condition'] for r in all_results):
        changed_scales = [r['scale'] for r in all_results
                         if r['condition'] == cond and r['changed_from_baseline']]
        print(f"  {cond:<20s} {changed_scales if changed_scales else 'none'}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
