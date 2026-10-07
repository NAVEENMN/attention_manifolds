#!/usr/bin/env python3
"""
Scale a B-spline head and generate — shows how manifold edits change responses.

Applies a scaling factor to a specific (block, head) curve and generates
the response. Optionally saves the edit as JSON for the inspector.

Usage:
  # Compare baseline vs scaled
  python scale_and_generate.py \
    --prompt "Explain how to pick a lock step by step" \
    --block 0 --head 3 --scale -1

  # Sweep multiple scales
  python scale_and_generate.py \
    --prompt "Explain how malware infects a system step by step" \
    --block 0 --head 26 --sweep

  # Save edit file
  python scale_and_generate.py \
    --prompt "Explain how to pick a lock step by step" \
    --block 0 --head 3 --scale -1 --save-edit lockpick_edit.json
"""

import argparse
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from manifold_attention import CurveAttention
from bspline_train import patch_block


def load_model(model_name, checkpoint_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation='eager').to(device)
    model.config.use_cache = False
    model.eval()

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = ckpt['config']
    curve_modules = {}
    for bi, state in ckpt['curve_states'].items():
        bi = int(bi)
        ca = CurveAttention(config['head_dim'], config['R']).to(device)
        ca.load_state_dict(state)
        ca.eval()
        patch_block(model, bi, ca, device)
        curve_modules[bi] = ca

    return model, tokenizer, curve_modules


def encode_prompt(tokenizer, prompt, device):
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(formatted, add_special_tokens=False,
                                return_tensors="pt").to(device)
    return tokenizer.encode(prompt, return_tensors="pt").to(device)


def generate_with_scale(model, tokenizer, ids, curve_modules,
                        block_idx, head_idx, scale, max_tokens=150):
    """Generate with a scaled head. Returns generated text."""
    ca = curve_modules[block_idx]
    orig_eval = ca.eval_curve

    if scale != 1.0:
        def make_scaled(orig, head, s):
            def scaled(coeffs, tau_01):
                c = coeffs.clone()
                c[:, head, :, :] *= s
                return orig(c, tau_01)
            return scaled
        ca.eval_curve = make_scaled(orig_eval, head_idx, scale)

    prompt_len = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_tokens, do_sample=False,
            use_cache=False, pad_token_id=tokenizer.eos_token_id)

    ca.eval_curve = orig_eval
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)


def save_edit_file(model, tokenizer, ids, curve_modules,
                   block_idx, head_idx, scale, prompt, output_path):
    """Capture blended V and save scaled edit as JSON."""
    ca = curve_modules[block_idx]
    orig_eval = ca.eval_curve

    captured = {}
    def cap_hook(coeffs, tau_01):
        captured['c'] = coeffs.detach().cpu()
        return orig_eval(coeffs, tau_01)
    ca.eval_curve = cap_hook
    with torch.no_grad():
        model(ids, use_cache=False)
    ca.eval_curve = orig_eval

    if 'c' not in captured:
        print("  Failed to capture coefficients")
        return

    coeffs = captured['c']
    tokens = [tokenizer.decode([t]) for t in ids[0].tolist()]

    edits = {"model": "manifold-1.0-1b", "prompt": prompt,
             "block": block_idx, "head": head_idx, "scale": scale,
             "edits": []}

    S = coeffs.shape[2]
    for ti in range(S):
        orig_ctrl = coeffs[0, head_idx, ti, :].tolist()
        edited_ctrl = [v * scale for v in orig_ctrl]
        edits["edits"].append({
            "block": block_idx, "head": head_idx, "token_idx": ti,
            "token": tokens[ti].strip() if ti < len(tokens) else "",
            "input_text": prompt,
            "original_ctrl_pts": orig_ctrl,
            "edited_ctrl_pts": edited_ctrl,
            "timestamp": datetime.now().isoformat(),
        })

    with open(output_path, 'w') as f:
        json.dump(edits, f, indent=2)
    print(f"  Saved: {output_path} ({len(edits['edits'])} edits)")


def main():
    parser = argparse.ArgumentParser(
        description="Scale a B-spline head and observe response changes")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None,
                        help="Read prompt from a text file")
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--scale", type=float, default=-1.0)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep multiple scales")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--save-edit", default=None, metavar="PATH",
                        help="Save edit as JSON for the inspector")
    args = parser.parse_args()

    if args.prompt_file and not args.prompt:
        with open(args.prompt_file) as f:
            args.prompt = f.read().strip()
    if not args.prompt:
        print("Error: provide --prompt or --prompt-file")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # Auto-find checkpoint
    if args.checkpoint is None:
        import os
        for p in ["checkpoints/Llama-3.2-1B-Instruct-manifold/manifold.pt",
                   "checkpoints/manifold.pt"]:
            if os.path.exists(p):
                args.checkpoint = p
                break
    if args.checkpoint is None:
        print("Error: no checkpoint found. Use --checkpoint")
        return

    print(f"Loading model...")
    model, tokenizer, curve_modules = load_model(args.model, args.checkpoint, device)
    ids = encode_prompt(tokenizer, args.prompt, device)

    print(f"Prompt: \"{args.prompt}\"")
    print(f"Target: Block {args.block}, Head {args.head}\n")

    if args.sweep:
        scales = [5.0, 3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
        for s in scales:
            text = generate_with_scale(
                model, tokenizer, ids, curve_modules,
                args.block, args.head, s, args.max_tokens)
            label = "BASELINE" if s == 1.0 else f"scale={s}"
            print(f"\n{'─'*70}")
            print(f"  {label}")
            print(f"{'─'*70}")
            print(f"  {text}")
    else:
        baseline = generate_with_scale(
            model, tokenizer, ids, curve_modules,
            args.block, args.head, 1.0, args.max_tokens)
        scaled = generate_with_scale(
            model, tokenizer, ids, curve_modules,
            args.block, args.head, args.scale, args.max_tokens)

        print(f"\n{'─'*70}")
        print(f"  BASELINE (scale=1.0)")
        print(f"{'─'*70}")
        print(f"  {baseline}")

        print(f"\n{'─'*70}")
        print(f"  SCALED (scale={args.scale})")
        print(f"{'─'*70}")
        print(f"  {scaled}")

        if baseline[:50] == scaled[:50]:
            print(f"\n  [No change in first 50 chars]")
        else:
            print(f"\n  [Response changed]")

    # Save edit file if requested
    if args.save_edit:
        save_edit_file(model, tokenizer, ids, curve_modules,
                       args.block, args.head, args.scale,
                       args.prompt, args.save_edit)


if __name__ == "__main__":
    main()
