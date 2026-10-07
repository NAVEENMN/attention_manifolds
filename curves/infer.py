#!/usr/bin/env python3
"""
Inference: compare base model vs manifold model, optionally with edits.

Usage:
  # Compare base vs manifold
  python infer.py --prompt "The theory of relativity" \
    --manifold checkpoints/manifold.pt

  # With edit file applied
  python infer.py --prompt "Explain how to pick a lock step by step" \
    --manifold checkpoints/manifold.pt \
    --edits lockpick_edit.json

  # Interactive mode
  python infer.py --manifold checkpoints/manifold.pt
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from manifold_attention import CurveAttention


def patch_block(model, block_idx, curve_attn):
    attn = model.model.layers[block_idx].self_attn
    nh = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    ng = nh // nkv
    hd = model.config.head_dim
    attn.curve_attention = curve_attn

    def make_fwd(am, ca, nh, nkv, ng, hd):
        def fwd(hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, cache_position=None, **kw):
            bsz, ql, _ = hidden_states.size()
            q = am.q_proj(hidden_states).view(bsz, ql, nh, hd).transpose(1, 2)
            k = am.k_proj(hidden_states).view(bsz, ql, nkv, hd).transpose(1, 2)
            v = am.v_proj(hidden_states).view(bsz, ql, nkv, hd).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if ng > 1:
                k = k.repeat_interleave(ng, dim=1)
                v = v.repeat_interleave(ng, dim=1)
            mask = attention_mask[:, :, :, :k.shape[-2]] if attention_mask is not None else None
            out = ca(q, k, v, mask=mask)
            out = out.transpose(1, 2).reshape(bsz, ql, -1)
            return am.o_proj(out), None
        return fwd

    attn.forward = make_fwd(attn, curve_attn, nh, nkv, ng, hd)


def load_manifold_model(model_name, checkpoint_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation='eager').to(device)
    model.config.use_cache = False
    model.eval()

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = ckpt['config']
    curve_modules = {}
    for block_idx, state in ckpt['curve_states'].items():
        block_idx = int(block_idx)
        ca = CurveAttention(config['head_dim'], config['R']).to(device)
        ca.load_state_dict(state)
        ca.eval()
        patch_block(model, block_idx, ca)
        curve_modules[block_idx] = ca

    return model, curve_modules


def apply_edits(edits_data, curve_modules):
    """Apply edits from JSON. Returns patches for restoration."""
    patches = []

    if "scale" in edits_data:
        bi = edits_data["block"]
        h = edits_data["head"]
        scale = edits_data["scale"]
        if bi not in curve_modules:
            return patches
        ca = curve_modules[bi]
        orig_eval = ca.eval_curve

        def make_scaled(orig, head, s):
            def scaled(coeffs, tau_01):
                c = coeffs.clone()
                c[:, head, :, :] *= s
                return orig(c, tau_01)
            return scaled

        ca.eval_curve = make_scaled(orig_eval, h, scale)
        patches.append((ca, orig_eval))

    elif "edits" in edits_data:
        by_block = {}
        for e in edits_data["edits"]:
            by_block.setdefault(e["block"], []).append(e)

        for bi, block_edits in by_block.items():
            if bi not in curve_modules:
                continue
            ca = curve_modules[bi]
            orig_eval = ca.eval_curve
            edit_map = {}
            for e in block_edits:
                edit_map[(e["head"], e["token_idx"])] = torch.tensor(
                    e["edited_ctrl_pts"], dtype=torch.float32)

            def make_patched(orig, emap, dev):
                def patched(coeffs, tau_01):
                    if coeffs.dim() < 3:
                        return orig(coeffs, tau_01)
                    c = coeffs.clone()
                    for (h, ti), ed in emap.items():
                        if c.shape[-3] > h and c.shape[-2] > ti:
                            c[..., h, ti, :] = ed.to(dev)
                    return orig(c, tau_01)
                return patched

            ca.eval_curve = make_patched(
                orig_eval, edit_map, next(ca.parameters()).device)
            patches.append((ca, orig_eval))

    return patches


def restore_edits(patches):
    for ca, orig in patches:
        ca.eval_curve = orig


def encode_prompt(tokenizer, prompt, device):
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(formatted, add_special_tokens=False,
                                return_tensors="pt").to(device)
    return tokenizer.encode(prompt, return_tensors="pt").to(device)


def generate(model, ids, tokenizer, max_tokens=150, temperature=0.0):
    kwargs = dict(max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id,
                  eos_token_id=tokenizer.eos_token_id, use_cache=False)
    if temperature == 0.0:
        kwargs['do_sample'] = False
    else:
        kwargs.update(do_sample=True, temperature=temperature, top_k=50, top_p=0.95)
    with torch.no_grad():
        out = model.generate(ids, **kwargs)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def print_comparison(prompt, results, width=70):
    print(f"\n{'─' * width}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"{'─' * width}")
    for label, text in results:
        print(f"\n  {label}:")
        words = text.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > width:
                print(line)
                line = "    " + w
            else:
                line += (" " if len(line) > 4 else "") + w
        if line.strip():
            print(line)
    print(f"\n{'─' * width}\n")


def main():
    parser = argparse.ArgumentParser(description="Inference: base vs manifold, with optional edits")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--manifold", default=None, help="Manifold checkpoint")
    parser.add_argument("--edits", default=None, help="Edit JSON file to apply")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None,
                        help="Read prompt from a text file")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # Load prompt from file if provided
    if args.prompt_file and not args.prompt:
        with open(args.prompt_file) as f:
            args.prompt = f.read().strip()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation='eager').to(device)
    base_model.config.use_cache = False
    base_model.eval()

    # Load manifold model
    manifold_model = None
    curve_modules = None
    if args.manifold:
        print("Loading manifold model...")
        manifold_model, curve_modules = load_manifold_model(
            args.model, args.manifold, device)

    # Load edits if provided
    edits_data = None
    if args.edits:
        with open(args.edits) as f:
            edits_data = json.load(f)
        n = len(edits_data.get("edits", []))
        scale = edits_data.get("scale", "N/A")
        print(f"Loaded edits: {args.edits} ({n} edits, scale={scale})")

    def run_prompt(prompt):
        ids = encode_prompt(tokenizer, prompt, device)
        results = []

        # Base
        base_out = generate(base_model, ids, tokenizer,
                            args.max_tokens, args.temperature)
        results.append(("BASE MODEL", base_out))

        # Manifold (no edits)
        if manifold_model:
            man_out = generate(manifold_model, ids, tokenizer,
                               args.max_tokens, args.temperature)
            results.append(("MANIFOLD", man_out))

        # Manifold with edits
        if manifold_model and edits_data:
            patches = apply_edits(edits_data, curve_modules)
            edited_out = generate(manifold_model, ids, tokenizer,
                                  args.max_tokens, args.temperature)
            restore_edits(patches)
            results.append(("MANIFOLD + EDITS", edited_out))

        print_comparison(prompt, results)

    if args.prompt:
        run_prompt(args.prompt)
    else:
        print("\nInteractive mode. Type a prompt, 'quit' to exit.\n")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in ('quit', 'exit', 'q'):
                break
            if not prompt.strip():
                continue
            run_prompt(prompt)


if __name__ == "__main__":
    main()
