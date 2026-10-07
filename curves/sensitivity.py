#!/usr/bin/env python3
"""
Sensitivity analysis: find which B-spline heads control model behavior.

For each (block, head), perturbs the control points and measures
the KL divergence of next-token predictions. High KL = high influence.

Usage:
  python sensitivity.py --prompt "Explain how malware infects a system"
  python sensitivity.py --prompt "Explain how to pick a lock step by step" --top 10
"""

import argparse
import torch
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


def compute_sensitivity(model, tokenizer, curve_modules, prompt, device, eps=0.1):
    """Compute sensitivity of each (block, head) for a given prompt."""
    # Encode with chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(formatted, add_special_tokens=False,
                               return_tensors="pt").to(device)
    else:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    nh = model.config.num_attention_heads

    with torch.no_grad():
        base_probs = torch.softmax(
            model(ids, use_cache=False).logits[0, -1, :].float(), dim=-1)

    scores = []
    for bi, ca in sorted(curve_modules.items()):
        orig_eval = ca.eval_curve
        for h in range(nh):
            def make_perturbed(orig, head, epsilon):
                def perturbed(coeffs, tau_01):
                    if coeffs.dim() == 4 and coeffs.shape[1] > head:
                        c = coeffs.clone()
                        c[:, head, :, :] += epsilon
                        return orig(c, tau_01)
                    return orig(coeffs, tau_01)
                return perturbed

            ca.eval_curve = make_perturbed(orig_eval, h, eps)
            with torch.no_grad():
                pert_probs = torch.softmax(
                    model(ids, use_cache=False).logits[0, -1, :].float(), dim=-1)
            kl = (base_probs * (base_probs.log() - pert_probs.log())).sum().item()
            scores.append((bi, h, abs(kl) / eps))
        ca.eval_curve = orig_eval

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for manifold attention")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None,
                        help="Read prompt from a text file")
    parser.add_argument("--top", type=int, default=20)
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

    print(f"Prompt: \"{args.prompt}\"")
    print(f"Computing sensitivity...\n")

    scores = compute_sensitivity(model, tokenizer, curve_modules, args.prompt, device)

    print(f"  {'#':>3}  {'Block':>5}  {'Head':>4}  {'Score':>9}")
    print(f"  {'─'*3}  {'─'*5}  {'─'*4}  {'─'*9}  {'─'*20}")

    max_score = scores[0][2] if scores else 1.0
    for rank, (bi, h, score) in enumerate(scores[:args.top]):
        bar_len = int(20 * score / max_score)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"  {rank+1:>3}  {bi:>5}  {h:>4}  {score:>9.4f}  {bar}")

    top = scores[0]
    print(f"\n  Most sensitive: Block {top[0]}, Head {top[1]}")
    print(f"  To steer: python scale_and_generate.py --prompt \"{args.prompt}\" "
          f"--block {top[0]} --head {top[1]} --scale -1")


if __name__ == "__main__":
    main()
