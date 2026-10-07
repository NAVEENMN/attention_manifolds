#!/usr/bin/env python3
"""
Distill standard attention → curve attention, block by block.

For each block:
  1. Collect attention module inputs and outputs from the teacher
  2. Train CurveAttention to match teacher outputs
  3. Replace the attention module with the trained curve version
  4. Measure PPL after each block replacement

Usage:
  python distill_attention.py --model meta-llama/Llama-3.2-1B-Instruct
  python distill_attention.py --model meta-llama/Llama-3.2-1B-Instruct --blocks 0 1 2 3
  python distill_attention.py --model meta-llama/Llama-3.2-1B-Instruct --R 64
"""

import argparse
import math
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from datasets import load_dataset
from manifold_attention import CurveAttention, FixedSampleAttention


def eval_ppl(model, eval_ids, device):
    model.eval()
    ids = eval_ids.to(device)
    with torch.no_grad():
        loss = model(ids, labels=ids, use_cache=False).loss.float().item()
    return math.exp(loss)


def collect_attention_data(model, tokenizer, device, n_samples=500, block_size=128):
    """Collect per-block attention inputs (after norm) and outputs (before residual).

    For each block, captures:
      - input to attention (post-layernorm hidden states)
      - output of attention (before residual add)
      - position embeddings (for RoPE)
    """
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = " ".join(t for t in dataset["text"] if t.strip())
    tokens = tokenizer.encode(text, add_special_tokens=False)[:n_samples * block_size]

    sequences = []
    for i in range(0, len(tokens) - block_size, block_size):
        sequences.append(tokens[i:i + block_size])
        if len(sequences) >= n_samples:
            break
    sequences = torch.tensor(sequences).to(device)

    n_layers = model.config.num_hidden_layers
    block_data = {i: {'attn_inputs': [], 'attn_outputs': [], 'pos_embeds': []}
                  for i in range(n_layers)}

    # Hook into each block's self_attn to capture input/output
    hooks = []
    for i in range(n_layers):
        block = model.model.layers[i]
        attn = block.self_attn

        def make_pre_hook(block_idx, block_ref):
            def hook_fn(module, args, kwargs):
                # The input to self_attn is the normalized hidden state
                hs = args[0] if len(args) > 0 else kwargs.get('hidden_states')
                pos = kwargs.get('position_embeddings', args[1] if len(args) > 1 else None)
                block_data[block_idx]['attn_inputs'].append(hs.detach().cpu())
                if pos is not None:
                    block_data[block_idx]['pos_embeds'].append(
                        (pos[0].detach().cpu(), pos[1].detach().cpu()))
            return hook_fn

        def make_post_hook(block_idx):
            def hook_fn(module, args, output):
                # output is (attn_output, attn_weights) or (attn_output, attn_weights, past_kv)
                attn_out = output[0] if isinstance(output, tuple) else output
                block_data[block_idx]['attn_outputs'].append(attn_out.detach().cpu())
            return hook_fn

        h1 = attn.register_forward_pre_hook(make_pre_hook(i, block), with_kwargs=True)
        h2 = attn.register_forward_hook(make_post_hook(i))
        hooks.extend([h1, h2])

    print(f"  Collecting attention data from {len(sequences)} sequences...")
    with torch.no_grad():
        for i in range(0, len(sequences), 4):
            batch = sequences[i:i + 4]
            model(batch, use_cache=False)

    for h in hooks:
        h.remove()

    # Concatenate
    for i in range(n_layers):
        block_data[i]['attn_inputs'] = torch.cat(block_data[i]['attn_inputs'], dim=0)
        block_data[i]['attn_outputs'] = torch.cat(block_data[i]['attn_outputs'], dim=0)
        n_tok = block_data[i]['attn_inputs'].shape[0] * block_data[i]['attn_inputs'].shape[1]
        print(f"    Block {i}: {n_tok:,} tokens")

    return block_data


def distill_one_block(block_idx, block_data, model, device, R=64,
                      epochs=30, lr=1e-3, batch_size=8):
    """Train CurveAttention to match one block's standard attention output."""

    attn = model.model.layers[block_idx].self_attn
    nh = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    ng = nh // nkv
    hd = model.config.head_dim

    inputs = block_data['attn_inputs']   # [N, seq, hidden]
    targets = block_data['attn_outputs']  # [N, seq, hidden]
    pos_embeds = block_data['pos_embeds']  # list of (cos, sin) tuples
    n_samples = inputs.shape[0]

    # Create CurveAttention module
    curve_attn = CurveAttention(hd, R).to(device)
    n_params = sum(p.numel() for p in curve_attn.parameters())
    print(f"  Block {block_idx}: CurveAttention (R={R}, {n_params:,} params)")

    optimizer = torch.optim.Adam(curve_attn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss = float('inf')
    best_state = None

    # Precompute shared position embeddings and causal mask
    seq_len = inputs.shape[1]
    # Use the first stored position embedding (all are identical for fixed-length sequences)
    rope_cos, rope_sin = pos_embeds[0]
    rope_cos, rope_sin = rope_cos.to(device), rope_sin.to(device)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        perm = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            x = inputs[idx].to(device)
            target = targets[idx].to(device)

            bsz, ql, _ = x.size()
            q = attn.q_proj(x).view(bsz, ql, nh, hd).transpose(1, 2)
            k = attn.k_proj(x).view(bsz, ql, nkv, hd).transpose(1, 2)
            v = attn.v_proj(x).view(bsz, ql, nkv, hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

            if ng > 1:
                k = k.repeat_interleave(ng, dim=1)
                v = v.repeat_interleave(ng, dim=1)

            curve_out = curve_attn(q, k, v, mask=causal_mask)
            curve_out = curve_out.transpose(1, 2).reshape(bsz, ql, -1)
            curve_out = attn.o_proj(curve_out)

            loss = F.mse_loss(curve_out, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(curve_attn.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        avg_loss = total_loss / steps

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in curve_attn.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                x = inputs[:4].to(device)
                target = targets[:4].to(device)
                bsz, ql, _ = x.size()
                q = attn.q_proj(x).view(bsz, ql, nh, hd).transpose(1, 2)
                k = attn.k_proj(x).view(bsz, ql, nkv, hd).transpose(1, 2)
                v = attn.v_proj(x).view(bsz, ql, nkv, hd).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
                if ng > 1:
                    k = k.repeat_interleave(ng, dim=1)
                    v = v.repeat_interleave(ng, dim=1)
                pred = curve_attn(q, k, v, mask=causal_mask)
                pred = pred.transpose(1, 2).reshape(bsz, ql, -1)
                pred = attn.o_proj(pred)
                cos_sim = F.cosine_similarity(pred.flatten(0, 1), target.flatten(0, 1), dim=-1).mean()

            print(f"    Epoch {epoch+1:>3d}/{epochs}: MSE={avg_loss:.6f}, cos={cos_sim:.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    # Restore best state
    if best_state is not None:
        curve_attn.load_state_dict(best_state)

    return curve_attn, best_loss


def patch_block_with_curve(model, block_idx, curve_attn, device):
    """Replace one block's attention with trained CurveAttention."""
    attn = model.model.layers[block_idx].self_attn
    nh = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    ng = nh // nkv
    hd = model.config.head_dim

    # Register as submodule
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
            if attention_mask is not None:
                mask = attention_mask[:, :, :, :k.shape[-2]]
            else:
                mask = None
            out = ca(q, k, v, mask=mask)
            out = out.transpose(1, 2).reshape(bsz, ql, -1)
            return am.o_proj(out), None
        return fwd

    attn.forward = make_fwd(attn, curve_attn, nh, nkv, ng, hd)


def main():
    parser = argparse.ArgumentParser(description="Distill attention → curve attention")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--blocks", type=int, nargs='*', default=None,
                        help="Blocks to distill (default: all)")
    parser.add_argument("--R", type=int, default=64, help="Sample points per query")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Distill Attention → Curve Attention")
    print(f"  Model: {args.model}")
    print(f"  R: {args.R}, Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation='eager').to(device)
    model.config.use_cache = False
    model.eval()

    n_layers = model.config.num_hidden_layers
    blocks_to_distill = args.blocks if args.blocks else list(range(n_layers))

    # Eval data
    raw_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    val_text = " ".join(raw_val["text"][:200])
    eval_ids = torch.tensor(
        tokenizer.encode(val_text, add_special_tokens=False)[:512]
    ).unsqueeze(0)

    base_ppl = eval_ppl(model, eval_ids, device)
    print(f"  Baseline PPL: {base_ppl:.2f}")

    # Collect attention data
    block_data = collect_attention_data(model, tokenizer, device, args.n_samples)

    # Freeze teacher parameters
    for p in model.parameters():
        p.requires_grad = False

    # Distill block by block
    curve_modules = {}
    block_losses = {}
    for block_idx in blocks_to_distill:
        print(f"\n{'─' * 50}")
        curve_attn, loss = distill_one_block(
            block_idx, block_data[block_idx], model, device,
            R=args.R, epochs=args.epochs, lr=args.lr)
        block_losses[block_idx] = loss

        # Patch the model
        patch_block_with_curve(model, block_idx, curve_attn, device)
        curve_modules[block_idx] = curve_attn

        # Measure PPL after this block is replaced
        ppl = eval_ppl(model, eval_ids, device)
        delta = ppl - base_ppl
        print(f"  Block {block_idx} replaced → PPL: {ppl:.2f} (base: {base_ppl:.2f}, delta: {delta:+.2f})")

    # Save
    model_name = args.model.split('/')[-1]
    out_dir = os.path.join(args.output_dir, f"{model_name}-curve-attn")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "curve_attention.pt")

    torch.save({
        'curve_states': {i: m.cpu().state_dict() for i, m in curve_modules.items()},
        'config': {
            'model': args.model,
            'R': args.R,
            'head_dim': model.config.head_dim,
            'blocks': blocks_to_distill,
        },
        'losses': block_losses,
        'base_ppl': base_ppl,
        'final_ppl': ppl,
    }, save_path)

    # Summary
    print(f"\n{'═' * 50}")
    print(f"  SUMMARY")
    print(f"{'═' * 50}")
    print(f"  Baseline PPL: {base_ppl:.2f}")
    print(f"  Final PPL:    {ppl:.2f} (delta: {ppl - base_ppl:+.2f})")
    print(f"  Blocks replaced: {len(blocks_to_distill)}")
    total_params = sum(sum(p.numel() for p in m.parameters()) for m in curve_modules.values())
    print(f"  Curve params: {total_params:,}")
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
