#!/usr/bin/env python3
"""
B-Spline Manifold Training.

Trains manifold attention parameters (tau_proj + sample_proj) with
next-token CE. All other model weights stay frozen.

Works on single GPU or multi-GPU with accelerate.

Usage:
  # Single GPU
  python bspline_train.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --checkpoint checkpoints/calibration.pt \
    --epochs 10 --lr 1e-4

  # Multi-GPU
  accelerate launch bspline_train.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --checkpoint checkpoints/calibration.pt \
    --epochs 10 --lr 1e-4 --batch-size 4 --grad-accum 4
"""

import argparse
import math
import time
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from datasets import load_dataset
from manifold_attention import CurveAttention


def eval_ppl_strided(model, tokenizer, text, device, max_length=512, stride=256):
    """Perplexity over multiple strided windows."""
    model.eval()
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encodings.input_ids[0]
    seq_len = input_ids.size(0)
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        window = input_ids[begin:end].unsqueeze(0).to(device)
        labels = window.clone()
        mask_len = max(begin, prev_end) - begin
        if mask_len > 0:
            labels[0, :mask_len] = -100
        with torch.no_grad():
            output = model(window, labels=labels, use_cache=False)
            n_scored = (labels[:, 1:] != -100).sum().item()
            if n_scored > 0:
                total_nll += output.loss.float().item() * n_scored
                total_tokens += n_scored
        prev_end = end
        if end == seq_len:
            break
    return math.exp(total_nll / total_tokens), total_tokens


def patch_block(model, block_idx, curve_attn, device):
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


def main():
    parser = argparse.ArgumentParser(description="B-Spline Manifold Training")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--checkpoint", required=True,
                        help="Calibration checkpoint from calibration.py")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-GPU batch size")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=50_000_000)
    parser.add_argument("--data", default="wikipedia",
                        choices=["wikipedia", "gsm8k"],
                        help="Training data source")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    # Auto-detect: use accelerate if available, else single GPU/MPS/CPU
    use_accel = False
    try:
        from accelerate import Accelerator
        accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
        device = accelerator.device
        is_main = accelerator.is_main_process
        use_accel = True
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        is_main = True

    if is_main:
        if args.wandb:
            import wandb
            wandb.init(project="spline-routing",
                       name=f"formation-{args.model.split('/')[-1]}-{args.block_size}",
                       config=vars(args),
                       tags=["formation", args.model.split('/')[-1]])

        n_gpus = accelerator.num_processes if use_accel else 1
        print(f"B-Spline Manifold Training")
        print(f"  Model: {args.model}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Device: {device} ({'accelerate, ' + str(n_gpus) + ' GPUs' if use_accel else 'single'})")
        print(f"  Data: {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation='eager').to(device)
    model.config.use_cache = False

    # Load calibrated manifold attention
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    config = ckpt['config']
    R, hd = config['R'], config['head_dim']

    curve_modules = {}
    curve_params = []
    for block_idx, state in ckpt['curve_states'].items():
        block_idx = int(block_idx)
        ca = CurveAttention(hd, R).to(device)
        ca.load_state_dict(state)
        patch_block(model, block_idx, ca, device)
        curve_modules[block_idx] = ca
        curve_params.extend(list(ca.parameters()))

    n_blocks = len(curve_modules)
    total_curve_params = sum(p.numel() for p in curve_params)
    if is_main:
        print(f"  {n_blocks} manifold blocks (R={R})")
        print(f"  Manifold params: {total_curve_params:,}")

    # Load training data
    eos_id = tokenizer.eos_token_id
    train_ids = []

    if args.data == "wikipedia":
        if is_main:
            print("\nLoading Wikipedia (clean)...")
        raw = load_dataset("wikimedia/wikipedia", "20231101.en",
                           split="train", streaming=True)
        n_articles = 0
        for item in raw:
            text = item["text"].strip()
            if len(text) < 200:
                continue
            train_ids.extend(tokenizer.encode(text, add_special_tokens=False))
            train_ids.append(eos_id)
            n_articles += 1
            if len(train_ids) >= args.max_tokens:
                break
            if is_main and n_articles % 5000 == 0:
                print(f"    {n_articles:,} articles, {len(train_ids):,} tokens...")
        if is_main:
            print(f"  {n_articles:,} articles, {len(train_ids):,} tokens")

    elif args.data == "gsm8k":
        if is_main:
            print("\nLoading GSM8K...")
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        for item in gsm:
            messages = [
                {"role": "system", "content": "Solve step by step."},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False)
                ids = tokenizer.encode(formatted, add_special_tokens=False)
            else:
                text = f"Q: {item['question']}\nA: {item['answer']}"
                ids = tokenizer.encode(text, add_special_tokens=False)
            train_ids.extend(ids[:args.block_size])
            train_ids.append(eos_id)
        if is_main:
            print(f"  {len(gsm)} problems, {len(train_ids):,} tokens")

    # Create sequences
    sequences = []
    for i in range(0, len(train_ids) - args.block_size + 1, args.block_size):
        sequences.append(train_ids[i:i + args.block_size])
    sequences = torch.tensor(sequences)
    if is_main:
        print(f"  Sequences: {len(sequences):,} × {args.block_size} tokens")

    min_seqs = args.batch_size * (accelerator.num_processes if use_accel else 1)
    if len(sequences) < min_seqs:
        raise ValueError(f"Not enough sequences ({len(sequences)}) for batch")

    # Eval data
    try:
        raw_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    except Exception:
        raw_val = load_dataset("wikitext", "wikitext-2-raw-v1")
    val_text = "\n".join(t for t in raw_val["validation"]["text"] if t.strip())

    # Freeze all except manifold params
    for p in model.parameters():
        p.requires_grad = False
    for p in curve_params:
        p.requires_grad = True

    optimizer = torch.optim.Adam(curve_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dataset = torch.utils.data.TensorDataset(sequences)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if use_accel:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        unwrapped = accelerator.unwrap_model(model)
    else:
        unwrapped = model

    calibrated_ppl, n_val = eval_ppl_strided(unwrapped, tokenizer, val_text, device)
    if is_main:
        print(f"  Calibrated PPL: {calibrated_ppl:.4f} ({n_val:,} val tokens)")

    best_ppl = calibrated_ppl
    model_name = args.model.split('/')[-1]
    out_dir = os.path.join(args.output_dir, f"{model_name}-manifold")
    if is_main:
        os.makedirs(out_dir, exist_ok=True)

    def get_tau_stats():
        norms = [ca.tau_proj.weight.data.abs().max().item()
                 for ca in curve_modules.values()]
        return max(norms), sum(norms) / len(norms)

    if is_main:
        tau_max_init, _ = get_tau_stats()
        n_gpus = accelerator.num_processes if use_accel else 1
        print(f"\n{'═' * 60}")
        print(f"  Manifold Training ({n_gpus} GPU{'s' if n_gpus > 1 else ''})")
        print(f"  tau_max init: {tau_max_init:.6f}")
        print(f"{'═' * 60}\n")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        steps = 0
        start = time.time()

        for (batch,) in dataloader:
            if not use_accel:
                batch = batch.to(device)
                loss = model(input_ids=batch, labels=batch, use_cache=False).loss.float()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(curve_params, 1.0)
                optimizer.step()
            else:
                with accelerator.accumulate(model):
                    loss = model(input_ids=batch, labels=batch, use_cache=False).loss.float()
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(curve_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1

            if steps % args.log_interval == 0 and is_main:
                tau_max, _ = get_tau_stats()
                elapsed = time.time() - start
                print(f"  Epoch {epoch+1} | Step {steps:>5d} | "
                      f"Loss {total_loss/steps:.4f} | "
                      f"tau_max={tau_max:.4f} | {elapsed:.0f}s")
                if args.wandb:
                    import wandb
                    wandb.log({"loss": total_loss / steps, "tau_max": tau_max},
                              step=epoch * 10000 + steps)

        scheduler.step()
        if use_accel:
            accelerator.wait_for_everyone()

        # Eval
        if use_accel:
            unwrapped = accelerator.unwrap_model(model)
        ppl, _ = eval_ppl_strided(unwrapped, tokenizer, val_text, device)
        tau_max, tau_avg = get_tau_stats()
        elapsed = time.time() - start

        if is_main:
            print(f"  Epoch {epoch+1} done | Loss {total_loss/steps:.4f} | "
                  f"PPL {ppl:.4f} (calibrated: {calibrated_ppl:.4f}) | "
                  f"tau_max={tau_max:.4f} | {elapsed:.0f}s")

            # Sample generation
            prompts = ["The theory of general relativity",
                       "In the history of ancient Rome",
                       "The process of photosynthesis involves"]
            test_prompt = prompts[epoch % len(prompts)]
            test_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = unwrapped.generate(
                    test_ids, max_new_tokens=80, do_sample=False,
                    use_cache=False, pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)
            print(f"  Sample: \"{gen_text[:150]}\"")

            if args.wandb:
                import wandb
                wandb.log({"eval/ppl": ppl, "eval/tau_max": tau_max},
                          step=epoch * 10000 + steps)

            if ppl < best_ppl:
                best_ppl = ppl
                save_path = os.path.join(out_dir, "manifold.pt")
                torch.save({
                    'curve_states': {bi: {k: v.cpu().clone()
                                          for k, v in ca.state_dict().items()}
                                     for bi, ca in curve_modules.items()},
                    'config': config,
                    'calibrated_ppl': calibrated_ppl,
                    'manifold_ppl': ppl,
                    'epoch': epoch + 1,
                }, save_path)
                print(f"  Saved (PPL {ppl:.4f})")

        if use_accel:
            accelerator.wait_for_everyone()

    if is_main:
        tau_final, tau_avg_final = get_tau_stats()
        print(f"\n{'═' * 60}")
        print(f"  FORMATION COMPLETE")
        print(f"  Calibrated PPL: {calibrated_ppl:.4f}")
        print(f"  Best PPL: {best_ppl:.4f}")
        print(f"  tau_max: {tau_max_init:.6f} → {tau_final:.4f}")
        print(f"  Manifold params: {total_curve_params:,}")
        print(f"{'═' * 60}")

        with open(os.path.join(out_dir, "results.json"), 'w') as f:
            json.dump({
                'calibrated_ppl': calibrated_ppl,
                'best_manifold_ppl': best_ppl,
                'tau_max_final': tau_final,
                'epochs': args.epochs,
                'data': args.data,
            }, f, indent=2)

        if args.wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
