"""Train coordinate-wise Q/K spline curves from a calibrated checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spline_qk import load_splines, save_splines, SplineProjection


def text_blocks(path, tokenizer, block_size, max_blocks):
    raw = Path(path).read_bytes()
    ids = tokenizer(raw.decode('utf-8'), add_special_tokens=False)['input_ids']
    limit = min(len(ids) // block_size, max_blocks)
    if limit < 1:
        raise ValueError(f'{path}: fewer than {block_size} tokens')
    data = torch.tensor(ids[:limit * block_size], dtype=torch.long).view(limit, block_size)
    return data, hashlib.sha256(raw).hexdigest()


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total, count = 0.0, 0
    for batch in data.split(batch_size):
        batch = batch.to(device)
        loss = model(input_ids=batch, labels=batch, use_cache=False).loss
        n = batch.shape[0] * (batch.shape[1] - 1)
        total += loss.item() * n
        count += n
    nll = total / count
    return {'nll': nll, 'ppl': math.exp(nll) if nll < 700 else float('inf'), 'target_tokens': count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--checkpoint', required=True, help='Calibrated checkpoint directory')
    parser.add_argument('--train-text', required=True)
    parser.add_argument('--eval-text', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-train-blocks', type=int, default=4096)
    parser.add_argument('--max-eval-blocks', type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load model and calibrated curves
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation='eager'
    ).to(args.device).eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    meta = load_splines(args.checkpoint, model)
    model.eval()

    # Collect trainable params (only spline coefficients)
    params = [m.coefficients for m in model.modules() if isinstance(m, SplineProjection)]
    print(json.dumps({'trainable_parameters': sum(p.numel() for p in params),
                      'num_spline_projections': len(params)}), flush=True)

    # Load data
    training, train_hash = text_blocks(args.train_text, tokenizer, args.block_size, args.max_train_blocks)
    validation, val_hash = text_blocks(args.eval_text, tokenizer, args.block_size, args.max_eval_blocks)
    if train_hash == val_hash:
        raise ValueError('Training and validation files must be different')

    baseline = evaluate(model, validation, args.batch_size, args.device)
    print(json.dumps({'baseline': baseline, 'train_blocks': len(training),
                      'eval_blocks': len(validation)}), flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    best = float('inf')
    history = []
    train_meta = {**meta, 'train_sha256': train_hash, 'eval_sha256': val_hash,
                  'train_args': vars(args), 'baseline': baseline}

    model.eval()
    for epoch in range(args.epochs):
        order = torch.randperm(len(training))
        batches = list(order.split(args.batch_size))
        total, count = 0.0, 0
        for start in range(0, len(batches), args.grad_accum):
            group = batches[start:start + args.grad_accum]
            group_tokens = sum(len(indices) * (args.block_size - 1) for indices in group)
            optimizer.zero_grad(set_to_none=True)
            for indices in group:
                batch = training[indices].to(args.device)
                n = len(batch) * (args.block_size - 1)
                loss = model(input_ids=batch, labels=batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                (loss * (n / group_tokens)).backward()
                total += loss.detach().item() * n
                count += n
            torch.nn.utils.clip_grad_norm_(params, args.clip_grad, error_if_nonfinite=True)
            optimizer.step()
        metrics = evaluate(model, validation, args.batch_size, args.device)
        row = {'epoch': epoch + 1, 'train_nll': total / count, 'validation': metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        train_meta['history'] = history
        train_meta['epoch'] = epoch + 1
        import shutil
        if (out / 'last').exists():
            shutil.rmtree(out / 'last')
        save_splines(out / 'last', model, train_meta)
        if metrics['nll'] < best:
            best = metrics['nll']
            if (out / 'best').exists():
                shutil.rmtree(out / 'best')
            save_splines(out / 'best', model, train_meta)
        (out / 'metrics.json').write_text(json.dumps(train_meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
