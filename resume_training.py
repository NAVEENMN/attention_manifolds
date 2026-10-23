"""Resume spline training from a trained checkpoint with OpenWebText data.

Loads the current best trained splines and continues optimizing for more epochs.
Includes data cleaning filters to remove low-quality samples.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from spline_qk import load_splines, save_splines, SplineProjection


def clean_text(text):
    """Return True if sample passes quality filters."""
    if len(text.split()) < 20:
        return False
    # HTML artifacts
    if any(t in text for t in ('<div', '<script', '<span', '</p>', '<table')):
        return False
    # Non-alpha junk
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    if len(text) > 0 and alpha / len(text) < 0.7:
        return False
    # Repetitive
    words = text.split()
    if len(words) > 10 and len(set(words)) / len(words) < 0.3:
        return False
    # URL-heavy
    if text.count('http') > 5:
        return False
    return True


def prepare_openwebtext(tokenizer, block_size, max_blocks, seed=42):
    """Stream OpenWebText, clean, tokenize, and return blocks."""
    print('Loading OpenWebText (streaming)...')
    ds = load_dataset('Skylion007/openwebtext', split='train', streaming=True)

    tokens_needed = max_blocks * block_size * 2  # 2x buffer for cleaning losses
    all_text = []
    total_chars = 0
    kept = 0
    skipped = 0

    for row in ds:
        text = row['text']
        if clean_text(text):
            all_text.append(text)
            total_chars += len(text)
            kept += 1
        else:
            skipped += 1
        # Estimate: ~4 chars per token, need tokens_needed tokens
        if total_chars > tokens_needed * 4:
            break

    print(f'  Kept {kept} samples, skipped {skipped} ({100*skipped/max(kept+skipped,1):.1f}% filtered)')

    combined = '\n\n'.join(all_text)
    sha = hashlib.sha256(combined.encode()).hexdigest()
    ids = tokenizer(combined, add_special_tokens=False)['input_ids']
    n = min(len(ids) // block_size, max_blocks)
    if n < 1:
        raise ValueError(f'Not enough tokens: got {len(ids)}, need {block_size}')
    data = torch.tensor(ids[:n * block_size], dtype=torch.long).view(n, block_size)
    print(f'  {len(ids)} tokens -> {n} blocks of {block_size}')
    return data, sha


def prepare_validation(tokenizer, block_size, max_blocks):
    """Use WikiText-2 validation as a clean, independent eval set."""
    print('Loading WikiText-2 validation...')
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')
    text = '\n\n'.join(ds['text'])
    sha = hashlib.sha256(text.encode()).hexdigest()
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    n = min(len(ids) // block_size, max_blocks)
    if n < 1:
        raise ValueError('Not enough validation tokens')
    data = torch.tensor(ids[:n * block_size], dtype=torch.long).view(n, block_size)
    print(f'  {n} blocks of {block_size}')
    return data, sha


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


def find_max_batch_size(model, block_size, device, max_try=16):
    """Binary search for largest batch size that fits in GPU memory."""
    print('Finding optimal batch size...')
    best = 1
    for bs in [2, 4, 6, 8, 10, 12, 14, 16]:
        if bs > max_try:
            break
        try:
            torch.cuda.empty_cache()
            dummy = torch.randint(0, 100, (bs, block_size), device=device)
            loss = model(input_ids=dummy, labels=dummy, use_cache=False).loss
            loss.backward()
            model.zero_grad(set_to_none=True)
            del dummy, loss
            torch.cuda.empty_cache()
            best = bs
            print(f'  batch_size={bs}: OK')
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'  batch_size={bs}: OOM')
            break
    print(f'  Using batch_size={best}')
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--checkpoint', required=True, help='Trained checkpoint to resume from')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--block-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=0, help='0 = auto-detect')
    parser.add_argument('--grad-accum', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-train-blocks', type=int, default=4096)
    parser.add_argument('--max-eval-blocks', type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load model and trained splines
    print(f'Loading {args.model} + trained splines...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation='eager'
    ).to(args.device).eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    meta = load_splines(args.checkpoint, model)
    model.eval()

    # Collect trainable params
    params = [m.coefficients for m in model.modules() if isinstance(m, SplineProjection)]
    n_params = sum(p.numel() for p in params)
    print(json.dumps({'trainable_parameters': n_params,
                      'num_spline_projections': len(params)}), flush=True)

    # Auto-detect batch size
    if args.batch_size == 0:
        args.batch_size = find_max_batch_size(model, args.block_size, args.device)

    # Adjust grad_accum to keep effective batch ~16
    effective = args.batch_size * args.grad_accum
    print(f'Effective batch: {args.batch_size} x {args.grad_accum} = {effective} blocks')

    # Load data
    training, train_hash = prepare_openwebtext(tokenizer, args.block_size, args.max_train_blocks, args.seed)
    validation, val_hash = prepare_validation(tokenizer, args.block_size, args.max_eval_blocks)

    baseline = evaluate(model, validation, args.batch_size, args.device)
    print(json.dumps({'resumed_from_ppl': baseline, 'train_blocks': len(training),
                      'eval_blocks': len(validation)}), flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    best = baseline['nll']
    history = meta.get('history', [])
    start_epoch = meta.get('epoch', 0)
    train_meta = {**meta,
                  'resumed_from': args.checkpoint,
                  'resume_train_sha256': train_hash,
                  'resume_eval_sha256': val_hash,
                  'resume_data': 'OpenWebText (cleaned)',
                  'resume_eval_data': 'WikiText-2 validation',
                  'resume_args': vars(args),
                  'resumed_baseline': baseline}

    model.eval()
    for epoch in range(args.epochs):
        actual_epoch = start_epoch + epoch + 1
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
        row = {'epoch': actual_epoch, 'train_nll': total / count, 'validation': metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        train_meta['history'] = history
        train_meta['epoch'] = actual_epoch
        import shutil
        if (out / 'last').exists():
            shutil.rmtree(out / 'last')
        save_splines(out / 'last', model, train_meta)
        if metrics['nll'] < best:
            best = metrics['nll']
            if (out / 'best').exists():
                shutil.rmtree(out / 'best')
            save_splines(out / 'best', model, train_meta)
            print(f'  New best: PPL={metrics["ppl"]:.4f}', flush=True)
        (out / 'metrics.json').write_text(json.dumps(train_meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
