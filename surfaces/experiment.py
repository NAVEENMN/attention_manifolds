"""Single-device float32 reference training and paired local interventions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from qk_spline_2d import adapters, calibrate, install, load_adapter, save_adapter


def load_backbone(name, revision, device):
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.float32, attn_implementation='eager')
    model.config.use_cache = False
    return model.to(device).eval(), AutoTokenizer.from_pretrained(name, revision=revision)


def text_blocks(path, tokenizer, block_size, max_blocks):
    raw = Path(path).read_bytes()
    # Plain-text LM objective. Use separately prepared train/validation files.
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


def train(args):
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use an empty output directory to avoid overwriting an experiment')
    model, tokenizer = load_backbone(args.model, args.revision, args.device)
    if args.block_size > model.config.max_position_embeddings:
        raise ValueError('block-size exceeds model context length')
    training, train_hash = text_blocks(args.train_text, tokenizer, args.block_size, args.max_train_blocks)
    validation, val_hash = text_blocks(args.eval_text, tokenizer, args.block_size, args.max_eval_blocks)
    if train_hash == val_hash:
        raise ValueError('Training and validation files must be different')
    baseline = evaluate(model, validation, args.batch_size, args.device)
    probe = validation[:1].to(args.device)
    with torch.no_grad():
        before = model(input_ids=probe, use_cache=False).logits.detach()
    selected = None if args.layers is None else [int(x) for x in args.layers.split(',')]
    install(model, args.n_basis, layers=selected)
    domain_stats = calibrate(model, (b.to(args.device) for b in
        training[:args.calibration_blocks].split(args.batch_size)), args.domain_std)
    with torch.no_grad():
        after = model(input_ids=probe, use_cache=False).logits
    identity_error = (after - before).abs().max().item()
    torch.testing.assert_close(after, before, atol=2e-5, rtol=2e-5)
    del before, after, probe
    metadata = {'model': args.model, 'revision': args.revision,
                'resolved_commit': getattr(model.config, '_commit_hash', None),
                'seed': args.seed, 'train_sha256': train_hash, 'eval_sha256': val_hash,
                'torch': str(torch.__version__), 'transformers': transformers.__version__,
                'args': vars(args), 'domain_stats': domain_stats,
                'baseline': baseline, 'identity_max_logit_error': identity_error,
                'train_blocks': len(training), 'eval_blocks': len(validation)}
    save_adapter(out / 'calibrated.pt', model, metadata)
    params = [a.coefficients for a in adapters(model).values()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(json.dumps({'baseline': baseline, 'identity_max_logit_error': identity_error,
                      'trainable_parameters': sum(p.numel() for p in params)}), flush=True)
    best = float('inf')
    history = []
    # eval() disables dropout but DOES NOT disable gradients. Backbone weights
    # remain frozen; attention surfaces learn through the full model.
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
        metadata['history'] = history
        metadata['epoch'] = epoch + 1
        save_adapter(out / 'last.pt', model, metadata)
        if metrics['nll'] < best:
            best = metrics['nll']
            save_adapter(out / 'best.pt', model, metadata)
        (out / 'metrics.json').write_text(json.dumps(metadata, indent=2) + '\n')


@torch.no_grad()
def rollout(model, ids, n_new, eos):
    result = ids.clone()
    for _ in range(n_new):
        token = model(input_ids=result, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        result = torch.cat([result, token], dim=1)
        if int(token.item()) in eos:
            break
    return result[0, ids.shape[1]:].tolist()


def intervene(args):
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    meta = ckpt['metadata']
    revision = meta.get('resolved_commit') or meta.get('revision', 'main')
    model, tokenizer = load_backbone(meta['model'], revision, args.device)
    load_adapter(args.checkpoint, model)
    model.eval()
    if args.layer not in adapters(model):
        raise ValueError('Selected layer has no trained surface')
    adapter = adapters(model)[args.layer]
    if not 0 <= args.head < model.config.num_attention_heads:
        raise ValueError('Invalid query head')
    if args.chat:
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}],
                add_generation_prompt=True, return_tensors='pt').to(args.device)
    else:
        ids = tokenizer(args.prompt, return_tensors='pt').input_ids.to(args.device)
    if ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError('Prompt plus generation exceeds configured context length')
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else (eos if isinstance(eos, list) else [eos])
    # Compare teacher-forced logits on the SAME prompt, independently of diverging rollouts.
    captures = []
    def capture(module, inputs, output):
        q, k = inputs
        bq, bk = module.basis(q, 'q'), module.basis(k, 'k')
        d, a, b, h = args.coordinate, args.q_index, args.k_index, args.head
        patch = bq[:, h, :, d, a, None] * bk[:, h, None, :, d, b]
        valid = torch.ones(patch.shape[-2:], dtype=torch.bool, device=patch.device).tril()
        # Remove row-constant component: that component cancels in softmax.
        mean = (patch * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)
        effective = (patch - mean) * valid
        captures.append({'active_causal_pair_fraction': ((patch != 0) & valid).sum().item()
                         / (valid.sum().item() * patch.shape[0]),
                         'effective_patch_rms_per_unit_delta':
                         (effective.square().sum() / valid.sum() / patch.shape[0]
                          / module.head_dim).sqrt().item(),
                         'q_clipped_fraction': ((module.coordinates(q, 'q') == 0) |
                                                 (module.coordinates(q, 'q') == 1)).float().mean().item(),
                         'k_clipped_fraction': ((module.coordinates(k, 'k') == 0) |
                                                 (module.coordinates(k, 'k') == 1)).float().mean().item()})
    # Validate edit indices before installing the measurement hook.
    with adapter.local_edit(args.head, args.coordinate, args.q_index, args.k_index, 0):
        pass
    hook = adapter.register_forward_hook(capture)
    try:
        with torch.no_grad():
            baseline = model(input_ids=ids, use_cache=False).logits.float()
    finally:
        hook.remove()
    base_tokens = rollout(model, ids, args.max_new_tokens, eos)
    rows = []
    for delta in args.deltas:
        with adapter.local_edit(args.head, args.coordinate, args.q_index, args.k_index, delta):
            with torch.no_grad():
                edited = model(input_ids=ids, use_cache=False).logits.float()
                # Small KL values suffer cancellation in float32.
                logp, logq = baseline.double().log_softmax(-1), edited.double().log_softmax(-1)
                kl = (logp.exp() * (logp - logq)).sum(-1).clamp_min(0)
            tokens = rollout(model, ids, args.max_new_tokens, eos)
        rows.append({'delta': delta, 'same_prompt_mean_kl_base_to_edit': kl.mean().item(),
                     'next_token_kl_base_to_edit': kl[0, -1].item(),
                     'same_prompt_logit_rms_change': (edited - baseline).square().mean().sqrt().item(),
                     'continuation_token_ids': tokens, 'changed_tokens': tokens != base_tokens,
                     'continuation': tokenizer.decode(tokens, skip_special_tokens=True)})
    with torch.no_grad():
        restored = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
    m = adapter.n_basis
    result = {'checkpoint': str(Path(args.checkpoint).resolve()), 'prompt': args.prompt,
              'prompt_token_ids': ids[0].tolist(), 'chat_template_used': args.chat,
              'layer': args.layer, 'head': args.head, 'coordinate': args.coordinate,
              'q_basis_index': args.q_index, 'k_basis_index': args.k_index,
              'q_support_normalized': [max(0, (args.q_index-3)/(m-3)), min(1, (args.q_index+1)/(m-3))],
              'k_support_normalized': [max(0, (args.k_index-3)/(m-3)), min(1, (args.k_index+1)/(m-3))],
              'patch_measurement': captures[0], 'baseline_continuation_token_ids': base_tokens,
              'baseline_continuation': tokenizer.decode(base_tokens, skip_special_tokens=True),
              'results': rows, 'restoration_verified': True}
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('Output already exists; choose a new filename')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    t = commands.add_parser('train')
    t.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct')
    t.add_argument('--revision', default='main', help='Prefer a pinned model commit')
    t.add_argument('--train-text', required=True)
    t.add_argument('--eval-text', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    t.add_argument('--n-basis', type=int, default=8)
    t.add_argument('--layers', help='Comma-separated layer indices; default all')
    t.add_argument('--block-size', type=int, default=256)
    t.add_argument('--batch-size', type=int, default=1)
    t.add_argument('--grad-accum', type=int, default=8)
    t.add_argument('--epochs', type=int, default=3)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--weight-decay', type=float, default=0.0)
    t.add_argument('--clip-grad', type=float, default=1.0)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--calibration-blocks', type=int, default=32)
    t.add_argument('--domain-std', type=float, default=3.0)
    t.add_argument('--max-train-blocks', type=int, default=4096)
    t.add_argument('--max-eval-blocks', type=int, default=128)
    e = commands.add_parser('intervene')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--prompt', required=True)
    e.add_argument('--chat', action='store_true')
    e.add_argument('--layer', type=int, required=True)
    e.add_argument('--head', type=int, required=True)
    e.add_argument('--coordinate', type=int, required=True)
    e.add_argument('--q-index', type=int, required=True)
    e.add_argument('--k-index', type=int, required=True)
    e.add_argument('--deltas', type=float, nargs='+', default=[-1, 0, 1])
    e.add_argument('--max-new-tokens', type=int, default=64)
    e.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    e.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        for name in ('batch_size', 'grad_accum', 'epochs', 'calibration_blocks', 'max_train_blocks', 'max_eval_blocks'):
            if getattr(args, name) < 1:
                parser.error(name + ' must be positive')
        if args.block_size < 2 or args.lr <= 0 or args.clip_grad <= 0:
            parser.error('Invalid block size, learning rate, or gradient clipping')
        train(args)
    else:
        if args.max_new_tokens < 1:
            parser.error('max-new-tokens must be positive')
        intervene(args)


if __name__ == '__main__':
    main()
