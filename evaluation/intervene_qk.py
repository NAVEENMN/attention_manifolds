"""One-coefficient interventions in qk-coordinate-curves-v1 checkpoints.

Every delta starts from the same checkpoint. Diagnostics use the same prompt
and baseline continuation; free generation is reported separately.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from spline_qk import SplineProjection, basis_and_derivative, load_splines


def select_curve(model, layer, projection, head, coordinate, coefficient):
    if projection not in ('q', 'k'):
        raise ValueError('Projection must be q or k')
    if not 0 <= layer < len(model.model.layers):
        raise ValueError('Layer index out of range')
    module = getattr(model.model.layers[layer].self_attn, projection + '_proj')
    if not isinstance(module, SplineProjection):
        raise ValueError('Selected projection has no coordinate curves')
    d = getattr(model.config, 'head_dim', None) or model.config.hidden_size // model.config.num_attention_heads
    heads = module.out_features // d
    if not (0 <= head < heads and 0 <= coordinate < d and 0 <= coefficient < module.n_basis):
        raise ValueError(f'Expected head < {heads}, coordinate < {d}, coefficient < {module.n_basis}; indices start at zero')
    return module, head * d + coordinate, d


@contextmanager
def coefficient_edit(module, coordinate, coefficient, amount):
    if not math.isfinite(amount):
        raise ValueError('Edit must be finite')
    saved = module.coefficients[coordinate, coefficient].detach().clone()
    try:
        with torch.no_grad():
            module.coefficients[coordinate, coefficient].copy_(saved + amount)
            if not torch.isfinite(module.coefficients[coordinate, coefficient]):
                raise ValueError('Edit overflowed coefficient dtype')
        yield
    finally:
        with torch.no_grad():
            module.coefficients[coordinate, coefficient].copy_(saved)


@torch.no_grad()
def probe(model, ids, module, coordinate, coefficient):
    captured = []
    def hook(_module, _args, output):
        captured.append(output[..., coordinate].detach().clone())
    handle = module.linear.register_forward_hook(hook)
    try:
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0].float().cpu()
    finally:
        handle.remove()
    z = torch.cat([v.reshape(-1) for v in captured])
    t = .5 + (z - module.center[coordinate]) / (2 * module.radius[coordinate])
    b, db = basis_and_derivative(t.clamp(0, 1), module.n_basis)
    feature = (b + (t - t.clamp(0, 1))[:, None] * db)[:, coefficient]
    return logits, {'raw_coordinate': z.cpu().tolist(),
                    'basis_response': feature.cpu().tolist(),
                    'outside_domain': ((t < 0) | (t > 1)).cpu().tolist()}


def compare_logits(base, edited, ids, prompt_length):
    # Position p-1 predicts the first continuation token at position p.
    a, b = base[prompt_length-1:-1], edited[prompt_length-1:-1]
    target = ids[0, prompt_length:].cpu()
    la, lb = a.double().log_softmax(-1), b.double().log_softmax(-1)
    return {
        'continuation_tokens': len(target),
        'mean_kl_baseline_to_edited': (la.exp() * (la - lb)).sum(-1).mean().item(),
        'next_token_kl': (la[0].exp() * (la[0] - lb[0])).sum().item(),
        'baseline_continuation_nll': F.cross_entropy(a, target).item(),
        'edited_continuation_nll': F.cross_entropy(b, target).item(),
        'argmax_change_fraction': (a.argmax(-1) != b.argmax(-1)).float().mean().item(),
    }


@torch.no_grad()
def generate(model, ids, tokenizer, max_new_tokens):
    return model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                          max_new_tokens=max_new_tokens, use_cache=True,
                          pad_token_id=tokenizer.eos_token_id)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompts', help='JSON list of prompt strings')
    p.add_argument('--chat', action='store_true')
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--projection', choices=['q', 'k'], required=True)
    p.add_argument('--head', type=int, required=True, help='Q head or native KV head, zero-based')
    p.add_argument('--coordinate', type=int, required=True, help='Coordinate within that head, zero-based')
    p.add_argument('--coefficient', type=int, required=True)
    p.add_argument('--deltas', type=float, nargs='+', default=[-.5, -.25, 0., .25, .5])
    p.add_argument('--units', choices=['radius', 'absolute'], default='radius')
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', required=True, help='New JSONL file; refuses to overwrite')
    a = p.parse_args()
    if a.max_new_tokens < 1 or not all(math.isfinite(x) for x in a.deltas):
        p.error('Positive max-new-tokens and finite deltas required')
    texts = [a.prompt] if a.prompt is not None else json.loads(Path(a.prompts).read_text())
    if not isinstance(texts, list) or not texts or not all(isinstance(s, str) and s.strip() for s in texts):
        p.error('Prompts must be a nonempty JSON list of nonempty strings')
    output = Path(a.output)
    if output.exists():
        p.error('Output already exists; choose a new path')
    checkpoint = Path(a.checkpoint)
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    meta = manifest['metadata']
    revision = meta.get('resolved_commit') or meta.get('revision', 'main')
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    model = AutoModelForCausalLM.from_pretrained(meta['model'], revision=revision,
        dtype=torch.float32, attn_implementation='eager').to(a.device)
    load_splines(checkpoint, model)
    model.eval()
    tok = AutoTokenizer.from_pretrained(meta['model'], revision=revision)
    module, index, head_dim = select_curve(model, a.layer, a.projection, a.head, a.coordinate, a.coefficient)
    center, radius = module.center[index].item(), module.radius[index].item()
    low, high = center - radius, center + radius
    # Cardinal basis m has support t in [(m-3)/(M-3), (m+1)/(M-3)].
    support = [max(low, low + 2*radius*(a.coefficient-3)/(module.n_basis-3)),
               min(high, low + 2*radius*(a.coefficient+1)/(module.n_basis-3))]
    groups = model.config.num_attention_heads // model.config.num_key_value_heads
    heads = [a.head] if a.projection == 'q' else list(range(a.head*groups, (a.head+1)*groups))
    checksum = hashlib.sha256()
    for path in [checkpoint/'manifest.json'] + [checkpoint/e['file'] for e in manifest['layers']]:
        checksum.update(path.read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        def write(row):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        write({'type': 'run', 'arguments': vars(a), 'checkpoint_sha256': checksum.hexdigest(),
               'metadata': meta, 'torch_version': torch.__version__, 'head_dim': head_dim,
               'flat_coordinate': index, 'affected_query_heads': heads,
               'domain': [low, high], 'coefficient_support_within_domain': support,
               'original_coefficient': module.coefficients[index, a.coefficient].item(),
               'note': 'Additive coefficient edit; boundary coefficients may affect extrapolation tails. '
                       'All diagnostics use fixed baseline token history; generation uses fresh caches.'})
        for prompt_index, text in enumerate(texts):
            if a.chat:
                ids = tok.apply_chat_template([{'role': 'user', 'content': text}],
                         add_generation_prompt=True, return_tensors='pt')
                if hasattr(ids, 'input_ids'):
                    ids = ids['input_ids']
            else:
                ids = tok(text, return_tensors='pt').input_ids
            ids = ids.to(a.device)
            if not ids.numel() or ids.shape[1] + a.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError('Empty tokenization or requested context exceeds model limit')
            baseline_ids = generate(model, ids, tok, a.max_new_tokens)
            base_logits, base_trace = probe(model, baseline_ids, module, index, a.coefficient)
            # The last continuation token is never processed to predict that continuation.
            base_features = torch.tensor(base_trace['basis_response'][:-1])
            baseline_text = tok.decode(baseline_ids[0, ids.shape[1]:], skip_special_tokens=True)
            write({'type': 'baseline', 'prompt_index': prompt_index, 'prompt': text,
                   'prompt_length': ids.shape[1], 'fixed_token_ids': baseline_ids[0].tolist(),
                   'baseline_response': baseline_text, 'trace': base_trace})
            for delta in dict.fromkeys([0.] + a.deltas):
                amount = delta * radius if a.units == 'radius' else delta
                with coefficient_edit(module, index, a.coefficient, amount):
                    if delta == 0:
                        logits, trace, generated = base_logits, base_trace, baseline_ids
                    else:
                        logits, trace = probe(model, baseline_ids, module, index, a.coefficient)
                        generated = generate(model, ids, tok, a.max_new_tokens)
                effect = amount * base_features
                write({'type': 'intervention', 'prompt_index': prompt_index, 'delta': delta,
                       'coefficient_delta_absolute': amount,
                       'visited_support_fraction': (base_features.abs() > 1e-8).float().mean().item(),
                       'outside_domain_fraction': sum(base_trace['outside_domain'][:-1])/len(base_features),
                       'direct_coordinate_delta_rms': effect.square().mean().sqrt().item(),
                       'direct_coordinate_delta_max_abs': effect.abs().max().item(),
                       'fixed_history_metrics': compare_logits(base_logits, logits, baseline_ids, ids.shape[1]),
                       'edited_trace': trace,
                       'response': tok.decode(generated[0, ids.shape[1]:], skip_special_tokens=True),
                       'generated_token_ids': generated[0, ids.shape[1]:].tolist(),
                       'generation_changed': not torch.equal(generated, baseline_ids)})
                print(f'Prompt {prompt_index}, delta {delta:+g}: saved', flush=True)


if __name__ == '__main__':
    main()
