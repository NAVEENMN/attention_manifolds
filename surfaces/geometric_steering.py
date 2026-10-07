"""Local Q/K surface sweeps; compatible with qk-logit-spline-2d-v1 checkpoints."""
import argparse
import json
import math
import random
from pathlib import Path

import torch

from experiment import load_backbone, rollout
from qk_spline_2d import adapters, load_adapter


def patch_stats(adapter, q, k, head, coordinate, a, b):
    """RMS after removing the row-constant softmax-null component."""
    # Normalize before selecting a coordinate (domain buffers span all D).
    from qk_spline_2d import cubic_basis
    u = adapter.coordinates(q, 'q')[:, head, :, coordinate]
    v = adapter.coordinates(k, 'k')[:, head, :, coordinate]
    p = cubic_basis(u, adapter.n_basis)[..., a, None] * cubic_basis(v, adapter.n_basis)[..., b].unsqueeze(-2)
    p = p.double() / math.sqrt(adapter.head_dim)
    valid = torch.ones(p.shape[-2:], device=p.device, dtype=torch.bool).tril()
    centered = (p - (p * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)) * valid
    return {'unit_effective_score_rms': (centered.square().sum() / (valid.sum() * p.shape[0])).sqrt().item(),
            'active_causal_fraction': ((p != 0) & valid).sum().item() / (valid.sum().item() * p.shape[0])}


@torch.no_grad()
def measure(model, ids, layer, head, capture_inputs=False):
    attn = model.model.layers[layer].self_attn
    saved = {}
    def attention_hook(module, inputs, output):
        saved['attention'] = output[1][:, head].detach().cpu().double()
    def spline_hook(module, inputs, output):
        saved['qk'] = tuple(x.detach() for x in inputs)
    hooks = [attn.register_forward_hook(attention_hook)]
    if capture_inputs:
        hooks.append(attn.qk_spline.register_forward_hook(spline_hook))
    try:
        result = model(input_ids=ids, use_cache=False)
        saved['logits'] = result.logits[:, -1].detach().cpu().double()
    finally:
        for hook in hooks:
            hook.remove()
    return saved


def compare(base, edited):
    logp = base['logits'].log_softmax(-1)
    logq = edited['logits'].log_softmax(-1)
    p, q = base['attention'], edited['attention']
    return {'next_token_kl_base_to_edit': (logp.exp() * (logp-logq)).sum(-1).mean().clamp_min(0).item(),
            'next_token_logit_rms_change': (base['logits']-edited['logits']).square().mean().sqrt().item(),
            'attention_mean_row_total_variation': (p-q).abs().sum(-1).mean().item()/2,
            'attention_mean_row_kl_base_to_edit':
                (p * (p.clamp_min(1e-300).log()-q.clamp_min(1e-300).log())).sum(-1).mean().clamp_min(0).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True, help='calibrated.pt best.pt')
    parser.add_argument('--prompts', required=True, help='JSON list of strings; same prompts for all conditions')
    parser.add_argument('--output', required=True, help='New JSONL file (flushed after each condition)')
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--head', type=int, required=True)
    parser.add_argument('--coordinate', type=int, required=True)
    parser.add_argument('--q-index', type=int, default=3)
    parser.add_argument('--k-index', type=int, default=4)
    parser.add_argument('--strengths', type=float, nargs='+', default=[-2, -1, 0, 1, 2])
    parser.add_argument('--mode', choices=['coefficient', 'score-rms'], default='coefficient')
    parser.add_argument('--random-controls', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-abs-delta', type=float, default=100.0)
    parser.add_argument('--min-patch-rms', type=float, default=1e-8)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.random_controls < 0 or args.max_new_tokens < 1 or args.max_abs_delta <= 0 or args.min_patch_rms <= 0:
        parser.error('Invalid limits')
    if not all(math.isfinite(x) for x in args.strengths + [args.max_abs_delta, args.min_patch_rms]):
        parser.error('Strengths and limits must be finite')
    prompts = json.loads(Path(args.prompts).read_text())
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        parser.error('prompts must be a nonempty JSON list of nonempty strings')
    destination = Path(args.output)
    if destination.exists():
        parser.error('Output exists; choose a new filename')
    first = torch.load(args.checkpoints[0], map_location='cpu', weights_only=True)
    if first.get('format') != 'qk-logit-spline-2d-v1':
        parser.error('Incompatible checkpoint format')
    meta = first['metadata']
    revision = meta.get('resolved_commit') or meta.get('revision', 'main')
    model, tokenizer = load_backbone(meta['model'], revision, args.device)
    load_adapter(args.checkpoints[0], model)
    model.eval()
    if args.layer not in adapters(model) or not 0 <= args.head < model.config.num_attention_heads:
        parser.error('Invalid layer or query head')
    adapter = adapters(model)[args.layer]
    with adapter.local_edit(args.head, args.coordinate, args.q_index, args.k_index, 0):
        pass
    candidates = [(a,b) for a in range(adapter.n_basis) for b in range(adapter.n_basis)
                  if (a,b) != (args.q_index,args.k_index)]
    rng = random.Random(args.seed)
    if args.random_controls > len(candidates):
        parser.error('Too many random controls')
    # Prespecified patches: never resample an inactive control to improve results.
    patches = [('target', args.q_index, args.k_index)] + [
        (f'random_{i}', a, b) for i,(a,b) in enumerate(rng.sample(candidates, args.random_controls))]
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else (eos if isinstance(eos, list) else [eos])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as output:
        def emit(row):
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()
        emit({'record': 'config', 'args': vars(args), 'patches': patches,
              'model': meta['model'], 'revision': revision,
              'note': 'Random controls matched to target effective score RMS on each unedited prompt; no semantic labels inferred.'})
        for filename in args.checkpoints:
            ckpt = torch.load(filename, map_location='cpu', weights_only=True)
            other = ckpt['metadata']
            if (ckpt.get('format') != first['format'] or ckpt['layers'] != first['layers'] or
                ckpt['n_basis'] != first['n_basis'] or other['model'] != meta['model'] or
                (other.get('resolved_commit') or other.get('revision','main')) != revision):
                raise ValueError('Checkpoints must use the same backbone, basis count and layers')
            for i, state in ckpt['state'].items():
                adapters(model)[int(i)].load_state_dict(state, strict=True)
            for number, prompt in enumerate(prompts):
                if args.chat:
                    ids = tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                            add_generation_prompt=True, return_tensors='pt')['input_ids'].to(args.device)
                else:
                    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(args.device)
                if ids.shape[1]+args.max_new_tokens > model.config.max_position_embeddings:
                    raise ValueError('Prompt plus generation exceeds model context')
                base = measure(model, ids, args.layer, args.head, True)
                q, k = base.pop('qk')
                statistics = {name: patch_stats(adapter, q, k, args.head, args.coordinate, a, b)
                              for name,a,b in patches}
                del q, k
                base_tokens = rollout(model, ids, args.max_new_tokens, eos)
                emit({'record':'baseline', 'checkpoint':filename, 'prompt_id':number, 'prompt':prompt,
                      'prompt_token_ids':ids[0].tolist(), 'continuation_token_ids':base_tokens,
                      'continuation':tokenizer.decode(base_tokens, skip_special_tokens=True)})
                for name, a, b in patches:
                    unit = statistics[name]['unit_effective_score_rms']
                    target_unit = statistics['target']['unit_effective_score_rms']
                    for strength in args.strengths:
                        row = {'record':'edit', 'checkpoint':filename, 'prompt_id':number,
                               'patch':name, 'q_index':a, 'k_index':b, 'strength':strength,
                               **statistics[name]}
                        target_rms = strength * target_unit if args.mode == 'coefficient' else strength
                        if args.mode == 'coefficient' and name == 'target':
                            delta = strength
                        elif strength == 0:
                            delta = 0.0
                        elif unit <= args.min_patch_rms or (args.mode == 'coefficient' and target_unit <= args.min_patch_rms):
                            emit({**row, 'status':'skipped_inactive_or_softmax_null_patch'})
                            continue
                        else:
                            delta = target_rms / unit
                        if abs(delta) > args.max_abs_delta:
                            emit({**row, 'status':'skipped_delta_cap', 'required_delta':delta})
                            continue
                        with adapter.local_edit(args.head, args.coordinate, a, b, delta):
                            edited = measure(model, ids, args.layer, args.head)
                            tokens = rollout(model, ids, args.max_new_tokens, eos)
                        emit({**row, 'status':'ok', 'coefficient_delta':delta,
                              'effective_score_rms':abs(delta)*unit, **compare(base,edited),
                              'changed_token_sequence':tokens != base_tokens,
                              'continuation_token_ids':tokens,
                              'continuation':tokenizer.decode(tokens, skip_special_tokens=True)})
                restored = measure(model, ids, args.layer, args.head)
                torch.testing.assert_close(restored['logits'], base['logits'], rtol=0, atol=0)
                torch.testing.assert_close(restored['attention'], base['attention'], rtol=0, atol=0)
                emit({'record':'restoration', 'checkpoint':filename, 'prompt_id':number, 'verified':True})
                print(f'{filename}: prompt {number+1}/{len(prompts)} complete', flush=True)
        emit({'record':'complete'})
    print(f'Saved {destination}', flush=True)


if __name__ == '__main__':
    main()
