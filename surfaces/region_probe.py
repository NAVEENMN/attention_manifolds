"""Explain a local Q/K compatibility intervention at token-pair level.

Produces JSON plus a self-contained HTML report. Requires the existing
qk_spline_2d.py and experiment.py; supports their float32/no-cache Llama adapter.
"""
import argparse
import contextlib
import html
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment import load_backbone, rollout
from qk_spline_2d import adapters, cubic_basis, load_adapter


def rms(x):
    return x.double().square().mean().sqrt().item()


@contextlib.contextmanager
def scale_head(attn, head, factor):
    """Scale one head's AV output before o_proj. Always remove the hook."""
    def hook(module, inputs):
        x = inputs[0].clone()
        lo = head * attn.head_dim
        x[..., lo:lo + attn.head_dim] *= factor
        return (x,) + inputs[1:]
    handle = attn.o_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def snapshot(model, ids, layer, head, coordinate):
    attn = model.model.layers[layer].self_attn
    adapter = attn.qk_spline
    saved = {}
    def spline_hook(module, inputs, output):
        q, k = inputs
        saved['u'] = module.coordinates(q, 'q')[0, head, :, coordinate].cpu().double()
        saved['v'] = module.coordinates(k, 'k')[0, head, :, coordinate].cpu().double()
        saved['score_correction'] = output[0, head].cpu().double()
    def attention_hook(module, inputs, output):
        saved['attention'] = output[1][0].cpu().double()
        saved['projected_output'] = output[0][0].cpu().double()
    def projection_hook(module, inputs):
        lo = head * attn.head_dim
        contribution = F.linear(inputs[0][..., lo:lo+attn.head_dim], module.weight[:, lo:lo+attn.head_dim])
        saved['head_contribution'] = contribution[0].cpu().double()
    handles = [adapter.register_forward_hook(spline_hook),
               attn.register_forward_hook(attention_hook),
               attn.o_proj.register_forward_pre_hook(projection_hook)]
    try:
        saved['next_logits'] = model(input_ids=ids, use_cache=False).logits[0, -1].cpu().double()
    finally:
        for handle in handles:
            handle.remove()
    return saved


def unit_patch(base, n_basis, a, b, head_dim):
    return (cubic_basis(base['u'], n_basis)[:, a, None] *
            cubic_basis(base['v'], n_basis)[None, :, b]) / math.sqrt(head_dim)


def pair_rows(base, edited, patch, tokens, head, query, limit):
    before, after = base['attention'][head], edited['attention'][head]
    diff = after-before
    valid = torch.ones_like(diff, dtype=torch.bool).tril()
    def row(i, j):
        return {'query_position':i, 'query_token':tokens[i], 'key_position':j, 'key_token':tokens[j],
                'q_coordinate':base['u'][i].item(), 'k_coordinate':base['v'][j].item(),
                'in_patch_support':bool(patch[i,j] > 0),
                'score_change':(edited['score_correction']-base['score_correction'])[i,j].item(),
                'attention_before':before[i,j].item(), 'attention_after':after[i,j].item(),
                'attention_change':diff[i,j].item()}
    ranked = diff.abs().masked_fill(~valid, -1).flatten().argsort(descending=True)
    top = [row(int(index)//len(tokens), int(index)%len(tokens)) for index in ranked[:min(limit, int(valid.sum()))]]
    selected = [row(query, j) for j in range(query+1)]
    selected.sort(key=lambda r: abs(r['attention_change']), reverse=True)
    return top, selected


def diagnostics(base, edited, patch, delta, head, query):
    before, after = base['attention'][head], edited['attention'][head]
    valid = torch.ones_like(patch, dtype=torch.bool).tril()
    support = (patch > 0) & valid
    actual_scores = edited['score_correction']-base['score_correction']
    expected_scores = delta*patch
    # Scores outside causal support are computed by the spline but masked later.
    centered = (expected_scores-(expected_scores*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True))*valid
    logp, logq = base['next_logits'].log_softmax(-1), edited['next_logits'].log_softmax(-1)
    outside = valid & ~support
    other = [i for i in range(base['attention'].shape[0]) if i != head]
    predicted = (before.clamp_min(1e-300).log()+expected_scores).masked_fill(~valid, float('-inf')).softmax(-1)
    return {
        'causal_pair_count':int(valid.sum()), 'active_pair_count':int(support.sum()),
        'query_coordinate_clipped_fraction':((base['u']==0)|(base['u']==1)).double().mean().item(),
        'key_coordinate_clipped_fraction':((base['v']==0)|(base['v']==1)).double().mean().item(),
        'active_query_rows':int(support.any(-1).sum()),
        'effective_score_rms':(centered.square().sum()/valid.sum()).sqrt().item(),
        'score_formula_max_error':(actual_scores-expected_scores).abs().max().item(),
        'max_score_change_outside_support':actual_scores[outside].abs().max().item() if outside.any() else 0.,
        'attention_formula_max_error':(predicted-after).abs().max().item(),
        'other_heads_same_layer_max_attention_change':
            (base['attention'][other]-edited['attention'][other]).abs().max().item() if other else 0.,
        'mean_row_attention_total_variation':(after-before).abs().sum(-1).mean().item()/2,
        'mean_support_mass_before':(before*support).sum(-1).mean().item(),
        'mean_support_mass_after':(after*support).sum(-1).mean().item(),
        'selected_query_support_mass_before':before[query][support[query]].sum().item(),
        'selected_query_support_mass_after':after[query][support[query]].sum().item(),
        'selected_query_has_active_patch':bool(support[query].any()),
        'outside_support_attention_l1':(after-before)[outside].abs().sum().item(),
        'projected_attention_output_rms_change':rms(edited['projected_output']-base['projected_output']),
        'next_token_kl_base_to_edit':(logp.exp()*(logp-logq)).sum().clamp_min(0).item(),
    }


@torch.no_grad()
def candidate_scores(model, tokenizer, ids, candidates, expected_index):
    rows = []
    for text in candidates:
        suffix = tokenizer(text, add_special_tokens=False)['input_ids']
        if not suffix:
            raise ValueError('Candidate tokenized to an empty sequence')
        if ids.shape[1]+len(suffix) > model.config.max_position_embeddings:
            raise ValueError('Candidate exceeds context window')
        continuation = torch.tensor([suffix], device=ids.device)
        full = torch.cat([ids, continuation], dim=-1)
        logits = model(input_ids=full, use_cache=False).logits[0, ids.shape[1]-1:full.shape[1]-1].double()
        lp = logits.log_softmax(-1).gather(-1, continuation[0,:,None]).squeeze(-1)
        rows.append({'text':text, 'token_ids':suffix, 'sum_log_probability':lp.sum().item(),
                     'mean_token_log_probability':lp.mean().item()})
    result = {'candidates':rows}
    if rows:
        winner = max(range(len(rows)), key=lambda i:rows[i]['sum_log_probability'])
        result['highest_sequence_probability_index'] = winner
        if expected_index is not None:
            result['expected_index'] = expected_index
            result['expected_candidate_ranked_first'] = winner == expected_index
    return result


def render_report(result):
    esc = lambda x: html.escape(str(x))
    parts = ['<!doctype html><meta charset="utf-8"><title>Q/K region probe</title>',
             '<style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#18212b} '
             'table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}'
             'pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}details{margin:16px 0}h2{margin-top:40px}</style>',
             '<h1>What changes in this Q/K region?</h1><p>'+esc(result['prompt'])+'</p>',
             '<p>Locality refers to the direct score edit. Softmax can change weights outside the patch, '
             'and later layers can propagate effects globally. Candidate scores and generated text require '
             'a separate semantic evaluation.</p>']
    def table(rows):
        out = ['<table><tr><th>Query position/token</th><th>Key position/token</th><th>u, v</th><th>In region</th>'
               '<th>Score Δ</th><th>Attention before → after</th><th>Attention Δ</th></tr>']
        for r in rows:
            out.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [
                f"{r['query_position']}: {r['query_token']!r}", f"{r['key_position']}: {r['key_token']!r}",
                f"{r['q_coordinate']:.3f}, {r['k_coordinate']:.3f}", r['in_patch_support'],
                f"{r['score_change']:+.5f}",f"{r['attention_before']:.5f} → {r['attention_after']:.5f}",
                f"{r['attention_change']:+.5f}"] )+'</tr>')
        return ''.join(out)+'</table>'
    for run in result['runs']:
        parts += ['<h2>'+esc(run['checkpoint'])+'</h2><h3>Unedited response</h3><pre>'+esc(run['baseline']['text'])+'</pre>']
        for edit in run['edits']:
            parts += ['<h3>Local coefficient offset '+esc(edit['delta'])+'</h3>',
                      '<pre>'+esc(json.dumps(edit['metrics'],indent=2))+'</pre>',
                      '<h4>Response with local edit</h4><pre>'+esc(edit['response']['text'])+'</pre>',
                      '<h4>Most changed causal token pairs</h4>',table(edit['top_pairs']),
                      '<details><summary>Keys attended to by the selected query position</summary>',
                      table(edit['selected_query_pairs']), '</details>',
                      '<details><summary>Matched head-scaling controls</summary><pre>'+esc(json.dumps(edit['head_scaling_controls'],indent=2))+'</pre></details>']
            if result['args']['candidates']:
                parts += ['<details><summary>Prespecified candidate scores: baseline and local edit</summary><pre>'+esc(json.dumps(
                    {'baseline':run['baseline']['candidate_scores'],'local_edit':edit['response']['candidate_scores']},indent=2))+'</pre></details>']
    return '\n'.join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--chat', action='store_true')
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--head', type=int, required=True)
    p.add_argument('--coordinate', type=int, required=True)
    p.add_argument('--q-index', type=int, default=3)
    p.add_argument('--k-index', type=int, default=4)
    p.add_argument('--deltas', type=float, nargs='+', default=[-2,0,2])
    p.add_argument('--query-position', type=int, default=-1, help='Position in actual tokenized/chat-formatted prompt')
    p.add_argument('--candidates', nargs='+', default=[], help='Prespecified exact assistant continuations')
    p.add_argument('--expected-index', type=int)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--top-pairs', type=int, default=20)
    p.add_argument('--max-abs-head-scale', type=float, default=10.)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', required=True, help='New output directory: report.html and results.json')
    args = p.parse_args()
    if not args.prompt.strip() or args.max_new_tokens < 1 or args.top_pairs < 1:
        p.error('Use a nonempty prompt and positive lengths')
    if not all(math.isfinite(x) for x in args.deltas+[args.max_abs_head_scale]) or args.max_abs_head_scale < 1:
        p.error('Invalid deltas or head-scale cap')
    if args.expected_index is not None and not 0 <= args.expected_index < len(args.candidates):
        p.error('expected-index must index the supplied candidates')
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    first = torch.load(args.checkpoints[0], map_location='cpu', weights_only=True)
    if first.get('format') != 'qk-logit-spline-2d-v1':
        p.error('Unsupported checkpoint format')
    meta = first['metadata']
    revision = meta.get('resolved_commit') or meta.get('revision','main')
    model, tokenizer = load_backbone(meta['model'], revision, args.device)
    load_adapter(args.checkpoints[0], model)
    model.eval()
    if args.layer not in adapters(model) or not 0 <= args.head < model.config.num_attention_heads:
        p.error('Invalid layer or head')
    adapter = adapters(model)[args.layer]
    with adapter.local_edit(args.head,args.coordinate,args.q_index,args.k_index,0):
        pass
    if args.chat:
        ids = tokenizer.apply_chat_template([{'role':'user','content':args.prompt}],
                    add_generation_prompt=True,return_tensors='pt')['input_ids'].to(args.device)
    else:
        ids = tokenizer(args.prompt,return_tensors='pt').input_ids.to(args.device)
    n = ids.shape[1]
    query = args.query_position if args.query_position >= 0 else n+args.query_position
    if not 0 <= query < n or n+args.max_new_tokens > model.config.max_position_embeddings:
        p.error('Query position or context length out of range')
    tokens = [tokenizer.decode([i],skip_special_tokens=False) for i in ids[0].tolist()]
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else (eos if isinstance(eos,list) else [eos])
    def response():
        token_ids = rollout(model,ids,args.max_new_tokens,eos)
        return {'token_ids':token_ids,'text':tokenizer.decode(token_ids,skip_special_tokens=True),
                'candidate_scores':candidate_scores(model,tokenizer,ids,args.candidates,args.expected_index)}
    result = {'args':vars(args),'prompt':args.prompt,'prompt_token_ids':ids[0].tolist(),
              'tokens':tokens,'selected_query_position':query,'runs':[], 'complete':False}
    def save():
        (destination/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        (destination/'report.html').write_text(render_report(result))
    attn = model.model.layers[args.layer].self_attn
    for filename in args.checkpoints:
        checkpoint = torch.load(filename,map_location='cpu',weights_only=True)
        cm = checkpoint['metadata']
        if (checkpoint.get('format')!=first['format'] or checkpoint['n_basis']!=first['n_basis'] or
            checkpoint['layers']!=first['layers'] or cm['model']!=meta['model'] or
            (cm.get('resolved_commit') or cm.get('revision','main'))!=revision):
            raise ValueError('Checkpoints must share backbone revision, layers and basis count')
        for i,state in checkpoint['state'].items():
            adapters(model)[int(i)].load_state_dict(state,strict=True)
        base = snapshot(model,ids,args.layer,args.head,args.coordinate)
        patch = unit_patch(base,adapter.n_basis,args.q_index,args.k_index,adapter.head_dim)
        run = {'checkpoint':filename,'baseline':response(),'edits':[],
               'query_coordinates':base['u'].tolist(),'key_coordinates':base['v'].tolist()}
        result['runs'].append(run)
        for delta in args.deltas:
            with adapter.local_edit(args.head,args.coordinate,args.q_index,args.k_index,delta):
                edited = snapshot(model,ids,args.layer,args.head,args.coordinate)
                generated = response()
            metrics = diagnostics(base,edited,patch,delta,args.head,query)
            top, selected = pair_rows(base,edited,patch,tokens,args.head,query,args.top_pairs)
            edit = {'delta':delta,'metrics':metrics,'response':generated,
                    'changed_token_sequence':generated['token_ids']!=run['baseline']['token_ids'],
                    'top_pairs':top,'selected_query_pairs':selected,'head_scaling_controls':[]}
            target = metrics['projected_attention_output_rms_change']
            denominator = rms(base['head_contribution'])
            for sign in (-1,1):
                if denominator < 1e-12 and target > 0:
                    edit['head_scaling_controls'].append({'status':'skipped_zero_head_contribution'})
                    continue
                factor = 1+sign*target/denominator if denominator >= 1e-12 else 1.
                if abs(factor) > args.max_abs_head_scale:
                    edit['head_scaling_controls'].append({'status':'skipped_scale_cap','required_factor':factor})
                    continue
                with scale_head(attn,args.head,factor):
                    control = snapshot(model,ids,args.layer,args.head,args.coordinate)
                    generated_control = response()
                control_metrics = diagnostics(base,control,patch,0,args.head,query)
                edit['head_scaling_controls'].append({'status':'ok','scale':factor,
                     'target_projected_output_rms':target,'metrics':control_metrics,'response':generated_control})
            run['edits'].append(edit)
            save()
            print(f'{filename}: delta={delta:g}, active_pairs={metrics["active_pair_count"]}, '
                  f'attention_TV={metrics["mean_row_attention_total_variation"]:.6g}',flush=True)
        restored = snapshot(model,ids,args.layer,args.head,args.coordinate)
        for key in ('next_logits','attention','projected_output'):
            torch.testing.assert_close(base[key],restored[key],rtol=0,atol=0)
        run['restoration_verified'] = True
        save()
    result['complete'] = True
    save()
    print(f'Open {destination / "report.html"}',flush=True)


if __name__ == '__main__':
    main()
