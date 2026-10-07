"""Scale all surface coefficients at a layer and observe behavioral changes."""
import argparse, json, torch
from experiment import load_backbone, rollout
from qk_spline_2d import adapters, load_adapter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--scales', type=float, nargs='+',
                        default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, -1.0, -3.0, -10.0])
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    meta = ckpt['metadata']
    revision = meta.get('resolved_commit') or meta.get('revision', 'main')
    model, tokenizer = load_backbone(meta['model'], revision, args.device)
    load_adapter(args.checkpoint, model)
    model.eval()

    if args.layer not in adapters(model):
        raise ValueError('Layer has no adapter')
    adapter = adapters(model)[args.layer]

    if args.chat:
        ids = tokenizer.apply_chat_template([{'role':'user','content':args.prompt}],
            add_generation_prompt=True, return_tensors='pt')['input_ids'].to(args.device)
    else:
        ids = tokenizer(args.prompt, return_tensors='pt').input_ids.to(args.device)

    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else (eos if isinstance(eos, list) else [eos])

    orig = adapter.coefficients.data.clone()
    results = []

    for scale in sorted(args.scales):
        adapter.coefficients.data.copy_(orig * scale)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        tokens = rollout(model, ids, args.max_new_tokens, eos)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        results.append({'scale': scale, 'continuation': text, 'token_ids': tokens})
        label = 'BASELINE' if scale == 1.0 else 'scale=' + str(scale)
        print('-' * 70)
        print('  ' + label)
        print('-' * 70)
        print('  ' + text[:200])
        print()

    adapter.coefficients.data.copy_(orig)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'checkpoint': args.checkpoint, 'prompt': args.prompt,
                       'layer': args.layer, 'results': results}, f, indent=2)
        print('Saved ' + args.output)

if __name__ == '__main__':
    main()
