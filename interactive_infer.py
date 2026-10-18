"""Attention Manifolds — Interactive Inference"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spline_qk import load_splines, SplineProjection


# ── Helpers ──────────────────────────────────────────────────────────────

def get_all_coefficients(model):
    coeffs = {}
    for i, layer in enumerate(model.model.layers):
        for kind in ('q', 'k'):
            proj = getattr(layer.self_attn, kind + '_proj')
            if isinstance(proj, SplineProjection):
                coeffs[(i, kind)] = proj.coefficients.data
    return coeffs


def classify(text):
    t = text.lower().replace('\u2019', "'")
    for p in ["i can't", "i cannot", "i'm not able", "i won't",
              "can't assist", "can't provide", "can't fulfill",
              "can't help", "i must decline", "i'm unable"]:
        if p in t:
            return "REFUSE"
    return "COMPLY"


def clear_memory(device):
    if 'cuda' in str(device):
        torch.cuda.empty_cache()
    elif 'mps' in str(device):
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    import gc
    gc.collect()


def generate(model, tokenizer, ids, max_tokens, device):
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_tokens, do_sample=False,
            use_cache=True, pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    del out
    clear_memory(device)
    return text


def encode_prompt(tokenizer, prompt, device):
    ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        add_generation_prompt=True, return_tensors='pt'
    )
    if hasattr(ids, 'input_ids'):
        ids = ids['input_ids']
    return ids.to(device)


# ── Formatting ───────────────────────────────────────────────────────────

def divider(label=''):
    w = 72
    if label:
        pad = w - len(label) - 4
        left = pad // 2
        right = pad - left
        return f"{'─' * left}  {label}  {'─' * right}"
    return '─' * w


def print_response(text, label, model_name, tag=''):
    header = model_name
    if tag:
        header = f"{tag} | {model_name}"
    print(f"\n{divider(header)}")
    print(f"  [{label}]")
    print()
    for line in text.split('\n'):
        print(f"  {line}")
    print(f"\n{divider()}")


# ── Intervention State ───────────────────────────────────────────────────

class InterventionState:
    def __init__(self, model, calibrated_coeffs):
        self.model = model
        self.calibrated = calibrated_coeffs
        self.trained = {k: v.clone() for k, v in get_all_coefficients(model).items()}
        self.n_layers = len(model.model.layers)
        self.mode = 'off'
        self.layers = []
        self.alpha = -1.0
        self.compare = False

    def apply(self):
        if self.mode == 'off' or self.calibrated is None:
            self._restore_all()
            return
        self._restore_all()
        layers = list(range(self.n_layers)) if self.mode == 'negate-all' else self.layers
        coeffs = get_all_coefficients(self.model)
        for (i, kind), coeff in coeffs.items():
            if i not in layers:
                continue
            cal = self.calibrated.get((i, kind))
            if cal is None:
                continue
            if self.mode in ('negate-all', 'negate'):
                coeff.copy_(2 * cal - self.trained[(i, kind)])
            elif self.mode == 'scale':
                delta = self.trained[(i, kind)] - cal
                coeff.copy_(cal + self.alpha * delta)
            elif self.mode == 'revert':
                coeff.copy_(cal)

    def _restore_all(self):
        coeffs = get_all_coefficients(self.model)
        for k, v in self.trained.items():
            coeffs[k].copy_(v)

    def status(self):
        if self.mode == 'off':
            return ''
        if self.mode == 'negate-all':
            return 'negate-all'
        layer_str = ','.join(str(l) for l in self.layers)
        if self.mode == 'scale':
            return f'scale {self.alpha:+.1f} L[{layer_str}]'
        return f'{self.mode} L[{layer_str}]'


# ── Analysis Commands ────────────────────────────────────────────────────

def run_sensitivity(model, tokenizer, state, prompt, device, max_tokens):
    if state.calibrated is None:
        print('  Needs --calibrated')
        return

    ids = encode_prompt(tokenizer, prompt, device)
    state._restore_all()
    base_text = generate(model, tokenizer, ids, max_tokens, device)
    base_label = classify(base_text)

    print(f"\n{divider('sensitivity')}")
    print(f"  Baseline: {base_label}")
    print()
    print(f"  {'Layer':>5}  {'Result':>7}  Response")
    print(f"  {'─'*60}")

    coeffs = get_all_coefficients(model)
    for layer in range(state.n_layers):
        for kind in ('q', 'k'):
            key = (layer, kind)
            if key in state.trained and key in state.calibrated:
                coeffs[key].copy_(2 * state.calibrated[key] - state.trained[key])
        text = generate(model, tokenizer, ids, max_tokens, device)
        label = classify(text)
        for kind in ('q', 'k'):
            key = (layer, kind)
            if key in state.trained:
                coeffs[key].copy_(state.trained[key])
        mark = ' ◀' if label != base_label else ''
        short = text[:55].replace('\n', ' ')
        print(f"  {layer:5d}  {label:>7}  {short}{mark}")

    print(f"{divider()}\n")


def run_sweep(model, tokenizer, state, prompt, device, max_tokens, layers=None):
    if state.calibrated is None:
        print('  Needs --calibrated')
        return

    ids = encode_prompt(tokenizer, prompt, device)
    alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    coeffs = get_all_coefficients(model)

    # Phase 1: all-layer sweep
    print(f"\n{divider('alpha sweep (all layers)')}")
    print(f"  {'Alpha':>6}  {'Result':>7}  Response")
    print(f"  {'─'*60}")

    baseline_label = None
    for alpha in alphas:
        state._restore_all()
        for (i, kind), coeff in coeffs.items():
            cal = state.calibrated.get((i, kind))
            if cal is None:
                continue
            coeff.copy_(cal + alpha * (state.trained[(i, kind)] - cal))
        text = generate(model, tokenizer, ids, max_tokens, device)
        label = classify(text)
        if alpha == 1.0:
            baseline_label = label
        mark = ' ◀ baseline' if alpha == 1.0 else (' ★ FLIP' if baseline_label and label != baseline_label else '')
        short = text[:50].replace('\n', ' ')
        print(f"  {alpha:+5.1f}  {label:>7}  {short}{mark}")
    state._restore_all()

    # Phase 2: per-layer
    target_layers = layers if layers else list(range(state.n_layers))
    print(f"\n{divider('per-layer (alpha=-1)')}")
    print(f"  {'Layer':>5}  {'Result':>7}")
    print(f"  {'─'*25}")

    sensitive = []
    for layer in target_layers:
        state._restore_all()
        for kind in ('q', 'k'):
            key = (layer, kind)
            if key in state.trained and key in state.calibrated:
                coeffs[key].copy_(2 * state.calibrated[key] - state.trained[key])
        text = generate(model, tokenizer, ids, max_tokens, device)
        label = classify(text)
        flipped = label != baseline_label
        if flipped:
            sensitive.append(layer)
        for kind in ('q', 'k'):
            key = (layer, kind)
            if key in state.trained:
                coeffs[key].copy_(state.trained[key])
        mark = ' ◀' if flipped else ''
        print(f"  {layer:5d}  {label:>7}{mark}")

    # Phase 3
    if sensitive:
        print(f"\n{divider(f'sweep on layers {sensitive}')}")
        print(f"  {'Alpha':>6}  {'Result':>7}  Response")
        print(f"  {'─'*60}")
        for alpha in alphas:
            state._restore_all()
            for (i, kind), coeff in coeffs.items():
                if i not in sensitive:
                    continue
                cal = state.calibrated.get((i, kind))
                if cal is None:
                    continue
                coeff.copy_(cal + alpha * (state.trained[(i, kind)] - cal))
            text = generate(model, tokenizer, ids, max_tokens, device)
            label = classify(text)
            mark = ' ◀ baseline' if alpha == 1.0 else (' ★ FLIP' if label != baseline_label else '')
            short = text[:50].replace('\n', ' ')
            print(f"  {alpha:+5.1f}  {label:>7}  {short}{mark}")
        state._restore_all()
        layers_str = ' '.join(str(l) for l in sensitive)
        print(f"\n  Try: /scale <alpha> {layers_str}")
    else:
        # Combination search
        print(f"\n{divider('combination search')}")
        print(f"  {'Layers':>20}  {'Result':>7}")
        print(f"  {'─'*35}")
        combo_sizes = sorted(set(c for c in [2, 4, 8, len(target_layers)//2, len(target_layers)] if 2 <= c <= len(target_layers)))
        found = None
        for size in combo_sizes:
            step = max(1, len(target_layers) // size)
            combo = target_layers[::step][:size]
            state._restore_all()
            for layer in combo:
                for kind in ('q', 'k'):
                    key = (layer, kind)
                    if key in state.trained and key in state.calibrated:
                        coeffs[key].copy_(2 * state.calibrated[key] - state.trained[key])
            text = generate(model, tokenizer, ids, max_tokens, device)
            label = classify(text)
            flipped = label != baseline_label
            mark = ' ★ FLIP' if flipped else ''
            print(f"  {str(combo):>20}  {label:>7}{mark}")
            if flipped and found is None:
                found = combo
            state._restore_all()
        if found:
            print(f"\n  Try: /negate {' '.join(str(l) for l in found)}")
        else:
            print(f"\n  Try: /scale -1.0")

    print(f"{divider()}\n")


# ── Main ─────────────────────────────────────────────────────────────────

HELP = """
  /sweep <prompt>              Full analysis: alpha + layers + combos
  /sensitivity <prompt>        Per-layer sensitivity
  /sweep-layers 3 6 <prompt>   Alpha sweep on specific layers
  /negate-all                  Negate all layers
  /negate 0 3 6                Negate specific layers
  /scale -1.0                  Scale all layers
  /scale -1.0 0 3 6            Scale specific layers
  /revert 0 3                  Revert layers to identity
  /reset                       Reset all interventions
  /compare                     Toggle compare mode
  /tokens N                    Set max tokens
  /help                        Show this help
  /quit                        Exit
"""


def main():
    parser = argparse.ArgumentParser(description='Attention Manifolds — Interactive Inference')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--calibrated', default=None)
    parser.add_argument('--max-tokens', type=int, default=200)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
    args = parser.parse_args()

    # Suppress transformers warnings
    import logging
    logging.getLogger('transformers').setLevel(logging.ERROR)

    manifest = json.loads((Path(args.checkpoint) / 'manifest.json').read_text())
    meta = manifest['metadata']
    model_short = meta['model'].split('/')[-1]
    n_layers = None

    print(f"\n  Loading {model_short}...", end='', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        meta['model'], dtype=torch.float32, attn_implementation='eager'
    ).to(args.device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(meta['model'])
    load_splines(args.checkpoint, model)
    n_layers = len(model.model.layers)
    print(f" done ({n_layers} layers)")

    calibrated_coeffs = None
    if args.calibrated:
        print(f"  Loading calibrated...", end='', flush=True)
        cal_model = AutoModelForCausalLM.from_pretrained(
            meta['model'], dtype=torch.float32, attn_implementation='eager'
        ).to(args.device).eval()
        load_splines(args.calibrated, cal_model)
        calibrated_coeffs = {k: v.clone() for k, v in get_all_coefficients(cal_model).items()}
        del cal_model
        clear_memory(args.device)
        print(f" done")

    state = InterventionState(model, calibrated_coeffs)
    max_tokens = args.max_tokens

    print(f"\n  Ready. Type /help for commands.\n")

    while True:
        try:
            status = state.status()
            prompt_str = f"({status}) " if status else ''
            line = input(f"  {prompt_str}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith('/'):
            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ('/quit', '/exit', '/q'):
                break
            elif cmd == '/help':
                print(HELP)
            elif cmd in ('/reset', '/off'):
                state.mode = 'off'
                state._restore_all()
            elif cmd == '/negate-all':
                state.mode = 'negate-all'
            elif cmd == '/negate':
                state.mode = 'negate'
                state.layers = [int(x) for x in parts[1:]] if len(parts) > 1 else list(range(state.n_layers))
            elif cmd == '/scale':
                state.mode = 'scale'
                if len(parts) > 1:
                    state.alpha = float(parts[1])
                if len(parts) > 2:
                    state.layers = [int(x) for x in parts[2:]]
                else:
                    state.layers = list(range(state.n_layers))
            elif cmd == '/revert':
                state.mode = 'revert'
                state.layers = [int(x) for x in parts[1:]] if len(parts) > 1 else list(range(state.n_layers))
            elif cmd == '/compare':
                state.compare = not state.compare
                print(f"  Compare: {'on' if state.compare else 'off'}")
            elif cmd == '/tokens':
                if len(parts) > 1:
                    max_tokens = int(parts[1])
                    print(f"  Max tokens: {max_tokens}")
            elif cmd == '/sensitivity':
                prompt = ' '.join(parts[1:])
                if prompt:
                    run_sensitivity(model, tokenizer, state, prompt, args.device, max_tokens)
                else:
                    print("  Usage: /sensitivity <prompt>")
            elif cmd == '/sweep':
                prompt = ' '.join(parts[1:])
                if prompt:
                    run_sweep(model, tokenizer, state, prompt, args.device, max_tokens)
                else:
                    print("  Usage: /sweep <prompt>")
            elif cmd == '/sweep-layers':
                layer_nums = []
                prompt_parts = []
                parsing = True
                for p in parts[1:]:
                    if parsing and p.isdigit():
                        layer_nums.append(int(p))
                    else:
                        parsing = False
                        prompt_parts.append(p)
                prompt = ' '.join(prompt_parts)
                if prompt and layer_nums:
                    run_sweep(model, tokenizer, state, prompt, args.device, max_tokens, layers=layer_nums)
                else:
                    print("  Usage: /sweep-layers 3 6 <prompt>")
            else:
                print(f"  Unknown command. Type /help")
            continue

        # Generate
        ids = encode_prompt(tokenizer, line, args.device)

        if state.compare and state.mode != 'off':
            state._restore_all()
            base_text = generate(model, tokenizer, ids, max_tokens, args.device)
            print_response(base_text, classify(base_text), model_short, 'baseline')

            state.apply()
            edit_text = generate(model, tokenizer, ids, max_tokens, args.device)
            state._restore_all()
            print_response(edit_text, classify(edit_text), model_short, state.status())
        else:
            if state.mode != 'off':
                state.apply()
            text = generate(model, tokenizer, ids, max_tokens, args.device)
            if state.mode != 'off':
                state._restore_all()
            tag = state.status() if state.mode != 'off' else ''
            print_response(text, classify(text), model_short, tag)


if __name__ == '__main__':
    main()
