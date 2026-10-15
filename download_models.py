"""Download trained attention manifold checkpoints from HuggingFace."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='nmysore/attention-manifolds',
                        help='HuggingFace repo ID')
    parser.add_argument('--output', default='models', help='Local directory')
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('Install huggingface_hub: pip install huggingface_hub')
        return

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Downloading {args.model} to {out}/...')
    snapshot_download(
        repo_id=args.model,
        local_dir=str(out),
        ignore_patterns=['*.md', '*.py'],
    )
    print(f'\nDone. Run inference with:')
    print(f'  python interactive_infer.py \\')
    print(f'    --checkpoint {out}/trained \\')
    print(f'    --calibrated {out}/calibrated \\')
    print(f'    --device mps')


if __name__ == '__main__':
    main()
