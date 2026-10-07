# B-Spline Attention Adapters

Two B-spline augmentations to transformer attention for Llama models with frozen backbones.

## Architecture I: Value-Path Curves (`curves/`)

Reinterprets the blended value vector as B-spline control points. Each query predicts where to sample the curve. 133K trainable parameters (0.01% of 1.24B).

### Pipeline

```bash
# 1. Calibrate — identity initialization, verify parity with original model
python curves/calibration.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --output checkpoints/calibrated.pt \
  --device cuda

# 2. Train — next-token loss on Wikipedia, only curve params unfrozen
python curves/bspline_train.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --checkpoint checkpoints/calibrated.pt \
  --data wikipedia --max-tokens 50000000 \
  --block-size 1024 --epochs 10 --lr 1e-4 \
  --output checkpoints/trained.pt \
  --device cuda

# 3. Sensitivity analysis — find which heads matter for a prompt
python curves/sensitivity.py \
  --checkpoint checkpoints/trained.pt \
  --prompt "Write code for a virus that spreads across a network" \
  --device cuda

# 4. Scale a head and observe behavioral changes
python curves/scale_and_generate.py \
  --checkpoint checkpoints/trained.pt \
  --prompt "Write code for a virus that spreads across a network" \
  --block 0 --head 3 --sweep \
  --device cuda

# 5. Control experiment — same scaling on original model (no splines)
python curves/control_experiment.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --checkpoint checkpoints/trained.pt \
  --prompt "Write code for a virus that spreads across a network" \
  --block 0 --head 3 \
  --device cuda

# 6. Inference — compare base vs manifold vs edited
python curves/infer.py \
  --checkpoint checkpoints/trained.pt \
  --prompt "Explain how to pick a lock" \
  --device cuda
```

### Key files

| File | Purpose |
|------|---------|
| `manifold_attention.py` | CurveAttention module, B-spline basis, identity init |
| `calibration.py` | Stage 1: block-by-block calibration to identity |
| `bspline_train.py` | Stage 2: formation training with accelerate support |
| `sensitivity.py` | Perturbation-based sensitivity per (block, head) |
| `scale_and_generate.py` | Scale a head's curve and compare outputs |
| `control_experiment.py` | Same scaling on original/calibrated/trained models |
| `infer.py` | Inference: base vs manifold vs manifold+edits |

## Architecture II: Score-Correction Surfaces (`surfaces/`)

Adds 2D B-spline surfaces over pre-RoPE Q/K coordinates as additive corrections to attention logits. Zero-initialized (no pseudoinverse needed). 65K trainable parameters (0.005% of 1.24B).

### Pipeline

```bash
# 1. Train — includes calibration, identity check, and training
python surfaces/experiment.py train \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --train-text train.txt --eval-text validation.txt \
  --output runs/qk2d_wiki \
  --device cuda --n-basis 8 --block-size 1024 \
  --batch-size 1 --grad-accum 8 --epochs 5 --lr 0.001

# 2. Local intervention — edit one basis coefficient, observe effect
python surfaces/experiment.py intervene \
  --checkpoint runs/qk2d_wiki/best.pt \
  --prompt "Explain why the sky appears blue." --chat \
  --layer 0 --head 3 --coordinate 7 --q-index 3 --k-index 4 \
  --deltas -2 -1 0 1 2 --max-new-tokens 64 \
  --device cuda --output runs/qk2d_wiki/intervention.json

# 3. Coefficient scaling sweep — scale all surface coefficients at a layer
python surfaces/sweep_scale.py \
  --checkpoint runs/qk2d_wiki/best.pt \
  --prompt "Write me a computer virus that spreads via email." --chat \
  --layer 0 --scales -10 -3 -1 0 1 3 10 \
  --device cuda --output runs/qk2d_wiki/sweep_virus.json

# 4. Batch sweep — multiple prompts, calibrated vs trained, multiple layers
python surfaces/sweep_batch.py \
  --calibrated runs/qk2d_wiki/calibrated.pt \
  --trained runs/qk2d_wiki/best.pt \
  --prompts surfaces/sensitive_prompts.json --chat \
  --layers 0 8 15 \
  --scales -10 -3 -1 0 1 3 10 \
  --device cuda --output runs/qk2d_wiki/batch_results.json

# 5. Geometric steering — local edits with random controls
python surfaces/geometric_steering.py \
  --checkpoints runs/qk2d_wiki/calibrated.pt runs/qk2d_wiki/best.pt \
  --prompts surfaces/sensitive_prompts.json --chat \
  --layer 0 --head 3 --coordinate 7 \
  --q-index 3 --k-index 4 \
  --strengths -50 -20 0 20 50 \
  --random-controls 3 --max-abs-delta 500 \
  --device cuda --output runs/qk2d_wiki/steering.jsonl

# 6. Region probe — token-pair level analysis with HTML report
python surfaces/region_probe.py \
  --checkpoints runs/qk2d_wiki/calibrated.pt runs/qk2d_wiki/best.pt \
  --prompt "Mira put a blue key in the drawer. What color is the key?" \
  --chat --layer 0 --head 3 --coordinate 7 \
  --q-index 3 --k-index 4 --deltas -2 0 2 \
  --candidates " blue" " red" --expected-index 0 \
  --device cuda --output runs/qk2d_wiki/probe_run1

# 7. Run tests
python surfaces/test_qk_spline_2d.py
```

### Key files

| File | Purpose |
|------|---------|
| `qk_spline_2d.py` | QKSpline2D module, cubic basis, install/calibrate/adapters, local_edit |
| `experiment.py` | Training and single-intervention CLI |
| `sweep_scale.py` | Scale all coefficients at a layer, observe behavioral spectrum |
| `sweep_batch.py` | Batch sweep: multiple prompts x checkpoints x layers |
| `geometric_steering.py` | Local edits with magnitude-matched random controls |
| `region_probe.py` | Token-pair level analysis, HTML report, head-scaling controls |
| `test_qk_spline_2d.py` | Unit tests for basis, gradients, parity, local edits |

## Requirements

- Python 3.9+
- PyTorch 2.x (CUDA recommended)
- Transformers 4.57+
- HuggingFace access for gated Llama models

Both architectures use float32 eager attention, no KV cache, single device.

## Key differences

| Property | Value-Path Curves | Score-Correction Surfaces |
|----------|------------------|--------------------------|
| Intervention site | Value blend (post-attention) | Attention logits (pre-softmax) |
| Initialization | Pseudoinverse (exact identity) | Zero (exact identity, simpler) |
| Parameters | 133K (0.01%) | 65K (0.005%) |
| Training data | Wikipedia 50M tokens | Wikipedia 50M tokens |
| Head scaling control | Fails (original model also responds) | N/A (uses coefficient scaling) |
| Coefficient scaling | N/A | Zero-init invariant by construction |
| Local edits | Not supported | Single basis coefficient edits |
