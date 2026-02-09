# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task Context

This repository is part of an A/B testing experiment (Task ID: TASK_10970) comparing Claude model performance on HRNet-based image classification. The task involves improving HRNet's ImageNet classification performance through topological data analysis (TDA), specifically using **bottleneck distance** to analyze and optimize the model's learned representations.

The reference implementation lives at `../model_b/hrnet_base/`. The model_a workspace starts as an empty clone — work is done here and compared against model_b.

## Commands

All commands below should be run from `hrnet_base/` (or wherever the HRNet implementation is placed):

```bash
# Install dependencies
pip install -r requirements.txt

# Train (must be run from tools/ or with tools/ in path)
python tools/train.py --cfg experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml

# Validate with a pretrained model
python tools/valid.py --cfg experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml \
    --testModel hrnetv2_w18_imagenet_pretrained.pth

# Specify output and log directories
python tools/train.py --cfg experiments/<config>.yaml --modelDir output/ --logDir log/ --dataDir data/
```

ImageNet data must be placed under `data/imagenet/images/` with `train/` and `val/` subdirectories.

## Architecture Overview

### HRNet Model (`lib/models/cls_hrnet.py`)

The core is `HighResolutionNet`, which maintains **multi-resolution feature maps in parallel** throughout the network rather than progressively reducing resolution. Key design:

1. **Stem**: Two stride-2 3×3 convolutions (3→64→64 channels)
2. **Stage 1**: Single-resolution bottleneck layers (64→256 channels)
3. **Stages 2–4**: Multi-resolution branches with `HighResolutionModule` fusing features across resolutions. Each stage adds a new lower-resolution branch:
   - Stage 2: 2 resolutions
   - Stage 3: 3 resolutions
   - Stage 4: 4 resolutions
4. **Classification Head** (`_make_head`): Progressively merges multi-resolution features via bottleneck increase + 2-strided downsampling, ending with global average pooling → 2048-dim → Linear(2048, 1000)

`HighResolutionModule` handles both within-resolution processing (via `_make_branches`) and cross-resolution fusion (via `_make_fuse_layers`): higher→lower via upsampling + 1×1 conv; lower→higher via stride-2 3×3 convs.

### Configuration System (`lib/config/`)

Uses YACS hierarchical config. `default.py` defines all defaults; experiment YAML files override them. Key config paths:
- `MODEL.EXTRA.STAGE{1-4}` — per-stage architecture (NUM_BRANCHES, BLOCK type, NUM_BLOCKS, NUM_CHANNELS, FUSE_METHOD)
- `TRAIN.LR`, `TRAIN.OPTIMIZER`, `TRAIN.LR_STEP` — training hyperparameters
- `DATASET.ROOT` — auto-constructed as `DATA_DIR/DATASET/images`

### Training Loop (`lib/core/function.py`)

Standard PyTorch loop: forward → CrossEntropyLoss → backward → optimizer step. Reports top-1/top-5 accuracy via `AverageMeter`. TensorboardX is used for logging scalars.

### Path Setup

`tools/_init_paths.py` adds `lib/` to `sys.path`, enabling imports like `from config import config` and `import models`.

## Topological Enhancement Goal

The objective is to apply **bottleneck distance** (from Topological Data Analysis) to improve model performance. This involves:

- Computing persistence diagrams from feature map activations at various network layers
- Using bottleneck distance to measure topological stability of representations across training/validation sets
- Guiding training decisions (e.g., hyperparameter selection, regularization, architecture choices) based on topological metrics

The reference for TDA integration is `../model_b/topology_analyzer.py`. Dependencies for TDA include `gudhi`, `ripser`, `persim`, and `scikit-tda`.

## Available Model Variants

Experiments defined in `experiments/` follow the naming pattern `cls_hrnet_w{width}_sgd_lr5e-2_wd1e-4_bs32_x100.yaml`. Width options: W18-Small-v1, W18-Small-v2, W18, W30, W32, W40, W44, W48, W64. Wider networks have more parameters and better accuracy but higher compute cost.
