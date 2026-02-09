# Getting Started with HRNet Topological Optimization

This guide will walk you through setting up and running your first topological optimization experiment.

## Table of Contents
1. [Installation](#installation)
2. [Verification](#verification)
3. [Quick Start](#quick-start)
4. [Understanding the Output](#understanding-the-output)
5. [Next Steps](#next-steps)

## Installation

### Option 1: Automated Setup (Recommended)

```bash
cd model_b
./setup.sh
source venv/bin/activate
```

### Option 2: Manual Setup

See [INSTALL.md](INSTALL.md) for detailed instructions.

### Option 3: Quick Test First

Want to see if it works before full installation?

```bash
# Install minimal dependencies
pip install torch torchvision numpy ripser persim matplotlib

# Run basic test
python test_installation.py
```

## Verification

### Step 1: Check Environment

```bash
python check_environment.py
```

Expected output:
```
✓ Python 3.x.x
✓ torch 2.x.x
✓ ripser 0.6.x
✓ persim 0.3.x
✓ All required packages are installed!
```

### Step 2: Run Comprehensive Test

```bash
python test_installation.py
```

This will test:
- Library imports
- Model creation
- Data loading
- Topological analysis
- Loss computation

Expected: "✓ ALL TESTS PASSED!"

### Step 3: (Optional) Run Jupyter Test

```bash
jupyter notebook test_topology_optimization.ipynb
```

Execute all cells (Cell → Run All) to see interactive tests.

## Quick Start

### Experiment 1: Baseline (No Topology)

Train without topological optimization to establish baseline:

```bash
python train_enhanced.py \
    --dataset cifar10 \
    --batch-size 128 \
    --epochs 50 \
    --lr 0.1 \
    --width 18 \
    --topology-weight 0.0 \
    --output-dir ./output/baseline
```

**What to expect:**
- Training progress bar
- Accuracy around 85-90% after 50 epochs
- Output files in `./output/baseline/`

**Time:** ~2 hours on CPU, ~30 minutes on GPU

### Experiment 2: With Topology Optimization

Train with topological optimization:

```bash
python train_enhanced.py \
    --dataset cifar10 \
    --batch-size 128 \
    --epochs 50 \
    --lr 0.1 \
    --width 18 \
    --topology-weight 0.01 \
    --topology-interval 5 \
    --output-dir ./output/topology_opt
```

**What to expect:**
- Training progress with topological loss
- Accuracy around 87-92% after 50 epochs (slightly better)
- More stable training (lower loss variance)
- Output files in `./output/topology_opt/`

**Time:** ~2.5 hours on CPU, ~35 minutes on GPU (slightly longer due to topology computation)

### Experiment 3: Analyze Results

Compare the two models:

```bash
# Analyze baseline
python analyze_topology.py \
    --checkpoint ./output/baseline/checkpoint_best.pth \
    --dataset cifar10 \
    --output-dir ./analysis/baseline \
    --analyze-layers

# Analyze optimized
python analyze_topology.py \
    --checkpoint ./output/topology_opt/checkpoint_best.pth \
    --dataset cifar10 \
    --output-dir ./analysis/topology_opt \
    --analyze-layers
```

**Time:** ~10-15 minutes per model

## Understanding the Output

### Training Outputs

```
output/
├── baseline/
│   ├── checkpoint_best.pth      # Best model weights
│   ├── checkpoint_latest.pth    # Latest model weights
│   └── tensorboard/             # Training logs
└── topology_opt/
    ├── checkpoint_best.pth
    ├── checkpoint_latest.pth
    └── tensorboard/
```

### Analysis Outputs

```
analysis/
├── baseline/
│   ├── analysis_report.txt           # Comprehensive statistics
│   ├── distance_matrix.png           # Inter-class separation
│   ├── topology_evolution.png        # Layer-wise topology
│   └── persistence_diagrams/         # Per-class diagrams
└── topology_opt/
    ├── analysis_report.txt
    ├── distance_matrix.png
    ├── topology_evolution.png
    └── persistence_diagrams/
```

### Key Files to Check

**1. analysis_report.txt**
```
Contains:
- Per-class Betti numbers
- Persistence entropy
- Inter-class bottleneck distances
- Interpretation guide
```

**What to look for:**
- Higher inter-class distances = better separation
- Moderate Betti numbers = good complexity
- Higher persistence entropy = richer topology

**2. distance_matrix.png**

Heatmap showing bottleneck distances between classes.

**What to look for:**
- Bright colors (high values) = classes are topologically distinct
- Dark colors (low values) = classes are similar
- Compare baseline vs optimized: optimized should have brighter colors

**3. topology_evolution.png**

Shows how topology changes through network layers.

**What to look for:**
- Smooth decrease in Betti-0 = gradual clustering
- Moderate Betti-1 = meaningful structure
- Stabilization in later layers = converged features

### Viewing with TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir ./output/
```

Then open http://localhost:6006 in your browser.

**Key metrics:**
- `train/epoch_loss` - Training loss (should decrease)
- `train/epoch_acc` - Training accuracy (should increase)
- `val/acc` - Validation accuracy (most important)
- `topology/persistence_entropy` - Topological complexity
- `topology/betti_0` - Number of clusters

## Common Scenarios

### Scenario 1: Just Testing (5 minutes)

```bash
# Quick test with 1 epoch
python train_enhanced.py \
    --dataset cifar10 \
    --epochs 1 \
    --batch-size 64 \
    --topology-weight 0.01 \
    --output-dir ./output/quick_test
```

### Scenario 2: Full Comparison (4-5 hours total)

```bash
# Run automated comparison
./quick_start.sh
```

This script will:
1. Train baseline model (50 epochs)
2. Analyze baseline topology
3. Train optimized model (50 epochs)
4. Analyze optimized topology
5. Generate comparison report

### Scenario 3: Production Training (16+ hours)

```bash
# Train for 200 epochs with larger model
python train_enhanced.py \
    --dataset cifar10 \
    --batch-size 128 \
    --epochs 200 \
    --lr 0.1 \
    --width 32 \
    --topology-weight 0.01 \
    --topology-interval 10 \
    --output-dir ./output/production
```

### Scenario 4: CIFAR-100 (More Classes)

```bash
python train_enhanced.py \
    --dataset cifar100 \
    --batch-size 128 \
    --epochs 200 \
    --width 32 \
    --topology-weight 0.01 \
    --output-dir ./output/cifar100
```

## Interpreting Results

### Good Results Indicators

✓ **Accuracy improvement:** 1-3% better than baseline
✓ **Higher inter-class distances:** +30-50% in distance matrix
✓ **Lower Betti variance:** More stable topology
✓ **Moderate entropy:** 2.5-3.5 range
✓ **Smooth training:** Less oscillation in loss

### Warning Signs

⚠ **Collapsing topology:** Betti-0 → 1 (all features merging)
⚠ **Unstable training:** High variance in bottleneck distances
⚠ **Very low separation:** Inter-class distances < 0.2
⚠ **Extreme entropy:** < 1.0 or > 5.0

### Troubleshooting

**Problem: Topology loss is very high**
- Solution: Reduce `topology-weight` (try 0.001 or 0.005)

**Problem: Model is overfitting**
- Solution: Increase `topology-weight` (try 0.02 or 0.05)

**Problem: Training is too slow**
- Solution: Increase `topology-interval` (compute less often)
- Or use smaller model (`--width 18` instead of 32)

**Problem: Out of memory**
- Solution: Reduce `batch-size` (try 64 or 32)
- Or reduce number of samples in topology computation

## Next Steps

### Learn More

1. **Read the guides:**
   - [README.md](README.md) - Project overview
   - [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - Detailed theory
   - [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Quick reference

2. **Experiment:**
   - Try different topology weights (0.001, 0.01, 0.1)
   - Compare different model widths (18, 32, 48)
   - Test on CIFAR-100 (100 classes)

3. **Customize:**
   - Modify `topology_analyzer.py` for different metrics
   - Adjust `train_enhanced.py` for your dataset
   - Create custom analysis scripts

### Example Experiments

**Experiment A: Topology Weight Sweep**
```bash
for weight in 0.0 0.001 0.01 0.1; do
    python train_enhanced.py \
        --dataset cifar10 \
        --topology-weight $weight \
        --output-dir ./output/sweep_$weight
done
```

**Experiment B: Model Size Comparison**
```bash
for width in 18 32 48; do
    python train_enhanced.py \
        --dataset cifar10 \
        --width $width \
        --topology-weight 0.01 \
        --output-dir ./output/width_$width
done
```

**Experiment C: Topology Interval Study**
```bash
for interval in 1 5 10 20; do
    python train_enhanced.py \
        --dataset cifar10 \
        --topology-weight 0.01 \
        --topology-interval $interval \
        --output-dir ./output/interval_$interval
done
```

## Tips for Success

1. **Start small:** Test with 1-2 epochs before full training
2. **Monitor TensorBoard:** Watch training in real-time
3. **Compare systematically:** Always train baseline first
4. **Save everything:** Keep all checkpoints and logs
5. **Document experiments:** Note settings and observations
6. **Be patient:** Topological analysis takes time but provides insights

## Quick Reference Commands

```bash
# Setup
./setup.sh && source venv/bin/activate

# Verify
python check_environment.py
python test_installation.py

# Train baseline
python train_enhanced.py --dataset cifar10 --topology-weight 0.0 --output-dir output/baseline

# Train with topology
python train_enhanced.py --dataset cifar10 --topology-weight 0.01 --output-dir output/optimized

# Analyze
python analyze_topology.py --checkpoint output/optimized/checkpoint_best.pth --output-dir analysis/optimized

# Monitor
tensorboard --logdir output/

# Quick test
jupyter notebook test_topology_optimization.ipynb
```

## Need Help?

1. **Check logs:**
   - Training logs: `output/*/tensorboard/`
   - Analysis reports: `analysis/*/analysis_report.txt`

2. **Review documentation:**
   - [INSTALL.md](INSTALL.md) - Installation issues
   - [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - Theory and interpretation
   - [README.md](README.md) - General information

3. **Common issues:**
   - Out of memory → Reduce batch size
   - Slow training → Increase topology interval
   - Poor results → Adjust topology weight

---

**Ready to start?** Run `./setup.sh` and then `./quick_start.sh`!
