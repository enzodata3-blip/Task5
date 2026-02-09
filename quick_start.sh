#!/bin/bash

# Quick Start Script for HRNet Topological Optimization
# This script sets up and runs a basic training experiment

set -e

echo "=========================================="
echo "HRNet Topological Optimization Quick Start"
echo "=========================================="
echo ""

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements_enhanced.txt

echo "✓ Dependencies installed"
echo ""

# Step 2: Create output directories
echo "Step 2: Creating output directories..."
mkdir -p output/quick_start
mkdir -p data
mkdir -p analysis_output

echo "✓ Directories created"
echo ""

# Step 3: Run a quick training experiment
echo "Step 3: Starting training experiment..."
echo "Training HRNet-W18 on CIFAR-10 with topological optimization"
echo ""
echo "Configuration:"
echo "  - Dataset: CIFAR-10"
echo "  - Model: HRNet-W18"
echo "  - Epochs: 50 (reduced for quick start)"
echo "  - Batch size: 128"
echo "  - Topology weight: 0.01"
echo "  - Topology analysis interval: 5 epochs"
echo ""

python train_enhanced.py \
    --dataset cifar10 \
    --batch-size 128 \
    --epochs 50 \
    --lr 0.1 \
    --width 18 \
    --topology-weight 0.01 \
    --topology-interval 5 \
    --output-dir ./output/quick_start

echo ""
echo "✓ Training completed"
echo ""

# Step 4: Run topological analysis
echo "Step 4: Running topological analysis..."
echo ""

python analyze_topology.py \
    --checkpoint ./output/quick_start/checkpoint_best.pth \
    --dataset cifar10 \
    --output-dir ./analysis_output/quick_start \
    --num-samples 500 \
    --analyze-layers

echo ""
echo "✓ Analysis completed"
echo ""

# Step 5: Display results
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""
echo "Training outputs:"
echo "  - Checkpoints: ./output/quick_start/"
echo "  - TensorBoard logs: ./output/quick_start/tensorboard/"
echo ""
echo "Analysis outputs:"
echo "  - Full report: ./analysis_output/quick_start/analysis_report.txt"
echo "  - Distance matrix: ./analysis_output/quick_start/distance_matrix.png"
echo "  - Topology evolution: ./analysis_output/quick_start/topology_evolution.png"
echo "  - Persistence diagrams: ./analysis_output/quick_start/persistence_diagrams/"
echo ""
echo "To view training progress with TensorBoard:"
echo "  tensorboard --logdir ./output/quick_start/tensorboard"
echo ""
echo "To view the analysis report:"
echo "  cat ./analysis_output/quick_start/analysis_report.txt"
echo ""
echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
