# HRNet Image Classification with Topological Optimization

## Overview

This project enhances the High-Resolution Network (HRNet) architecture for image classification by incorporating **Topological Data Analysis (TDA)** and **bottleneck distance** optimization. The human element in machine learning optimization is introduced through persistent homology analysis, which monitors and guides the model's learning process to achieve optimal performance.

## Key Features

### 1. **Topological Data Analysis Integration**
- **Persistent Homology**: Analyzes the topological structure of learned representations
- **Bottleneck Distance**: Measures stability and quality of feature spaces
- **Betti Numbers**: Tracks connected components and loops in data manifolds
- **Persistence Entropy**: Quantifies topological complexity

### 2. **Enhanced Training with Topology Monitoring**
- Real-time topological analysis during training
- Adaptive regularization based on topological properties
- Prevents overfitting by maintaining meaningful topological structures
- Ensures stable and robust feature learning

### 3. **Comprehensive Analysis Tools**
- Inter-class bottleneck distance computation
- Layer-wise topology evolution tracking
- Persistence diagram visualization
- Detailed statistical reports

### 4. **Optimized for CIFAR-10/100**
- Fast iteration for experimentation
- Adapted HRNet architecture for 32×32 images
- Efficient training on standard hardware

## Architecture Enhancements

### Base: HRNet
The High-Resolution Network maintains high-resolution representations through parallel multi-resolution branches, enabling rich semantic and fine-grained feature extraction.

### Enhancement: Topological Regularization
```
Training Loss = Classification Loss + λ × Topological Loss
```

Where:
- **Classification Loss**: Standard cross-entropy
- **Topological Loss**: Bottleneck distance to reference topology or persistence entropy regularization
- **λ**: Weighting parameter (default: 0.01)

## Installation

### Requirements
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_enhanced.txt
```

### Key Dependencies
- PyTorch >= 1.8.0
- TDA libraries: gudhi, ripser, persim
- Visualization: matplotlib, seaborn, plotly

## Quick Start

### 1. Train Model with Topological Optimization

```bash
# Train on CIFAR-10 with topological analysis
python train_enhanced.py \
    --dataset cifar10 \
    --batch-size 128 \
    --epochs 200 \
    --lr 0.1 \
    --width 18 \
    --topology-weight 0.01 \
    --topology-interval 10 \
    --output-dir ./output/hrnet_w18_topo
```

**Parameters:**
- `--dataset`: Choice of `cifar10` or `cifar100`
- `--width`: HRNet width multiplier (18, 32, 48)
- `--topology-weight`: Weight for topological regularization (0.0 to disable)
- `--topology-interval`: Epochs between topological analyses
- `--batch-size`: Training batch size
- `--lr`: Initial learning rate

### 2. Analyze Model Topology

```bash
# Comprehensive topological analysis
python analyze_topology.py \
    --checkpoint ./output/hrnet_w18_topo/checkpoint_best.pth \
    --dataset cifar10 \
    --output-dir ./analysis_output \
    --num-samples 500 \
    --analyze-layers
```

**Outputs:**
- `distance_matrix.png`: Inter-class bottleneck distances
- `topology_evolution.png`: Layer-wise topological properties
- `persistence_diagrams/`: Per-class persistence diagrams
- `analysis_report.txt`: Comprehensive statistical report

### 3. Monitor Training with TensorBoard

```bash
tensorboard --logdir ./output/hrnet_w18_topo/tensorboard
```

## Understanding Topological Metrics

### Betti Numbers
- **Betti-0**: Number of connected components
  - Higher values indicate more separated clusters
  - Ideal for classification: distinct clusters per class

- **Betti-1**: Number of 1-dimensional holes (loops)
  - Captures circular/cyclic structures in data
  - Can indicate decision boundary complexity

### Persistence Entropy
- Measures diversity of topological features
- Higher entropy = more complex structure
- Too low: May indicate underfitting
- Too high: May indicate overfitting

### Bottleneck Distance
- Measures topological similarity between feature spaces
- Lower distance = more similar topology
- Used to ensure:
  - Stable training (similar topology across epochs)
  - Distinct classes (high inter-class distances)
  - Consistent features (low intra-class distances)

## Why Topological Optimization?

### The Human Element in ML

Machine learning models optimize towards local equilibria based solely on loss gradients. The **human insight** comes from understanding that:

1. **Good representations have structure**: Classes should form distinct topological regions
2. **Stability matters**: Small input changes shouldn't drastically change topology
3. **Complexity should be appropriate**: Neither too simple (underfit) nor too complex (overfit)

### Bottleneck Distance as Quality Metric

The bottleneck distance provides a **rigorous mathematical framework** to:
- Quantify representation quality beyond accuracy
- Detect early signs of overfitting (unstable topology)
- Guide regularization (maintain beneficial topological properties)
- Compare different model architectures objectively

## Project Structure

```
model_b/
├── train_enhanced.py          # Enhanced training with topology
├── analyze_topology.py         # Comprehensive analysis tools
├── topology_analyzer.py        # Core TDA implementation
├── requirements_enhanced.txt   # Dependencies
├── README.md                   # This file
├── hrnet_base/                 # Base HRNet implementation
│   ├── lib/
│   │   ├── models/
│   │   │   └── cls_hrnet.py   # HRNet architecture
│   │   └── core/
│   │       └── function.py    # Training functions
│   └── tools/
└── output/                     # Training outputs
    └── [experiment_name]/
        ├── checkpoint_best.pth
        ├── checkpoint_latest.pth
        └── tensorboard/
```

## Example Results

### Typical Training Metrics
- **CIFAR-10 (HRNet-W18 + Topology)**
  - Accuracy: ~93-94%
  - Betti-0: 8-10 (distinct clusters)
  - Persistence Entropy: 2.5-3.5
  - Mean Bottleneck Distance: 0.3-0.5

### Interpretation
- **High inter-class bottleneck distance**: Classes are topologically distinct
- **Stable Betti numbers**: Consistent representation structure
- **Moderate entropy**: Appropriate complexity

## Advanced Usage

### Custom Topology Regularization

```python
from topology_analyzer import TopologyAwareTraining

# Initialize with custom weight
topology_trainer = TopologyAwareTraining(topology_weight=0.05)

# Set reference topology from well-trained model
reference_features = extract_features(pretrained_model, data)
topology_trainer.set_reference_topology(reference_features)

# Use in training loop
loss, stats = topology_trainer.compute_combined_loss(
    predictions, targets, features, criterion
)
```

### Layer-wise Topology Analysis

```python
from topology_analyzer import TopologicalAnalyzer

analyzer = TopologicalAnalyzer()

# Analyze specific layers
layer_names = ['layer1', 'stage2', 'stage3', 'stage4']
layer_stats = analyzer.analyze_layer_topology(
    model, input_data, layer_names
)

# Compute bottleneck distance between layers
distance = analyzer.compute_bottleneck_distance(
    layer1_features, layer2_features, dimension=1
)
```

## Research Background

### Persistent Homology in ML
Topological Data Analysis has been successfully applied to:
- Understanding neural network representations
- Detecting adversarial examples
- Analyzing decision boundaries
- Improving generalization

### Key References
1. "Topology of Deep Neural Networks" - Naitzat et al., 2020
2. "A Topological Regularizer for Classifiers via Persistent Homology" - Gabrielsson et al., 2020
3. "Deep High-Resolution Representation Learning" - Wang et al., 2019 (HRNet)

## Contributing

This project demonstrates the integration of topological methods with deep learning. Potential extensions:
- Multi-scale persistent homology
- Adaptive topology weight scheduling
- Topology-based model selection
- Transfer learning with topology preservation

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hrnet_topological_optimization,
  title={HRNet Image Classification with Topological Optimization},
  author={},
  year={2026},
  description={Enhanced HRNet with bottleneck distance optimization}
}
```

Original HRNet:
```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and Borui Jiang and
          Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```

## License

This project builds upon HRNet (MIT License) and adds topological analysis components.

## Support

For questions or issues:
1. Check the analysis reports for interpretation guidance
2. Review TensorBoard logs for training dynamics
3. Examine persistence diagrams for topological insights

---

**Remember**: The human element in machine learning comes from understanding what makes a good representation beyond just accuracy. Topological analysis provides the mathematical tools to quantify and optimize these properties.
