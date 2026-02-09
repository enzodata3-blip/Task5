"""
Topological Analysis and Visualization Script
Analyzes model representations using bottleneck distance and persistence diagrams
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hrnet_base', 'lib'))

from topology_analyzer import TopologicalAnalyzer
from train_enhanced import HRNetCIFAR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features_by_class(model, data_loader, device, num_classes=10, max_samples=1000):
    """Extract features organized by class label"""
    model.eval()

    features_by_class = {i: [] for i in range(num_classes)}
    samples_by_class = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Extracting features'):
            data = data.to(device)

            # Extract features
            _, features = model(data, return_features=True)
            features = features.cpu().numpy()

            # Organize by class
            for feat, label in zip(features, target.numpy()):
                label = int(label)
                if samples_by_class[label] < max_samples:
                    features_by_class[label].append(feat)
                    samples_by_class[label] += 1

            # Check if we have enough samples
            if all(count >= max_samples for count in samples_by_class.values()):
                break

    # Convert to numpy arrays
    for label in features_by_class:
        if features_by_class[label]:
            features_by_class[label] = np.array(features_by_class[label])

    return features_by_class


def analyze_class_topology(features_by_class, analyzer, output_dir):
    """Analyze topological properties of each class"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_stats = {}

    logger.info("Analyzing topology for each class...")

    for label, features in features_by_class.items():
        if len(features) > 0:
            logger.info(f"Class {label}: {len(features)} samples")

            # Compute persistence diagram
            stats = analyzer.compute_persistence_diagram(features, label=f'class_{label}')

            if stats:
                class_stats[label] = stats

                # Log statistics
                logger.info(f"  Betti numbers: {stats['betti_numbers']}")
                logger.info(f"  Persistence entropy: {stats['persistence_entropy']:.4f}")

                # Visualize persistence diagram
                analyzer.visualize_persistence_diagram(
                    features,
                    save_path=output_dir / f'persistence_class_{label}.png'
                )

    return class_stats


def compute_inter_class_distances(features_by_class, analyzer):
    """Compute bottleneck distances between all pairs of classes"""
    num_classes = len(features_by_class)
    distance_matrix = np.zeros((num_classes, num_classes))

    logger.info("Computing inter-class bottleneck distances...")

    for i in tqdm(range(num_classes)):
        for j in range(i+1, num_classes):
            if len(features_by_class[i]) > 0 and len(features_by_class[j]) > 0:
                distance = analyzer.compute_bottleneck_distance(
                    features_by_class[i],
                    features_by_class[j],
                    dimension=1
                )
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    return distance_matrix


def visualize_distance_matrix(distance_matrix, class_names, output_path):
    """Visualize bottleneck distance matrix"""
    plt.figure(figsize=(12, 10))

    sns.heatmap(distance_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={'label': 'Bottleneck Distance'})

    plt.title('Inter-Class Bottleneck Distance Matrix\n(Lower values = more similar topology)',
              fontsize=14, pad=20)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Distance matrix saved to {output_path}")
    plt.close()


def analyze_topology_evolution(model, data_loader, device, analyzer, layer_names, output_dir):
    """Analyze how topology evolves through network layers"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a batch of data
    data, _ = next(iter(data_loader))
    data = data.to(device)

    logger.info("Analyzing topology evolution through layers...")

    # Analyze layer topology
    layer_stats = analyzer.analyze_layer_topology(model, data, layer_names)

    # Plot evolution
    if layer_stats:
        layers = list(layer_stats.keys())
        betti_0 = [layer_stats[layer]['betti_numbers'][0] if len(layer_stats[layer]['betti_numbers']) > 0 else 0
                   for layer in layers]
        betti_1 = [layer_stats[layer]['betti_numbers'][1] if len(layer_stats[layer]['betti_numbers']) > 1 else 0
                   for layer in layers]
        entropies = [layer_stats[layer]['persistence_entropy'] for layer in layers]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Betti-0 (connected components)
        axes[0].plot(range(len(layers)), betti_0, marker='o', linewidth=2, markersize=8)
        axes[0].set_ylabel('Betti-0 (Components)', fontsize=11)
        axes[0].set_title('Topological Evolution Through Network Layers', fontsize=14, pad=15)
        axes[0].grid(True, alpha=0.3)

        # Betti-1 (loops)
        axes[1].plot(range(len(layers)), betti_1, marker='s', linewidth=2, markersize=8, color='orange')
        axes[1].set_ylabel('Betti-1 (Loops)', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # Persistence entropy
        axes[2].plot(range(len(layers)), entropies, marker='^', linewidth=2, markersize=8, color='green')
        axes[2].set_ylabel('Persistence Entropy', fontsize=11)
        axes[2].set_xlabel('Layer', fontsize=11)
        axes[2].set_xticks(range(len(layers)))
        axes[2].set_xticklabels(layers, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'topology_evolution.png', dpi=150, bbox_inches='tight')
        logger.info(f"Topology evolution plot saved")
        plt.close()

    return layer_stats


def generate_report(class_stats, distance_matrix, layer_stats, output_path):
    """Generate comprehensive analysis report"""
    report_lines = []

    report_lines.append("="*80)
    report_lines.append("TOPOLOGICAL ANALYSIS REPORT")
    report_lines.append("HRNet Image Classification with Bottleneck Distance Optimization")
    report_lines.append("="*80)
    report_lines.append("")

    # Class-wise statistics
    report_lines.append("1. PER-CLASS TOPOLOGICAL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Class':<10} {'Betti-0':<12} {'Betti-1':<12} {'Entropy':<15}")
    report_lines.append("-" * 80)

    for label, stats in sorted(class_stats.items()):
        betti = stats['betti_numbers']
        betti_0 = betti[0] if len(betti) > 0 else 0
        betti_1 = betti[1] if len(betti) > 1 else 0
        entropy = stats['persistence_entropy']
        report_lines.append(f"{label:<10} {betti_0:<12} {betti_1:<12} {entropy:<15.4f}")

    report_lines.append("")
    report_lines.append("")

    # Distance matrix statistics
    report_lines.append("2. INTER-CLASS BOTTLENECK DISTANCES")
    report_lines.append("-" * 80)

    # Get upper triangle of distance matrix (excluding diagonal)
    upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    report_lines.append(f"Mean bottleneck distance: {np.mean(upper_triangle):.4f}")
    report_lines.append(f"Std bottleneck distance: {np.std(upper_triangle):.4f}")
    report_lines.append(f"Min bottleneck distance: {np.min(upper_triangle):.4f}")
    report_lines.append(f"Max bottleneck distance: {np.max(upper_triangle):.4f}")

    # Find most similar and most different class pairs
    if len(upper_triangle) > 0:
        min_idx = np.unravel_index(np.argmin(distance_matrix + np.eye(len(distance_matrix)) * 1e10),
                                    distance_matrix.shape)
        max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)

        report_lines.append(f"\nMost similar classes: {min_idx[0]} and {min_idx[1]} "
                            f"(distance = {distance_matrix[min_idx]:.4f})")
        report_lines.append(f"Most different classes: {max_idx[0]} and {max_idx[1]} "
                            f"(distance = {distance_matrix[max_idx]:.4f})")

    report_lines.append("")
    report_lines.append("")

    # Layer evolution
    if layer_stats:
        report_lines.append("3. TOPOLOGY EVOLUTION THROUGH LAYERS")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Layer':<30} {'Betti-0':<12} {'Betti-1':<12} {'Entropy':<15}")
        report_lines.append("-" * 80)

        for layer_name, stats in layer_stats.items():
            betti = stats['betti_numbers']
            betti_0 = betti[0] if len(betti) > 0 else 0
            betti_1 = betti[1] if len(betti) > 1 else 0
            entropy = stats['persistence_entropy']
            report_lines.append(f"{layer_name:<30} {betti_0:<12} {betti_1:<12} {entropy:<15.4f}")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("INTERPRETATION GUIDE")
    report_lines.append("="*80)
    report_lines.append("""
Betti Numbers:
- Betti-0: Number of connected components (higher = more separated clusters)
- Betti-1: Number of loops/cycles (higher = more complex structure)

Persistence Entropy:
- Measure of topological complexity (higher = more diverse structures)

Bottleneck Distance:
- Measures similarity between topological structures
- Lower values = more similar representations
- Used to ensure stable learning and robust features
""")

    # Write report
    report_text = "\n".join(report_lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Report saved to {output_path}")

    # Print summary to console
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description='Topological Analysis of HRNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default='./analysis_output')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Number of samples per class for analysis')
    parser.add_argument('--analyze-layers', action='store_true',
                        help='Analyze topology evolution through layers')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading {args.dataset.upper()} dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform)
        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform)
        num_classes = 100
        class_names = [str(i) for i in range(100)]

    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = HRNetCIFAR(num_classes=num_classes, width=checkpoint['args'].width)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded (accuracy: {checkpoint['best_acc']:.2f}%)")

    # Initialize analyzer
    analyzer = TopologicalAnalyzer(max_dimension=1)

    # Extract features by class
    logger.info("Extracting features by class...")
    features_by_class = extract_features_by_class(
        model, data_loader, device, num_classes, args.num_samples
    )

    # Analyze class topology
    class_stats = analyze_class_topology(features_by_class, analyzer, output_dir / 'persistence_diagrams')

    # Compute inter-class distances
    distance_matrix = compute_inter_class_distances(features_by_class, analyzer)

    # Visualize distance matrix
    visualize_distance_matrix(distance_matrix, class_names, output_dir / 'distance_matrix.png')

    # Analyze layer evolution (if requested)
    layer_stats = {}
    if args.analyze_layers:
        layer_names = ['layer1', 'stage2', 'stage3', 'stage4']
        layer_stats = analyze_topology_evolution(
            model, data_loader, device, analyzer, layer_names, output_dir
        )

    # Generate comprehensive report
    generate_report(class_stats, distance_matrix, layer_stats, output_dir / 'analysis_report.txt')

    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()
