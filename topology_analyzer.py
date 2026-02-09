"""
Topological Data Analysis Module for Image Classification
Uses persistent homology and bottleneck distance to optimize model performance

Compatible with latest library versions (2024+):
- ripser >= 0.6.0
- persim >= 0.3.0
- torch >= 2.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import warnings

# Import TDA libraries with error handling
try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    warnings.warn("ripser not available. Install with: pip install ripser")

try:
    from persim import bottleneck
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False
    warnings.warn("persim not available. Install with: pip install persim")

logger = logging.getLogger(__name__)


class TopologicalAnalyzer:
    """
    Analyzes neural network representations using topological data analysis.
    Computes bottleneck distance to measure representation quality and stability.
    """

    def __init__(self, max_dimension=1, distance_threshold=1.0):
        """
        Initialize topological analyzer.

        Args:
            max_dimension: Maximum homology dimension to compute (0=components, 1=loops, 2=voids)
            distance_threshold: Maximum distance for persistence computation
        """
        self.max_dimension = max_dimension
        self.distance_threshold = distance_threshold
        self.baseline_diagrams = {}
        self.history = {
            'bottleneck_distances': [],
            'persistence_entropies': [],
            'betti_numbers': []
        }

    def compute_persistence_diagram(self, features: np.ndarray,
                                     label: Optional[str] = None) -> Dict:
        """
        Compute persistence diagram from feature representations.

        Args:
            features: Feature matrix (n_samples, n_features)
            label: Optional label for this computation

        Returns:
            Dictionary containing persistence diagrams and statistics
        """
        if not RIPSER_AVAILABLE:
            logger.warning("ripser not available, returning default stats")
            return self._get_default_stats()

        try:
            # Normalize features
            features = self._normalize_features(features)

            # Sample if too many points (for computational efficiency)
            if features.shape[0] > 500:
                indices = np.random.choice(features.shape[0], 500, replace=False)
                features = features[indices]

            # Ensure we have enough samples
            if features.shape[0] < 3:
                logger.warning(f"Too few samples ({features.shape[0]}), need at least 3")
                return self._get_default_stats()

            # Compute persistence using Vietoris-Rips complex
            # Updated API for ripser 0.6.0+
            result = ripser(
                features,
                maxdim=self.max_dimension,
                thresh=self.distance_threshold,
                coeff=2  # Use Z/2Z coefficients for faster computation
            )

            # Extract persistence diagrams
            diagrams = result['dgms']

            # Ensure diagrams are in correct format
            if not isinstance(diagrams, list):
                diagrams = [diagrams]

            # Compute topological features
            stats = {
                'diagrams': diagrams,
                'betti_numbers': self._compute_betti_numbers(diagrams),
                'persistence_entropy': self._compute_persistence_entropy(diagrams),
                'lifetime_stats': self._compute_lifetime_stats(diagrams)
            }

            if label:
                self.baseline_diagrams[label] = diagrams

            return stats

        except Exception as e:
            logger.error(f"Error computing persistence diagram: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._get_default_stats()

    def _get_default_stats(self) -> Dict:
        """Return default statistics when computation fails"""
        return {
            'diagrams': [np.array([[0, 0]])],
            'betti_numbers': [0],
            'persistence_entropy': 0.0,
            'lifetime_stats': {}
        }

    def compute_bottleneck_distance(self,
                                     features1: np.ndarray,
                                     features2: np.ndarray,
                                     dimension: int = 1) -> float:
        """
        Compute bottleneck distance between two feature representations.
        This measures the stability/similarity of topological structures.

        Args:
            features1: First feature matrix
            features2: Second feature matrix
            dimension: Homology dimension to compare

        Returns:
            Bottleneck distance value
        """
        if not PERSIM_AVAILABLE:
            logger.warning("persim not available, returning inf")
            return float('inf')

        try:
            # Compute persistence diagrams
            stats1 = self.compute_persistence_diagram(features1)
            stats2 = self.compute_persistence_diagram(features2)

            if stats1 is None or stats2 is None:
                return float('inf')

            # Ensure dimension is available
            if dimension >= len(stats1['diagrams']) or dimension >= len(stats2['diagrams']):
                logger.warning(f"Dimension {dimension} not available, using dimension 0")
                dimension = 0

            dgm1 = stats1['diagrams'][dimension]
            dgm2 = stats2['diagrams'][dimension]

            # Ensure diagrams are numpy arrays
            if not isinstance(dgm1, np.ndarray):
                dgm1 = np.array(dgm1)
            if not isinstance(dgm2, np.ndarray):
                dgm2 = np.array(dgm2)

            # Handle empty diagrams
            if len(dgm1) == 0 or len(dgm2) == 0:
                logger.warning("Empty persistence diagram encountered")
                return float('inf')

            # Compute bottleneck distance
            # persim 0.3.0+ API
            distance = bottleneck(dgm1, dgm2, matching=False)

            # Ensure distance is a scalar
            if isinstance(distance, np.ndarray):
                distance = float(distance.item())
            else:
                distance = float(distance)

            self.history['bottleneck_distances'].append(distance)

            return distance

        except Exception as e:
            logger.error(f"Error computing bottleneck distance: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return float('inf')

    def analyze_layer_topology(self,
                                model: nn.Module,
                                input_data: torch.Tensor,
                                layer_names: List[str]) -> Dict:
        """
        Analyze topological properties of intermediate layer representations.

        Args:
            model: Neural network model
            input_data: Input tensor
            layer_names: List of layer names to analyze

        Returns:
            Dictionary of topological statistics per layer
        """
        model.eval()
        layer_stats = {}

        # Hook to capture layer outputs
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name)))

        # Forward pass
        with torch.no_grad():
            _ = model(input_data)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Analyze each layer
        for name, activation in activations.items():
            # Flatten spatial dimensions if present
            if len(activation.shape) > 2:
                batch_size = activation.shape[0]
                features = activation.reshape(batch_size, -1).cpu().numpy()
            else:
                features = activation.cpu().numpy()

            stats = self.compute_persistence_diagram(features, label=name)
            if stats:
                layer_stats[name] = stats
                logger.info(f"Layer {name}: Betti numbers = {stats['betti_numbers']}")

        return layer_stats

    def compute_topological_loss(self,
                                  current_features: np.ndarray,
                                  target_topology: Optional[Dict] = None) -> float:
        """
        Compute a topological regularization loss.
        Encourages the model to maintain beneficial topological properties.

        Args:
            current_features: Current feature representations
            target_topology: Target topological properties (optional)

        Returns:
            Topological loss value
        """
        try:
            current_stats = self.compute_persistence_diagram(current_features)

            if current_stats is None:
                return 0.0

            # If no target topology, encourage non-trivial topology
            if target_topology is None:
                # Penalize trivial topology (few persistent features)
                persistence_entropy = current_stats['persistence_entropy']
                # Encourage higher entropy (more complex topology)
                loss = -np.log(persistence_entropy + 1e-8)
            else:
                # Compute bottleneck distance to target topology
                if 'diagrams' in target_topology:
                    distance = 0.0
                    for dim in range(self.max_dimension + 1):
                        if dim < len(current_stats['diagrams']) and \
                           dim < len(target_topology['diagrams']):
                            distance += bottleneck(
                                current_stats['diagrams'][dim],
                                target_topology['diagrams'][dim]
                            )
                    loss = distance
                else:
                    loss = 0.0

            return loss

        except Exception as e:
            logger.error(f"Error computing topological loss: {e}")
            return 0.0

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to unit scale."""
        # Handle NaN and inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Standardize features
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        return features

    def _compute_betti_numbers(self, diagrams: List[np.ndarray]) -> List[int]:
        """
        Compute Betti numbers (count of persistent features at each dimension).
        Betti_0 = connected components, Betti_1 = loops, etc.
        """
        betti = []
        for dim, dgm in enumerate(diagrams):
            if len(dgm) > 0:
                # Count features with significant persistence
                persistence = dgm[:, 1] - dgm[:, 0]
                # Filter out near-zero persistence
                significant = np.sum(persistence > 0.1)
                betti.append(int(significant))
            else:
                betti.append(0)

        self.history['betti_numbers'].append(betti)
        return betti

    def _compute_persistence_entropy(self, diagrams: List[np.ndarray]) -> float:
        """
        Compute persistence entropy - measure of complexity of topological features.
        Higher entropy indicates more diverse topological structures.
        """
        total_entropy = 0.0

        for dgm in diagrams:
            if len(dgm) > 0:
                # Compute persistence (lifetime)
                persistence = dgm[:, 1] - dgm[:, 0]
                # Remove infinite persistence
                persistence = persistence[np.isfinite(persistence)]

                if len(persistence) > 0:
                    # Normalize to get probability distribution
                    L = np.sum(persistence)
                    if L > 0:
                        p = persistence / L
                        # Compute entropy
                        entropy = -np.sum(p * np.log(p + 1e-10))
                        total_entropy += entropy

        self.history['persistence_entropies'].append(total_entropy)
        return total_entropy

    def _compute_lifetime_stats(self, diagrams: List[np.ndarray]) -> Dict:
        """Compute statistics of feature lifetimes."""
        stats = {}

        for dim, dgm in enumerate(diagrams):
            if len(dgm) > 0:
                persistence = dgm[:, 1] - dgm[:, 0]
                persistence = persistence[np.isfinite(persistence)]

                if len(persistence) > 0:
                    stats[f'dim_{dim}'] = {
                        'mean_lifetime': float(np.mean(persistence)),
                        'max_lifetime': float(np.max(persistence)),
                        'total_persistence': float(np.sum(persistence)),
                        'num_features': len(persistence)
                    }

        return stats

    def visualize_persistence_diagram(self,
                                       features: np.ndarray,
                                       save_path: Optional[str] = None):
        """Visualize persistence diagrams."""
        stats = self.compute_persistence_diagram(features)

        if stats is None:
            return

        diagrams = stats['diagrams']

        fig, axes = plt.subplots(1, len(diagrams), figsize=(5*len(diagrams), 5))
        if len(diagrams) == 1:
            axes = [axes]

        for dim, (ax, dgm) in enumerate(zip(axes, diagrams)):
            if len(dgm) > 0:
                ax.scatter(dgm[:, 0], dgm[:, 1], alpha=0.6)

                # Plot diagonal
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),
                    np.max([ax.get_xlim(), ax.get_ylim()]),
                ]
                ax.plot(lims, lims, 'k-', alpha=0.3, zorder=0)

                ax.set_xlabel('Birth')
                ax.set_ylabel('Death')
                ax.set_title(f'H_{dim} Persistence Diagram')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Persistence diagram saved to {save_path}")

        plt.close()

    def get_summary_statistics(self) -> Dict:
        """Get summary of topological analysis history."""
        summary = {
            'avg_bottleneck_distance': np.mean(self.history['bottleneck_distances'])
                if self.history['bottleneck_distances'] else 0.0,
            'avg_persistence_entropy': np.mean(self.history['persistence_entropies'])
                if self.history['persistence_entropies'] else 0.0,
            'latest_betti_numbers': self.history['betti_numbers'][-1]
                if self.history['betti_numbers'] else [],
            'num_analyses': len(self.history['bottleneck_distances'])
        }
        return summary


class TopologyAwareTraining:
    """
    Training utilities that incorporate topological analysis
    for adaptive optimization.
    """

    def __init__(self, topology_weight: float = 0.01):
        """
        Initialize topology-aware training.

        Args:
            topology_weight: Weight for topological regularization loss
        """
        self.analyzer = TopologicalAnalyzer()
        self.topology_weight = topology_weight
        self.reference_topology = None

    def compute_combined_loss(self,
                               predictions: torch.Tensor,
                               targets: torch.Tensor,
                               features: torch.Tensor,
                               base_criterion: nn.Module) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss with topological regularization.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            features: Intermediate feature representations
            base_criterion: Base loss function (e.g., CrossEntropyLoss)

        Returns:
            Combined loss tensor and statistics dictionary
        """
        # Base classification loss
        base_loss = base_criterion(predictions, targets)

        # Topological loss (if enabled)
        topo_loss = 0.0
        if self.topology_weight > 0:
            # Extract features as numpy array
            features_np = features.detach().cpu().numpy()

            # Flatten if needed
            if len(features_np.shape) > 2:
                batch_size = features_np.shape[0]
                features_np = features_np.reshape(batch_size, -1)

            # Compute topological loss
            topo_loss_value = self.analyzer.compute_topological_loss(
                features_np,
                target_topology=self.reference_topology
            )
            topo_loss = self.topology_weight * topo_loss_value

        # Combined loss
        total_loss = base_loss + topo_loss

        stats = {
            'base_loss': base_loss.item(),
            'topo_loss': float(topo_loss) if isinstance(topo_loss, (int, float)) else topo_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, stats

    def set_reference_topology(self, features: np.ndarray):
        """Set reference topology from well-trained model or ideal representation."""
        self.reference_topology = self.analyzer.compute_persistence_diagram(features)
        logger.info("Reference topology set for regularization")
