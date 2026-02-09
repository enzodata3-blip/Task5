#!/usr/bin/env python3
"""
Comprehensive Installation Test
Tests all components to ensure everything works correctly
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test basic imports"""
    print("=" * 80)
    print("TEST 1: Import Core Libraries")
    print("=" * 80)

    tests = []

    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        tests.append(False)

    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        tests.append(False)

    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ torchvision import failed: {e}")
        tests.append(False)

    try:
        from ripser import ripser
        print(f"✓ ripser available")
        tests.append(True)
    except ImportError as e:
        print(f"✗ ripser import failed: {e}")
        tests.append(False)

    try:
        from persim import bottleneck
        print(f"✓ persim available")
        tests.append(True)
    except ImportError as e:
        print(f"✗ persim import failed: {e}")
        tests.append(False)

    print()
    return all(tests)


def test_custom_modules():
    """Test custom modules"""
    print("=" * 80)
    print("TEST 2: Import Custom Modules")
    print("=" * 80)

    # Add hrnet_base to path
    project_root = Path(os.getcwd())
    hrnet_lib_path = project_root / 'hrnet_base' / 'lib'
    if hrnet_lib_path.exists():
        sys.path.insert(0, str(hrnet_lib_path))
        print(f"✓ HRNet library path added: {hrnet_lib_path}")
    else:
        print(f"⚠ Warning: HRNet library path not found: {hrnet_lib_path}")

    tests = []

    try:
        from topology_analyzer import TopologicalAnalyzer, TopologyAwareTraining
        print("✓ topology_analyzer imported")
        tests.append(True)
    except ImportError as e:
        print(f"✗ topology_analyzer import failed: {e}")
        tests.append(False)

    try:
        from train_enhanced import HRNetCIFAR
        print("✓ train_enhanced imported")
        tests.append(True)
    except ImportError as e:
        print(f"✗ train_enhanced import failed: {e}")
        tests.append(False)

    print()
    return all(tests)


def test_topology_analyzer():
    """Test TopologicalAnalyzer"""
    print("=" * 80)
    print("TEST 3: TopologicalAnalyzer Functionality")
    print("=" * 80)

    try:
        import numpy as np
        from topology_analyzer import TopologicalAnalyzer

        # Create test data
        np.random.seed(42)
        data = np.random.randn(50, 10)

        # Initialize analyzer
        analyzer = TopologicalAnalyzer(max_dimension=1, distance_threshold=5.0)
        print("✓ TopologicalAnalyzer initialized")

        # Compute persistence diagram
        stats = analyzer.compute_persistence_diagram(data, label='test')

        if stats and 'betti_numbers' in stats:
            print(f"✓ Persistence diagram computed")
            print(f"  Betti numbers: {stats['betti_numbers']}")
            print(f"  Persistence entropy: {stats['persistence_entropy']:.4f}")

            # Test bottleneck distance
            data2 = np.random.randn(50, 10)
            distance = analyzer.compute_bottleneck_distance(data, data2, dimension=0)

            if distance < float('inf'):
                print(f"✓ Bottleneck distance computed: {distance:.4f}")
                print()
                return True
            else:
                print("✗ Bottleneck distance computation failed")
                print()
                return False
        else:
            print("✗ Persistence diagram computation failed")
            print()
            return False

    except Exception as e:
        print(f"✗ TopologicalAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_model():
    """Test HRNet model"""
    print("=" * 80)
    print("TEST 4: HRNet Model")
    print("=" * 80)

    try:
        import torch
        from train_enhanced import HRNetCIFAR

        # Create model
        model = HRNetCIFAR(num_classes=10, width=18)
        print(f"✓ Model created")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        dummy_input = torch.randn(4, 3, 32, 32).to(device)

        # Standard forward
        output = model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")

        # Forward with features
        output, features = model(dummy_input, return_features=True)
        print(f"✓ Forward with features successful")
        print(f"  Features shape: {features.shape}")

        print()
        return True

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_data_loading():
    """Test CIFAR-10 data loading"""
    print("=" * 80)
    print("TEST 5: Data Loading (CIFAR-10)")
    print("=" * 80)

    try:
        import torch
        import torchvision
        import torchvision.transforms as transforms

        # Prepare transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load small subset
        print("Downloading CIFAR-10 (if needed)...")
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        print(f"✓ Dataset loaded")
        print(f"  Samples: {len(dataset)}")
        print(f"  Classes: {dataset.classes}")

        # Create data loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0  # Use 0 for compatibility
        )

        # Test loading one batch
        images, labels = next(iter(loader))
        print(f"✓ Data loader working")
        print(f"  Batch shape: {images.shape}")

        print()
        return True

    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_combined_loss():
    """Test topology-aware training"""
    print("=" * 80)
    print("TEST 6: Topology-Aware Loss Computation")
    print("=" * 80)

    try:
        import torch
        import torch.nn as nn
        from train_enhanced import HRNetCIFAR
        from topology_analyzer import TopologyAwareTraining

        # Create model and data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HRNetCIFAR(num_classes=10, width=18).to(device)
        dummy_input = torch.randn(8, 3, 32, 32).to(device)
        dummy_labels = torch.randint(0, 10, (8,)).to(device)

        # Initialize topology trainer
        topology_trainer = TopologyAwareTraining(topology_weight=0.01)
        criterion = nn.CrossEntropyLoss()

        print("✓ Components initialized")

        # Forward pass
        with torch.no_grad():
            output, features = model(dummy_input, return_features=True)

        # Compute combined loss
        loss, stats = topology_trainer.compute_combined_loss(
            output, dummy_labels, features, criterion
        )

        print(f"✓ Combined loss computed")
        print(f"  Base loss: {stats['base_loss']:.4f}")
        print(f"  Topological loss: {stats['topo_loss']:.4f}")
        print(f"  Total loss: {stats['total_loss']:.4f}")

        print()
        return True

    except Exception as e:
        print(f"✗ Combined loss test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("*" * 80)
    print("COMPREHENSIVE INSTALLATION TEST")
    print("*" * 80)
    print("\n")

    results = []

    # Run all tests
    results.append(("Import Core Libraries", test_imports()))
    results.append(("Import Custom Modules", test_custom_modules()))
    results.append(("TopologicalAnalyzer", test_topology_analyzer()))
    results.append(("HRNet Model", test_model()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Combined Loss", test_combined_loss()))

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12s} {test_name}")
        if not passed:
            all_passed = False

    print()
    print("=" * 80)

    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print()
        print("Your installation is working correctly!")
        print()
        print("Next steps:")
        print("  1. Run the Jupyter notebook:")
        print("     jupyter notebook test_topology_optimization.ipynb")
        print()
        print("  2. Or start training:")
        print("     python train_enhanced.py --dataset cifar10 --topology-weight 0.01")
        print()
        print("  3. Or run quick start:")
        print("     ./quick_start.sh")
        print()
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print()
        print("Please check the errors above and:")
        print("  1. Ensure all dependencies are installed:")
        print("     pip install -r requirements_enhanced.txt")
        print()
        print("  2. Check your environment:")
        print("     python check_environment.py")
        print()
        print("  3. Review INSTALL.md for troubleshooting")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
