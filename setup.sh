#!/bin/bash

# Setup script for HRNet Topological Optimization
# This script creates a virtual environment and installs all dependencies

set -e

echo "=========================================="
echo "HRNet Topological Optimization Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"

PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
    echo "❌ Error: Python 3.8 or higher is required"
    echo "   Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version is compatible"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install PyTorch (with appropriate version for your system)
echo "Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [[ $(uname -m) == "arm64" ]]; then
        # Apple Silicon
        echo "  Detected: macOS with Apple Silicon"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    else
        # Intel Mac
        echo "  Detected: macOS with Intel"
        pip install torch torchvision
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo "  Detected: Linux with NVIDIA GPU"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  Detected: Linux without GPU"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
else
    # Default
    echo "  Detected: Other OS, installing default PyTorch"
    pip install torch torchvision
fi
echo "✓ PyTorch installed"
echo ""

# Install core scientific packages
echo "Installing core scientific packages..."
pip install numpy scipy scikit-learn
echo "✓ Core packages installed"
echo ""

# Install visualization packages
echo "Installing visualization packages..."
pip install matplotlib seaborn pandas
echo "✓ Visualization packages installed"
echo ""

# Install utilities
echo "Installing utility packages..."
pip install tqdm pyyaml Pillow tensorboard
echo "✓ Utility packages installed"
echo ""

# Install TDA packages (these may take longer)
echo "Installing Topological Data Analysis packages..."
echo "  (This may take a few minutes...)"

# Install ripser
echo "  Installing ripser..."
pip install --no-cache-dir ripser
if [ $? -eq 0 ]; then
    echo "  ✓ ripser installed"
else
    echo "  ⚠ Warning: ripser installation had issues (will retry)"
    pip install ripser --no-build-isolation
fi

# Install persim
echo "  Installing persim..."
pip install persim
if [ $? -eq 0 ]; then
    echo "  ✓ persim installed"
else
    echo "  ⚠ Warning: persim installation had issues"
fi

echo "✓ TDA packages installation complete"
echo ""

# Install Jupyter (optional but recommended)
echo "Installing Jupyter (for testing notebook)..."
pip install jupyter ipywidgets
echo "✓ Jupyter installed"
echo ""

# Verify installation
echo "Verifying installation..."
python3 check_environment.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ SETUP COMPLETE!"
    echo "=========================================="
    echo ""
    echo "Your environment is ready. You can now:"
    echo ""
    echo "1. Test the setup with the Jupyter notebook:"
    echo "   jupyter notebook test_topology_optimization.ipynb"
    echo ""
    echo "2. Or run a quick training test:"
    echo "   ./quick_start.sh"
    echo ""
    echo "3. Or start full training:"
    echo "   python train_enhanced.py --dataset cifar10 --topology-weight 0.01"
    echo ""
    echo "Note: Remember to activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "⚠ SETUP COMPLETED WITH WARNINGS"
    echo "=========================================="
    echo ""
    echo "Some packages may not have installed correctly."
    echo "Check the output above for details."
    echo ""
    echo "You can try:"
    echo "  1. Running: pip install -r requirements_enhanced.txt"
    echo "  2. Checking: python3 check_environment.py"
    echo ""
fi
