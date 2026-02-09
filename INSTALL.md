# Installation Guide

This guide will help you set up the HRNet Topological Optimization environment on your system.

## Prerequisites

### Required
- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **pip** (Python package manager)
- **git** (for cloning repository)

### Optional
- **CUDA-capable GPU** (for faster training, but CPU works fine for testing)
- **Jupyter** (for running the test notebook)

## Quick Installation (Recommended)

### Step 1: Clone and Navigate
```bash
cd model_b
```

### Step 2: Run Automated Setup
```bash
./setup.sh
```

This script will:
1. Check your Python version
2. Create a virtual environment
3. Install all required packages
4. Verify the installation

### Step 3: Activate Environment
```bash
source venv/bin/activate
```

### Step 4: Test Installation
```bash
# Option 1: Run environment check
python check_environment.py

# Option 2: Run Jupyter test notebook
jupyter notebook test_topology_optimization.ipynb

# Option 3: Run quick training test
./quick_start.sh
```

## Manual Installation

If the automated setup doesn't work, follow these steps:

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch

**For CPU (works on all systems):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA 11.8 (if you have NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For macOS with Apple Silicon:**
```bash
pip install torch torchvision
```

### 4. Install Core Dependencies
```bash
pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm pyyaml Pillow tensorboard
```

### 5. Install TDA Packages

These packages may require compilation and can take a few minutes:

```bash
# Install ripser (persistent homology computation)
pip install --no-cache-dir ripser

# Install persim (bottleneck distance)
pip install persim
```

**If ripser fails to install:**
```bash
# Try without build isolation
pip install ripser --no-build-isolation

# Or install from conda (if you use conda)
conda install -c conda-forge ripser
```

### 6. Install Jupyter (Optional but Recommended)
```bash
pip install jupyter ipywidgets
```

### 7. Verify Installation
```bash
python check_environment.py
```

## Troubleshooting

### Problem: Python version too old
**Solution:** Install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)

### Problem: ripser installation fails
**Symptoms:** Compilation errors, missing compiler

**Solutions:**
1. **On macOS:** Install Xcode Command Line Tools
   ```bash
   xcode-select --install
   ```

2. **On Linux:** Install build essentials
   ```bash
   sudo apt-get install build-essential python3-dev  # Ubuntu/Debian
   sudo yum install gcc gcc-c++ python3-devel        # CentOS/RHEL
   ```

3. **On Windows:** Install Microsoft Visual C++ Build Tools
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Or use conda: `conda install -c conda-forge ripser`

4. **Alternative:** Use pre-built wheels
   ```bash
   pip install --only-binary :all: ripser
   ```

### Problem: CUDA not available but GPU is present
**Solution:** Reinstall PyTorch with CUDA support
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Memory errors during topology computation
**Solution:** Reduce number of samples
- Edit `topology_analyzer.py`
- Change line: `if features.shape[0] > 500:` to `if features.shape[0] > 200:`

### Problem: Import errors for custom modules
**Solution:** Ensure you're in the correct directory
```bash
cd model_b
python -c "import topology_analyzer; print('OK')"
```

### Problem: Jupyter notebook not connecting
**Solution:** Install kernel for the virtual environment
```bash
pip install ipykernel
python -m ipykernel install --user --name=hrnet-topo
# Then select "hrnet-topo" kernel in Jupyter
```

## Testing Your Installation

### Quick Test (5 minutes)
```bash
# Test imports
python -c "
import torch
import torchvision
from ripser import ripser
from persim import bottleneck
from topology_analyzer import TopologicalAnalyzer
print('✓ All imports successful!')
"
```

### Comprehensive Test (15-30 minutes)
```bash
# Run test notebook
jupyter notebook test_topology_optimization.ipynb
# Execute all cells (Cell → Run All)
```

### Full Training Test (1-2 hours)
```bash
# Run quick start script
./quick_start.sh
```

## Verifying GPU Support

Check if PyTorch can use your GPU:
```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
else:
    print('Using CPU (this is fine for testing)')
"
```

## Package Versions

Here are the tested package versions:

| Package | Minimum | Recommended | Tested |
|---------|---------|-------------|--------|
| Python | 3.8.0 | 3.10.0 | 3.10.12 |
| PyTorch | 2.0.0 | 2.1.0 | 2.1.2 |
| NumPy | 1.21.0 | 1.24.0 | 1.24.3 |
| ripser | 0.6.0 | 0.6.4 | 0.6.4 |
| persim | 0.3.0 | 0.3.1 | 0.3.1 |

## Next Steps

Once installation is complete:

1. **Test the setup:**
   ```bash
   python check_environment.py
   ```

2. **Run the test notebook:**
   ```bash
   jupyter notebook test_topology_optimization.ipynb
   ```

3. **Read the documentation:**
   - `README.md` - Project overview
   - `OPTIMIZATION_GUIDE.md` - Detailed explanation
   - `PROJECT_SUMMARY.md` - Quick reference

4. **Start training:**
   ```bash
   python train_enhanced.py --dataset cifar10 --topology-weight 0.01
   ```

## Getting Help

If you encounter issues:

1. **Check environment:**
   ```bash
   python check_environment.py
   ```

2. **Review error messages** - Most issues are related to:
   - Python version
   - Missing compilers (for ripser)
   - CUDA configuration (for GPU)

3. **Check logs:**
   - Installation logs in terminal output
   - Training logs in `output/*/tensorboard/`

4. **Simplified installation:**
   If all else fails, install without TDA packages and use CPU:
   ```bash
   pip install torch torchvision numpy scipy matplotlib seaborn pandas tqdm
   # The code will work but without topological analysis
   ```

## Alternative: Using Conda

If you prefer conda:

```bash
# Create conda environment
conda create -n hrnet-topo python=3.10
conda activate hrnet-topo

# Install PyTorch
conda install pytorch torchvision -c pytorch

# Install other packages
conda install numpy scipy scikit-learn matplotlib seaborn pandas tqdm
pip install tensorboard pyyaml

# Install TDA packages
conda install -c conda-forge ripser
pip install persim

# Install Jupyter
conda install jupyter
```

## Docker (Advanced)

For a containerized environment (coming soon):
```bash
docker build -t hrnet-topo .
docker run -it hrnet-topo
```

---

**Need more help?** Check the troubleshooting section or open an issue with:
- Your OS and Python version
- Output of `python check_environment.py`
- Full error messages
