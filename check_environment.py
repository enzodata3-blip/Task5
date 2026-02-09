#!/usr/bin/env python3
"""
Environment Compatibility Checker
Verifies all dependencies are correctly installed and compatible
"""

import sys
import importlib
from typing import Tuple, List

def check_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (need >= 3.8)"

def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """Check if package is installed and optionally verify minimum version"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')

        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) >= pkg_version.parse(min_version):
                return True, version
            else:
                return False, f"{version} (need >= {min_version})"
        else:
            return True, version
    except ImportError:
        return False, "NOT INSTALLED"
    except Exception as e:
        return False, f"ERROR: {str(e)}"

def main():
    print("=" * 80)
    print("ENVIRONMENT COMPATIBILITY CHECK")
    print("=" * 80)
    print()

    # Check Python version
    print("Python Version:")
    python_ok, python_version = check_python_version()
    status = "✓" if python_ok else "✗"
    print(f"  {status} {python_version}")
    print()

    # Define required packages with minimum versions
    core_packages = [
        ("torch", "2.0.0"),
        ("torchvision", "0.15.0"),
        ("numpy", "1.21.0"),
        ("scipy", "1.7.0"),
    ]

    visualization_packages = [
        ("matplotlib", "3.5.0"),
        ("seaborn", "0.12.0"),
        ("pandas", "1.3.0"),
    ]

    ml_packages = [
        ("sklearn", "1.0.0"),
        ("tensorboard", None),
    ]

    tda_packages = [
        ("ripser", "0.6.0"),
        ("persim", "0.3.0"),
    ]

    utility_packages = [
        ("tqdm", "4.62.0"),
        ("yaml", "6.0"),
        ("PIL", "9.0.0"),
    ]

    all_categories = [
        ("Core Deep Learning", core_packages),
        ("Visualization", visualization_packages),
        ("Machine Learning", ml_packages),
        ("Topological Data Analysis", tda_packages),
        ("Utilities", utility_packages),
    ]

    all_ok = python_ok
    missing_packages = []
    outdated_packages = []

    for category_name, packages in all_categories:
        print(f"{category_name}:")
        for package_name, min_version in packages:
            package_ok, version_info = check_package(package_name, min_version)
            status = "✓" if package_ok else "✗"
            print(f"  {status} {package_name:20s} {version_info}")

            if not package_ok:
                all_ok = False
                if "NOT INSTALLED" in version_info:
                    missing_packages.append(package_name)
                elif min_version and "need >=" in version_info:
                    outdated_packages.append((package_name, min_version))
        print()

    # Test imports of custom modules
    print("Custom Modules:")
    custom_modules = [
        "topology_analyzer",
        "train_enhanced",
    ]

    for module_name in custom_modules:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name} - {str(e)}")
            all_ok = False
        except Exception as e:
            print(f"  ⚠ {module_name} - Warning: {str(e)}")
    print()

    # Summary
    print("=" * 80)
    if all_ok:
        print("✓ ENVIRONMENT CHECK PASSED")
        print()
        print("All dependencies are correctly installed and compatible!")
        print("You can now run:")
        print("  - jupyter notebook test_topology_optimization.ipynb")
        print("  - python train_enhanced.py --dataset cifar10")
        print("  - ./quick_start.sh")
    else:
        print("✗ ENVIRONMENT CHECK FAILED")
        print()

        if missing_packages:
            print("Missing packages:")
            for pkg in missing_packages:
                print(f"  - {pkg}")
            print()
            print("Install missing packages:")
            print(f"  pip install {' '.join(missing_packages)}")
            print()

        if outdated_packages:
            print("Outdated packages (need upgrade):")
            for pkg, min_ver in outdated_packages:
                print(f"  - {pkg} (need >= {min_ver})")
            print()
            print("Upgrade outdated packages:")
            packages_to_upgrade = [pkg for pkg, _ in outdated_packages]
            print(f"  pip install --upgrade {' '.join(packages_to_upgrade)}")
            print()

        print("Or install all requirements:")
        print("  pip install -r requirements_enhanced.txt")

    print("=" * 80)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
