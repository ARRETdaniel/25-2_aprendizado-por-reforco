#!/usr/bin/env python3
"""
Test Suite 1: Docker Infrastructure Testing
Phase 3: System Testing for TD3 Autonomous Navigation System

Tests:
  1.1: Docker Container Launch & GPU Access
  1.2: CARLA Server Launch (will be started separately)
  1.3: Python Dependencies
"""

import subprocess
import sys
import torch
import numpy as np
import time

def test_1_1_docker_gpu_access():
    """Test 1.1: Verify GPU is accessible from PyTorch"""
    print("\n" + "="*70)
    print("🐳 TEST 1.1: Docker Container Launch & GPU Access")
    print("="*70)

    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"✅ CUDA Available: {cuda_available}")
        assert cuda_available, "CUDA not available!"

        # Get device info
        device_count = torch.cuda.device_count()
        print(f"✅ Device Count: {device_count}")
        assert device_count > 0, "No CUDA devices found!"

        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Device Name: {device_name}")

        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ Device Memory: {device_memory:.2f} GB")

        # Test tensor allocation on GPU
        test_tensor = torch.randn(100, 100).cuda()
        print(f"✅ Successfully allocated tensor on GPU: {test_tensor.device}")

        print("\n✅ TEST 1.1 PASSED: Docker container has GPU access\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 1.1 FAILED: {e}\n")
        return False


def test_1_3_python_dependencies():
    """Test 1.3: Verify all required Python packages are installed"""
    print("="*70)
    print("🐍 TEST 1.3: Python Dependencies Verification")
    print("="*70)

    dependencies = {
        'carla': 'CARLA API',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'yaml': 'PyYAML',
        'gymnasium': 'Gymnasium',
        'torchvision': 'TorchVision (for vision transformers)',
        'tensorboard': 'TensorBoard (for logging)',
    }

    all_passed = True

    for package, description in dependencies.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {description:<35} ({package}): {version}")
        except ImportError as e:
            print(f"❌ {description:<35} ({package}): NOT INSTALLED")
            all_passed = False

    if all_passed:
        print("\n✅ TEST 1.3 PASSED: All required dependencies installed\n")
    else:
        print("\n❌ TEST 1.3 FAILED: Some dependencies missing\n")

    return all_passed


def get_python_version():
    """Get Python version info"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def main():
    """Run all Test Suite 1 tests"""

    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "SYSTEM TESTING: Test Suite 1 - Docker Infrastructure" + " "*13 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    # Print system info
    print(f"\n🖥️  System Information:")
    print(f"   Python: {get_python_version()}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"   NumPy: {np.__version__}")

    results = []

    # Test 1.1: GPU Access
    results.append(("Test 1.1: Docker Container & GPU Access", test_1_1_docker_gpu_access()))

    # Test 1.3: Dependencies
    results.append(("Test 1.3: Python Dependencies", test_1_3_python_dependencies()))

    # Note about Test 1.2
    print("="*70)
    print("⚠️  TEST 1.2: CARLA Server Launch")
    print("="*70)
    print("""
Test 1.2 (CARLA Server Launch) must be run separately because it requires:
  1. Starting CARLA in a Docker container (headless mode)
  2. Waiting 15-30 seconds for initialization
  3. Checking if server process is running

This is typically done via: bash CarlaUE4.sh -RenderOffScreen -nosound

Next step: Start CARLA server in a separate terminal/container window,
then run Test Suite 2 (CARLA Server Connectivity) to validate connection.
    """)
    results.append(("Test 1.2: CARLA Server Launch", None))

    # Summary
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*25 + "SUMMARY - TEST SUITE 1" + " "*21 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    for test_name, result in results:
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⏳ MANUAL (see notes above)"

        print(f"{status:<12} {test_name}")

    # Final status
    passed = sum(1 for _, r in results if r is True)
    total = sum(1 for _, r in results if r is not None)

    if total > 0:
        print(f"\nResult: {passed}/{total} automated tests passed")
        print("\n✅ READY FOR TEST SUITE 2: Once CARLA server is running,")
        print("   proceed to CARLA Server Connectivity tests (test_2_carla_connectivity.py)")

    print("\n" + "█"*70 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
