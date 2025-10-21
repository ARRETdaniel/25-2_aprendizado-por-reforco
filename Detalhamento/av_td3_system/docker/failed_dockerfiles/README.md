# Archived Dockerfiles (Failed Approaches)

## Context
During development (October 19-21, 2025), multiple approaches were attempted to resolve the Python version incompatibility between CARLA 0.9.16's base image (Python 3.8.10) and the CARLA wheel files (Python 3.10-3.12 only).

## Root Cause Analysis

**Problem Discovered:**
```bash
# Base image Python version:
$ docker run --rm carlasim/carla:0.9.16 python3 --version
Python 3.8.10

# Available CARLA wheel files:
carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl  # Requires Python 3.10
carla-0.9.16-cp311-cp311-manylinux_2_31_x86_64.whl  # Requires Python 3.11
carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl  # Requires Python 3.12

# INCOMPATIBILITY: cp38 (base) ≠ cp310/cp311/cp312 (wheels)
```

The official CARLA 0.9.16 documentation claims "Python 3.7-3.12 supported," but the actual Docker image ships with Python 3.8 while only providing wheels for Python 3.10+.

---

## Failed Approaches

### 1. `Dockerfile.ros2-attempt` (October 19, 2025)

**Approach:** Install ROS 2 Foxy on top of CARLA base with `--break-system-packages` flag

**Strategy:**
- Add ROS 2 repository and install ros-foxy-desktop
- Install Python dependencies with `pip3 install --break-system-packages`
- Attempt to use system Python 3.8

**Critical Issues:**
- Still used Python 3.8.10 from base image (incompatible with CARLA wheels)
- `--break-system-packages` is a hacky workaround that breaks package management
- Adds unnecessary complexity with ROS 2 manual installation (CARLA 0.9.16 has built-in ROS 2 support)
- Never successfully completed a build

**Error Log:**
```
ERROR: Could not find a version that satisfies the requirement carla
ERROR: No matching distribution found for carla
```

**Build Status:** ❌ FAILED (never built successfully)

---

### 2. `Dockerfile.carla-0.9.16` (October 20, 2025)

**Approach:** Multi-wheel fallback strategy (try cp310 || cp311 || cp312)

**Strategy:**
```dockerfile
RUN python3 -m pip install /workspace/PythonAPI/carla/dist/carla-0.9.16-cp310-*.whl || \
    python3 -m pip install /workspace/PythonAPI/carla/dist/carla-0.9.16-cp311-*.whl || \
    python3 -m pip install /workspace/PythonAPI/carla/dist/carla-0.9.16-cp312-*.whl
```

**Critical Issues:**
- Logical flaw: Base Python 3.8 cannot install ANY of these wheels (cp310/cp311/cp312)
- The `||` fallback is meaningless when all options require Python 3.10+
- Would always fail on the first attempt and never succeed on subsequent fallbacks

**Error Log:**
```
ERROR: carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl is not a supported wheel on this platform.
ERROR: carla-0.9.16-cp311-cp311-manylinux_2_31_x86_64.whl is not a supported wheel on this platform.
ERROR: carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl is not a supported wheel on this platform.
```

**Additional Issues:**
- Included incorrect verification code:
  ```dockerfile
  RUN python3 -c "import carla; print(f'CARLA API: {carla.__version__}')"
  # This would fail even if import succeeded:
  # AttributeError: module 'carla' has no attribute '__version__'
  ```

**Build Status:** ❌ FAILED (build never completed)

---

## Successful Solution

**File:** `Dockerfile` (formerly `Dockerfile.carla-python310`)

**Approach:** Install Miniforge to provide Python 3.10 environment

**Strategy:**
1. Download Miniforge (community-driven conda distribution)
2. Install Miniforge to `/opt/conda`
3. Create Python 3.10 environment with conda
4. Set environment as default
5. Install CARLA wheel for cp310
6. Install all other dependencies with pip

**Key Advantages:**
- ✅ **Community-driven** - Uses conda-forge (no Anaconda Terms of Service)
- ✅ **Python 3.10 available** - Matches wheel requirements perfectly (cp310)
- ✅ **Ubuntu-independent** - Works on any Linux distribution
- ✅ **Isolated environment** - Doesn't break base system packages
- ✅ **Minimal complexity** - No channel configuration needed
- ✅ **Proven reliable** - conda-forge is widely used and well-maintained

**Build Result:** ✅ SUCCESS (October 21, 2025)

**Final Image:**
- Tag: `td3-av-system:v2.0-python310`
- Size: 30.6GB
- Python: 3.10.19
- All dependencies verified working

**Verification Output:**
```
CARLA API: Successfully imported
Gymnasium: 0.29.1
PyTorch: 2.4.1+cu121, CUDA available: False
OpenCV: 4.8.1
NumPy: 1.24.3
PyYAML: 6.0.1
```

---

## Lessons Learned

1. **Always verify base image capabilities** - Don't trust documentation claims without testing
2. **Check Python version compatibility** - Wheel naming (cp38, cp310, etc.) is critical
3. **Avoid hacky workarounds** - `--break-system-packages` indicates a deeper problem
4. **Consider alternative Python installations** - conda/miniforge can solve version mismatches
5. **Test incrementally** - Don't add complexity (like ROS 2) until core functionality works

---

## Timeline

- **October 19, 2025** - First attempt with ROS 2 installation (failed)
- **October 20, 2025** - Second attempt with multi-wheel fallback (failed)
- **October 20, 2025** - Investigation of deadsnakes PPA (not available for Ubuntu 20.04)
- **October 20, 2025** - Miniconda attempt (failed due to Anaconda TOS)
- **October 20, 2025** - Conda-forge configuration attempt (failed - TOS persists)
- **October 21, 2025** - **Miniforge solution (SUCCESS)** ✅

Total attempts: 6 failed builds → 1 successful solution

---

## References

- CARLA Documentation: https://carla.readthedocs.io/en/latest/build_docker/
- Miniforge GitHub: https://github.com/conda-forge/miniforge
- Build logs archived in parent directory: `build-python310-*.log`
