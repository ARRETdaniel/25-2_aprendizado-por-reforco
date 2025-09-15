#!/bin/bash
"""
CARLA Launch Script for RTX 2060 6GB VRAM
Optimized settings following CARLA documentation
"""

echo "🚗 Starting CARLA with RTX 2060 optimizations..."

# CARLA executable path
CARLA_ROOT="/home/danielterra/carla-source"

# Check if CARLA executable exists
if [ ! -f "$CARLA_ROOT/Unreal/CarlaUE4/Binaries/Linux/CarlaUE4Editor" ]; then
    echo "❌ CARLA executable not found at expected location"
    echo "Checking for alternative locations..."
    
    # Try to find CarlaUE4.sh or similar
    CARLA_EXEC=$(find /home/danielterra -name "CarlaUE4.sh" -type f 2>/dev/null | head -1)
    
    if [ -z "$CARLA_EXEC" ]; then
        echo "❌ No CARLA executable found. Please check your CARLA installation."
        echo "💡 You might need to:"
        echo "   1. Complete the CARLA build process"
        echo "   2. Run 'make package' in CARLA source directory"
        echo "   3. Or download pre-built CARLA from GitHub releases"
        exit 1
    else
        echo "✅ Found CARLA executable: $CARLA_EXEC"
        CARLA_ROOT=$(dirname "$(dirname "$CARLA_EXEC")")
    fi
fi

echo "📍 CARLA Root: $CARLA_ROOT"

# GPU Memory Optimization Parameters for RTX 2060 6GB
GPU_OPTS=(
    "-opengl"                    # Use OpenGL instead of Vulkan
    "-quality-level=Low"         # Lowest quality preset
    "-world-port=2000"          # Default port
    "-resx=800"                 # Reduced resolution width
    "-resy=600"                 # Reduced resolution height
    "-windowed"                 # Windowed mode uses less VRAM
    "-carla-rpc-port=2000"      # RPC port
    "-fps=20"                   # Limit FPS to save GPU resources
)

# Memory-specific optimizations
MEMORY_OPTS=(
    "-r.Streaming.PoolSize=512"      # Reduce texture streaming pool (MB)
    "-r.RHI.MaximumFrameLatency=1"   # Reduce frame latency
    "-r.Vulkan.DeviceMemoryBudget=4096"  # Limit Vulkan memory (MB)
    "-malloc=System"                  # Use system malloc instead of jemalloc
)

# Town selection for lower memory usage
LIGHT_TOWNS=(
    "Town01"    # Smallest town
    "Town02"    # Small town
    "Town03"    # Medium town
)

echo "🔧 Applying RTX 2060 optimizations:"
echo "   - OpenGL renderer (lower VRAM usage than Vulkan)"
echo "   - Low quality settings"
echo "   - Reduced resolution (800x600)"
echo "   - Limited texture streaming pool"
echo "   - Recommended towns: ${LIGHT_TOWNS[*]}"

# Create the launch command
LAUNCH_CMD="$CARLA_ROOT/CarlaUE4.sh ${GPU_OPTS[*]} ${MEMORY_OPTS[*]}"

echo "🚀 Launching CARLA with command:"
echo "$LAUNCH_CMD"
echo ""

# Kill any existing CARLA processes
echo "🧹 Cleaning up existing CARLA processes..."
pkill -f CarlaUE4 2>/dev/null || true
pkill -f UE4Editor 2>/dev/null || true
sleep 2

# Set environment variables
export CARLA_ROOT="$CARLA_ROOT"
export UE4_ROOT="$CARLA_ROOT/Unreal"

# Launch CARLA
echo "⏳ Starting CARLA server (this may take 30-60 seconds)..."
echo "📊 Monitor GPU memory usage with: nvidia-smi"
echo "🛑 Stop CARLA with: Ctrl+C or pkill -f CarlaUE4"
echo ""

# Execute the command
cd "$CARLA_ROOT"
exec ./CarlaUE4.sh "${GPU_OPTS[@]}" "${MEMORY_OPTS[@]}"
