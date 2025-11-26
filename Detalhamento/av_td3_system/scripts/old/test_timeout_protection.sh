#!/bin/bash
# Test Timeout Protection Implementation
#
# Purpose: Validate CARLA tick timeout fix with minimal 1K run
# Expected: Either completes 1K steps OR gracefully handles timeout
#
# Author: Freeze Investigation Team
# Date: 2025-11-18

set -e  # Exit on error

echo "=========================================="
echo "🧪 CARLA Timeout Protection Test"
echo "=========================================="
echo ""
echo "Test Configuration:"
echo "  Max timesteps: 1,000"
echo "  NPCs: 5 (reduced to minimize freeze risk)"
echo "  Scenario: 0 (Town01)"
echo "  Expected duration: ~10 minutes"
echo ""
echo "Success Criteria:"
echo "  ✅ Completes 1,000 steps without freeze"
echo "  ✅ OR gracefully handles timeout with recovery"
echo "  ❌ FAIL: Silent freeze (no log output for >30s)"
echo ""
echo "=========================================="
echo ""

# Ensure CARLA is running
echo "Checking CARLA server..."
if ! pgrep -x "CarlaUE4" > /dev/null; then
    echo "⚠️ CARLA server not detected. Starting CARLA..."
    echo "   (Please start CARLA manually in another terminal if this fails)"
    # Uncomment if you have CARLA start script:
    # cd ~/CARLA_0.9.16 && ./CarlaUE4.sh -RenderOffScreen &
    # sleep 30  # Wait for server to initialize
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../../venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Run minimal test
echo ""
echo "🚀 Starting timeout protection test..."
echo ""

python3 train_td3.py \
    --scenario 0 \
    --max-timesteps 1000 \
    --npcs 5 \
    --batch-size 256 \
    --save-freq 500 \
    --log-freq 10

# Check exit status
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST PASSED: Completed without errors"
    echo ""
    echo "Next steps:"
    echo "  1. Review logs for any timeout warnings"
    echo "  2. Check TensorBoard metrics"
    echo "  3. Proceed to 5K validation if clean"
else
    echo "❌ TEST FAILED: Exit code $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs for timeout errors"
    echo "  2. Verify CARLA server is running"
    echo "  3. Review FREEZE_ROOT_CAUSE_ANALYSIS.md"
fi
echo "=========================================="

exit $EXIT_CODE
