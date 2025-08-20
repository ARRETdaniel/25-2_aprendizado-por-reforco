@echo off
echo ========================================
echo   CARLA DRL Pipeline - First Run Setup
echo ========================================
echo.

REM Check if CARLA is available
set CARLA_PATH=C:\CARLA_0.8.4\CarlaUE4\Binaries\Win64\CarlaUE4.exe
if not exist "%CARLA_PATH%" (
    echo ❌ CARLA not found at %CARLA_PATH%
    echo Please install CARLA 0.8.4 or update the path
    pause
    exit /b 1
)

echo ✅ CARLA found at %CARLA_PATH%

REM Check if virtual environment exists
set VENV_PATH=%~dp0..\carla_py36_env
if not exist "%VENV_PATH%" (
    echo ❌ Python 3.6 virtual environment not found
    echo Please run setup_complete_system.py first
    pause
    exit /b 1
)

echo ✅ Python 3.6 environment found

REM Check WSL2 availability
wsl --status >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ WSL2 not available
    echo Please install and configure WSL2 with Ubuntu
    pause
    exit /b 1
)

echo ✅ WSL2 available

echo.
echo 🚀 Starting CARLA DRL Pipeline...
echo.
echo This will open multiple windows:
echo   1. CARLA Server (3D simulation)
echo   2. CARLA Client (Python 3.6 with sensors)
echo   3. ROS 2 Bridge (WSL2 communication)
echo   4. DRL Training (WSL2 with visualization)
echo   5. TensorBoard (Web dashboard)
echo.

pause

REM Step 1: Start CARLA Server
echo [1/5] Starting CARLA Server...
start "CARLA Server" /D "C:\CARLA_0.8.4" "%CARLA_PATH%" /Game/Maps/Town01 -windowed -carla-server -benchmark -fps=30 -quality-level=Low

echo Waiting for CARLA server to start...
timeout /t 15 /nobreak

REM Step 2: Start CARLA Client
echo [2/5] Starting CARLA Client...
cd /d "%~dp0..\carla_client_py36"
start "CARLA Client" cmd /k ""%VENV_PATH%\Scripts\activate" && python main_enhanced.py --config ..\configs\complete_system_config.yaml --visualize"

echo Waiting for CARLA client to initialize...
timeout /t 10 /nobreak

REM Step 3: Start ROS 2 Bridge in WSL2
echo [3/5] Starting ROS 2 Bridge (WSL2)...
start "ROS 2 Bridge" wsl -d Ubuntu-22.04 -e bash -c "cd '/mnt/c/Users/%USERNAME%/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco' && source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=42 && cd ros2_gateway && source install/setup.bash && ros2 launch carla_bridge carla_bridge.launch.py; exec bash"

echo Waiting for ROS 2 bridge to start...
timeout /t 15 /nobreak

REM Step 4: Start DRL Training in WSL2
echo [4/5] Starting DRL Training (WSL2)...
start "DRL Training" wsl -d Ubuntu-22.04 -e bash -c "cd '/mnt/c/Users/%USERNAME%/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco' && eval \"\$(conda shell.bash hook)\" && conda activate drl_py312 && export ROS_DOMAIN_ID=42 && cd drl_agent && python train_ppo.py --episodes 10 --visualize --config ../configs/complete_system_config.yaml; exec bash"

echo Waiting for DRL training to start...
timeout /t 10 /nobreak

REM Step 5: Start TensorBoard
echo [5/5] Starting TensorBoard...
start "TensorBoard" wsl -d Ubuntu-22.04 -e bash -c "cd '/mnt/c/Users/%USERNAME%/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco' && eval \"\$(conda shell.bash hook)\" && conda activate drl_py312 && tensorboard --logdir monitoring/tensorboard_logs --host 0.0.0.0 --port 6006; exec bash"

echo.
echo ========================================
echo      🎉 CARLA DRL SYSTEM RUNNING!
echo ========================================
echo.
echo 📺 Visual Components Active:
echo   • CARLA 3D Simulation: Town01 environment
echo   • Camera Feeds: RGB + Depth camera windows  
echo   • Training Plots: Real-time reward/loss graphs
echo   • TensorBoard: http://localhost:6006
echo.
echo 📊 System Status:
echo   • CARLA Server: Running on port 2000
echo   • Python 3.6 Client: Sensor data + YOLO detection
echo   • ROS 2 Bridge: Cross-version communication (WSL2)
echo   • PPO Training: Deep Reinforcement Learning (WSL2)
echo   • Monitoring: TensorBoard metrics logging
echo.
echo 🎮 What to Expect:
echo   1. CARLA window shows 3D simulation with red Tesla
echo   2. Camera windows display RGB and depth feeds
echo   3. Training plots show real-time reward progression
echo   4. Console logs show episode progress and metrics
echo   5. TensorBoard shows comprehensive training analytics
echo.
echo 🔧 System Controls:
echo   • View TensorBoard: Open http://localhost:6006 in browser
echo   • Monitor logs: Check individual terminal windows
echo   • Stop training: Ctrl+C in any terminal
echo   • System status: Run health_check.py script
echo.
echo ⚡ Performance Tips:
echo   • Close unnecessary applications for better FPS
echo   • Monitor GPU usage in Task Manager
echo   • Check Windows Defender exclusions for CARLA
echo   • Use Quality Level Low for better performance
echo.
echo 🎯 Training Progress:
echo   • Episodes: 10 (for first demo)
echo   • Expected duration: ~15-20 minutes
echo   • Success metric: Increasing average reward
echo   • Checkpoints: Saved every 5 episodes
echo.
echo 📈 Expected Results:
echo   • Initial episodes: Random exploration (low rewards)
echo   • Learning phase: Gradual reward improvement
echo   • Convergence: Stable driving behavior
echo   • Final: Vehicle learns basic lane keeping
echo.
echo Press any key to close this window...
echo (Keep other windows open for training)
pause >nul
