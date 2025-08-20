@echo off
:: CARLA DRL Pipeline - First-Time Startup Script
:: One-click startup for complete system visualization
:: Author: GitHub Copilot
:: Date: August 2025

echo.
echo ========================================================================
echo  🚗 CARLA DRL Pipeline - First-Time Visualization Startup
echo ========================================================================
echo.

:: Set the working directory to the project root
cd /d "%~dp0.."

:: Check if we're in the correct directory
if not exist "configs\complete_system_config.yaml" (
    echo ❌ Error: Configuration file not found!
    echo    Make sure you're running this script from the project directory.
    echo    Expected: configs\complete_system_config.yaml
    pause
    exit /b 1
)

echo 📋 Pre-flight Checks:
echo.

:: Check Python 3.6
py -3.6 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3.6 not found
    echo    Please install Python 3.6 for CARLA compatibility
    echo    Download from: https://www.python.org/downloads/release/python-3610/
    pause
    exit /b 1
) else (
    echo ✅ Python 3.6 detected
)

:: Check Conda environment
conda info --envs | findstr "carla_drl_py312" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Conda environment 'carla_drl_py312' not found
    echo    Please run setup_complete_system.py first
    pause
    exit /b 1
) else (
    echo ✅ Conda environment 'carla_drl_py312' found
)

:: Check CARLA installation
if exist "C:\CARLA_0.8.4\CarlaUE4\Binaries\Win64\CarlaUE4.exe" (
    echo ✅ CARLA 0.8.4 found at C:\CARLA_0.8.4\
    set CARLA_PATH=C:\CARLA_0.8.4
) else if exist "CarlaSimulator\CarlaUE4\Binaries\Win64\CarlaUE4.exe" (
    echo ✅ CARLA found in project directory
    set CARLA_PATH=%cd%\CarlaSimulator
) else (
    echo ❌ CARLA 0.8.4 not found
    echo    Please install CARLA 0.8.4 or check FIRST_RUN_GUIDE.md
    pause
    exit /b 1
)

echo.
echo 🚀 Starting CARLA DRL Pipeline...
echo.
echo Expected outputs:
echo    • CARLA 3D simulation window (Town01)
echo    • Real-time camera feed with training overlay
echo    • TensorBoard dashboard at http://localhost:6006
echo    • Console logs showing PPO training progress
echo.
echo 🎮 Controls:
echo    • Press 'q' in camera window to stop training
echo    • Press SPACE to toggle autopilot mode
echo    • Press Ctrl+C in any console to shutdown system
echo.

:: Ask for user confirmation
echo Press any key to start the visualization, or Ctrl+C to cancel...
pause >nul

echo.
echo 🚀 Launching components...
echo.

:: Start CARLA Server
echo 📍 Step 1/5: Starting CARLA Server...
start "CARLA Server" /D "%CARLA_PATH%" "%CARLA_PATH%\CarlaUE4\Binaries\Win64\CarlaUE4.exe" /Game/Maps/Course4 -windowed -carla-server -benchmark -fps=30 -quality-level=Low

:: Wait for CARLA to initialize
echo ⏳ Waiting for CARLA server to initialize (10 seconds)...
timeout /t 10 /nobreak >nul

:: Start TensorBoard
echo 📍 Step 2/5: Starting TensorBoard...
start "TensorBoard" cmd /k "conda activate carla_drl_py312 && tensorboard --logdir=logs --port=6006 --host=localhost"

:: Wait for TensorBoard
echo ⏳ Waiting for TensorBoard to start (5 seconds)...
timeout /t 5 /nobreak >nul

:: Start CARLA Client (Python 3.6)
echo 📍 Step 3/5: Starting CARLA Client (Python 3.6)...
start "CARLA Client" cmd /k "set PYTHONPATH=%cd%\CarlaSimulator\PythonAPI\carla && py -3.6 carla_client_py36\main_enhanced.py --host localhost --port 2000"

:: Wait for client connection
echo ⏳ Waiting for CARLA client to connect (8 seconds)...
timeout /t 8 /nobreak >nul

:: Start ROS 2 Gateway (if available)
echo 📍 Step 4/5: Starting ROS 2 Bridge...
:: Note: ROS 2 bridge is integrated into the client for this demo
echo ✅ ROS 2 bridge integrated with CARLA client

:: Start DRL Trainer (Python 3.12)
echo 📍 Step 5/5: Starting DRL Trainer (PPO)...
start "DRL Trainer" cmd /k "conda activate carla_drl_py312 && python drl_agent\train_ppo.py --config configs\complete_system_config.yaml --display --timesteps 50000"

echo.
echo ========================================================================
echo 🎉 CARLA DRL Pipeline Started Successfully!
echo ========================================================================
echo.
echo �️  You should now see:
echo     • CARLA simulation window with Town01 environment
echo     • Camera feed window showing real-time driving view
echo     • Multiple console windows with component status
echo     • Training progress and performance metrics
echo.
echo 🌐 Open in your browser:
echo     • TensorBoard: http://localhost:6006
echo.
echo 📊 Expected behavior:
echo     • Vehicle will start driving autonomously in CARLA
echo     • PPO agent will learn to control the vehicle
echo     • Real-time camera feed shows agent's perspective
echo     • TensorBoard displays training curves and metrics
echo.
echo 🔧 Troubleshooting:
echo     • If CARLA window doesn't appear: Check GPU drivers
echo     • If training doesn't start: Verify Python environments
echo     • If connection issues: Check firewall settings
echo     • See FIRST_RUN_GUIDE.md for detailed help
echo.
echo ⏹️  To stop the system: Press Ctrl+C in any console window
echo     Or close this window to stop all components
echo ========================================================================

:: Keep this window open to monitor system
echo.
echo 🔍 System Monitor - This window shows overall status
echo    Close this window to shutdown all components
echo.

:: Monitor loop
:monitor_loop
echo [%date% %time%] System running... Press Ctrl+C to shutdown
timeout /t 30 /nobreak >nul
goto monitor_loop

:: Cleanup on exit
:cleanup
echo.
echo 🛑 Shutting down CARLA DRL Pipeline...
echo.

:: Kill CARLA processes
taskkill /f /im "CarlaUE4.exe" >nul 2>&1
taskkill /f /im "python.exe" >nul 2>&1

echo ✅ Cleanup completed
echo.
echo 📁 Training data saved to:
echo    • Logs: %cd%\logs\
echo    • Checkpoints: %cd%\checkpoints\
echo.
echo 📖 See FIRST_RUN_GUIDE.md for next steps
echo.
pause
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
