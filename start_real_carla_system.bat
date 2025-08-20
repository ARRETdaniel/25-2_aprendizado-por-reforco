@echo off
rem Complete CARLA-DRL Integration System Startup
rem This script starts the full pipeline: CARLA -> ZMQ Bridge -> DRL Training

title Real CARLA DRL Integration System

echo.
echo ===============================================
echo    CARLA-DRL Integration System v1.0
echo ===============================================
echo.
echo This system coordinates:
echo - CARLA 0.8.4 Simulation (Python 3.6)
echo - ZMQ Communication Bridge
echo - Real-time DRL Training (Python 3.12)
echo.

rem Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Conda not found! Please install Anaconda/Miniconda
    pause
    exit /b 1
)

rem Check CARLA directory
set CARLA_DIR=%~dp0CarlaSimulator
if not exist "%CARLA_DIR%" (
    echo ❌ CARLA directory not found: %CARLA_DIR%
    echo Please ensure CARLA 0.8.4 is extracted in the workspace
    pause
    exit /b 1
)

echo ✅ System requirements check passed
echo.

rem Menu system
:menu
echo Choose operation mode:
echo.
echo 1. 🚀 Full System Start (CARLA + DRL Training)
echo 2. 🏁 CARLA Client Only (Python 3.6)
echo 3. 🧠 DRL Training Only (Python 3.12)
echo 4. 🔧 System Health Check
echo 5. 📊 View Logs
echo 6. ❌ Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto full_start
if "%choice%"=="2" goto carla_only
if "%choice%"=="3" goto drl_only
if "%choice%"=="4" goto health_check
if "%choice%"=="5" goto view_logs
if "%choice%"=="6" goto exit

echo Invalid choice. Please try again.
goto menu

:full_start
echo.
echo 🚀 Starting Full CARLA-DRL Integration System
echo ============================================
echo.

rem Check if CARLA server is running
echo 🔍 Checking for CARLA server...
netstat -an | findstr "2000" >nul
if %errorlevel% neq 0 (
    echo ⚠️ CARLA server not detected on port 2000
    echo Please start CARLA server first:
    echo   cd %CARLA_DIR%
    echo   CarlaUE4.exe -carla-server -windowed -ResX=800 -ResY=600
    echo.
    echo Start CARLA server now? (y/n)
    set /p start_carla="Start CARLA server? (y/n): "
    if /i "%start_carla%"=="y" (
        echo Starting CARLA server...
        start "CARLA Server" /D "%CARLA_DIR%" CarlaUE4.exe -carla-server -windowed -ResX=800 -ResY=600 -quality-level=Low
        echo Waiting 10 seconds for CARLA to start...
        timeout /t 10 /nobreak >nul
    ) else (
        echo Please start CARLA server manually and try again
        pause
        goto menu
    )
)

echo ✅ CARLA server ready
echo.

rem Start CARLA client in background
echo 🔗 Starting CARLA ZMQ Client (Python 3.6)...
start "CARLA ZMQ Client" cmd /k "cd /d %~dp0carla_client_py36 && py -3.6 carla_zmq_client.py"

echo Waiting 5 seconds for client to connect...
timeout /t 5 /nobreak >nul

rem Start DRL training
echo 🧠 Starting DRL Training (Python 3.12)...
cd /d "%~dp0drl_agent"
call conda activate carla_drl_py312

echo.
echo 🎯 Starting Real CARLA PPO Training...
echo Press Ctrl+C to stop training gracefully
echo.
python real_carla_ppo_trainer.py

goto end

:carla_only
echo.
echo 🏁 Starting CARLA Client Only
echo =============================
echo.

rem Check CARLA server
netstat -an | findstr "2000" >nul
if %errorlevel% neq 0 (
    echo ❌ CARLA server not running on port 2000
    echo Please start CARLA server first
    pause
    goto menu
)

echo Starting CARLA ZMQ Client...
cd /d "%~dp0carla_client_py36"
py -3.6 carla_zmq_client.py
goto end

:drl_only
echo.
echo 🧠 Starting DRL Training Only
echo =============================
echo.

echo Activating Python 3.12 environment...
call conda activate carla_drl_py312

cd /d "%~dp0drl_agent"
echo.
echo Starting DRL training (will use synthetic data if CARLA not connected)...
python real_carla_ppo_trainer.py
goto end

:health_check
echo.
echo 🔧 System Health Check
echo ======================
echo.

rem Check conda environments
echo Checking conda environments...
conda env list | findstr "carla_drl_py312" >nul
if %errorlevel% neq 0 (
    echo ❌ carla_drl_py312 environment not found
) else (
    echo ✅ carla_drl_py312 environment found
)

rem Check Python versions
echo.
echo Checking Python versions...
py -3.6 --version 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python 3.6 not available
) else (
    echo ✅ Python 3.6 available
)

call conda activate carla_drl_py312 >nul 2>&1
python --version 2>nul | findstr "3.12" >nul
if %errorlevel% neq 0 (
    echo ❌ Python 3.12 not available in carla_drl_py312
) else (
    echo ✅ Python 3.12 available in carla_drl_py312
)

rem Check CARLA
echo.
echo Checking CARLA installation...
if exist "%CARLA_DIR%\CarlaUE4.exe" (
    echo ✅ CARLA executable found
) else (
    echo ❌ CARLA executable not found
)

if exist "%CARLA_DIR%\PythonAPI\carla\dist\carla-0.8.4-py3.6-win-amd64.egg" (
    echo ✅ CARLA Python API found
) else (
    echo ❌ CARLA Python API not found
)

rem Check key files
echo.
echo Checking key system files...
if exist "%~dp0carla_client_py36\carla_zmq_client.py" (
    echo ✅ CARLA ZMQ client found
) else (
    echo ❌ CARLA ZMQ client missing
)

if exist "%~dp0communication\zmq_bridge.py" (
    echo ✅ ZMQ bridge found
) else (
    echo ❌ ZMQ bridge missing
)

if exist "%~dp0drl_agent\real_carla_ppo_trainer.py" (
    echo ✅ DRL trainer found
) else (
    echo ❌ DRL trainer missing
)

rem Check network ports
echo.
echo Checking network ports...
netstat -an | findstr "2000" >nul
if %errorlevel% neq 0 (
    echo ⚠️ CARLA server port 2000 not in use
) else (
    echo ✅ CARLA server running on port 2000
)

netstat -an | findstr "5556" >nul
if %errorlevel% neq 0 (
    echo ⚠️ ZMQ bridge port 5556 not in use
) else (
    echo ✅ ZMQ bridge running on port 5556
)

echo.
echo Health check complete!
pause
goto menu

:view_logs
echo.
echo 📊 Viewing Training Logs
echo =======================
echo.

if exist "%~dp0drl_agent\logs" (
    echo Available log directories:
    dir "%~dp0drl_agent\logs" /b
    echo.
    echo To view TensorBoard logs:
    echo   cd drl_agent\logs
    echo   tensorboard --logdir .
    echo.
    echo Open logs folder? (y/n)
    set /p open_logs="Open logs folder? (y/n): "
    if /i "%open_logs%"=="y" (
        start "" "%~dp0drl_agent\logs"
    )
) else (
    echo No logs found. Run training first.
)

pause
goto menu

:exit
echo.
echo Exiting CARLA-DRL Integration System
echo.
exit /b 0

:end
echo.
echo Press any key to return to menu...
pause >nul
goto menu
