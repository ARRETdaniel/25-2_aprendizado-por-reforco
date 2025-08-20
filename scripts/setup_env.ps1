# Environment Setup Script for CARLA DRL Pipeline
# This script configures the development environment with proper Python paths and ROS 2 setup

Write-Host "🚀 Setting up CARLA DRL Pipeline Environment" -ForegroundColor Green
Write-Host "=" * 60

# Check if running in elevated mode
$isElevated = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isElevated) {
    Write-Host "⚠️  Note: Some operations may require elevated privileges" -ForegroundColor Yellow
}

# Set workspace root
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
Set-Location $WorkspaceRoot

Write-Host "📁 Workspace: $WorkspaceRoot" -ForegroundColor Cyan

# Environment Variables
Write-Host "`n🔧 Setting Environment Variables" -ForegroundColor Blue

# CARLA Environment
$CarlaPath = "C:\carla"
$CarlaPythonAPI = "$CarlaPath\PythonAPI\carla\dist\carla-0.8.4-py3.6-win-amd64.egg"

if (Test-Path $CarlaPythonAPI) {
    $env:CARLA_ROOT = $CarlaPath
    $env:CARLA_PYTHON_API = $CarlaPythonAPI
    Write-Host "✅ CARLA environment configured: $CarlaPath" -ForegroundColor Green
} else {
    Write-Host "⚠️  CARLA not found at $CarlaPath - manual configuration required" -ForegroundColor Yellow
}

# ROS 2 Environment
$ROS2Path = "C:\opt\ros\humble"
if (Test-Path "$ROS2Path\setup.bat") {
    Write-Host "✅ ROS 2 found: $ROS2Path" -ForegroundColor Green
    & "$ROS2Path\setup.bat"
    $env:ROS_DISTRO = "humble"
} else {
    Write-Host "⚠️  ROS 2 not found at $ROS2Path" -ForegroundColor Yellow
}

# Python Virtual Environment
$VenvPath = "$WorkspaceRoot\venv"
$RequirementsFile = "$WorkspaceRoot\requirements.txt"

Write-Host "`n🐍 Python Environment Setup" -ForegroundColor Blue

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv $VenvPath
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Virtual environment created: $VenvPath" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
$ActivateScript = "$VenvPath\Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    & $ActivateScript
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
    
    # Upgrade pip and install dependencies
    Write-Host "📦 Installing/updating dependencies..." -ForegroundColor Yellow
    
    python -m pip install --upgrade pip
    
    if (Test-Path $RequirementsFile) {
        pip install -r $RequirementsFile
        Write-Host "✅ Dependencies installed from requirements.txt" -ForegroundColor Green
    } else {
        # Install core dependencies manually
        $CorePackages = @(
            "numpy>=1.21.0",
            "opencv-python>=4.5.0",
            "pyzmq>=22.0.0",
            "pyyaml>=6.0",
            "pydantic>=1.8.0",
            "stable-baselines3[extra]>=1.8.0",
            "tensorboard>=2.8.0",
            "matplotlib>=3.5.0",
            "psutil>=5.8.0",
            "GPUtil>=1.4.0"
        )
        
        foreach ($package in $CorePackages) {
            pip install $package
        }
        
        Write-Host "✅ Core dependencies installed" -ForegroundColor Green
    }
    
} else {
    Write-Host "❌ Virtual environment activation script not found" -ForegroundColor Red
}

# Build ROS 2 Packages
Write-Host "`n🔨 Building ROS 2 Packages" -ForegroundColor Blue

$ROS2WorkspacePath = "$WorkspaceRoot\ros2_gateway"
if (Test-Path $ROS2WorkspacePath) {
    Set-Location $ROS2WorkspacePath
    
    Write-Host "Building CARLA Gateway package..." -ForegroundColor Yellow
    colcon build --packages-select carla_gateway
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ ROS 2 packages built successfully" -ForegroundColor Green
        
        # Source the built packages
        if (Test-Path "install\setup.bat") {
            & "install\setup.bat"
            Write-Host "✅ ROS 2 workspace sourced" -ForegroundColor Green
        }
    } else {
        Write-Host "⚠️  ROS 2 build completed with warnings/errors" -ForegroundColor Yellow
    }
    
    Set-Location $WorkspaceRoot
} else {
    Write-Host "⚠️  ROS 2 workspace not found: $ROS2WorkspacePath" -ForegroundColor Yellow
}

# Create necessary directories
Write-Host "`n📁 Creating Directory Structure" -ForegroundColor Blue

$Directories = @(
    "logs",
    "logs\tensorboard", 
    "logs\carla",
    "logs\ros2",
    "models",
    "models\checkpoints",
    "models\best",
    "data",
    "data\recordings",
    "data\episodes",
    "results",
    "results\evaluation",
    "results\plots"
)

foreach ($dir in $Directories) {
    $fullPath = "$WorkspaceRoot\$dir"
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "✅ Created: $dir" -ForegroundColor Green
    }
}

# Set up Git hooks (if .git exists)
if (Test-Path "$WorkspaceRoot\.git") {
    Write-Host "`n📝 Setting up Git hooks" -ForegroundColor Blue
    
    $PreCommitHook = "$WorkspaceRoot\.git\hooks\pre-commit"
    if (-not (Test-Path $PreCommitHook)) {
        @"
#!/bin/sh
# Pre-commit hook for CARLA DRL Pipeline

echo "Running pre-commit checks..."

# Check Python syntax
python -m py_compile carla_client_py36/main.py
python -m py_compile drl_agent/train.py
python -m py_compile drl_agent/infer.py

# Check YAML configurations
python -c "import yaml; yaml.safe_load(open('configs/carla_config.yaml'))"
python -c "import yaml; yaml.safe_load(open('configs/training_config.yaml'))"

echo "Pre-commit checks passed!"
"@ | Out-File -FilePath $PreCommitHook -Encoding UTF8
        Write-Host "✅ Git pre-commit hook installed" -ForegroundColor Green
    }
}

# Configuration validation
Write-Host "`n🔍 Validating Configuration Files" -ForegroundColor Blue

$ConfigFiles = @(
    "configs\carla_config.yaml",
    "configs\training_config.yaml",
    "configs\ros2_config.yaml"
)

foreach ($configFile in $ConfigFiles) {
    $fullPath = "$WorkspaceRoot\$configFile"
    if (Test-Path $fullPath) {
        try {
            python -c "import yaml; yaml.safe_load(open('$fullPath'))"
            Write-Host "✅ Valid: $configFile" -ForegroundColor Green
        } catch {
            Write-Host "❌ Invalid YAML: $configFile" -ForegroundColor Red
        }
    } else {
        Write-Host "⚠️  Missing: $configFile" -ForegroundColor Yellow
    }
}

# Environment Summary
Write-Host "`n📋 Environment Summary" -ForegroundColor Blue
Write-Host "=" * 60

Write-Host "Python Version: $(python --version)" -ForegroundColor Cyan
Write-Host "Workspace: $WorkspaceRoot" -ForegroundColor Cyan
Write-Host "Virtual Environment: $VenvPath" -ForegroundColor Cyan

if ($env:CARLA_ROOT) {
    Write-Host "CARLA Root: $env:CARLA_ROOT" -ForegroundColor Cyan
}

if ($env:ROS_DISTRO) {
    Write-Host "ROS 2 Distro: $env:ROS_DISTRO" -ForegroundColor Cyan
}

# Quick start commands
Write-Host "`n🚀 Quick Start Commands" -ForegroundColor Blue
Write-Host "=" * 60

Write-Host "1. Start CARLA Server:" -ForegroundColor Yellow
Write-Host "   C:\carla\CarlaUE4.exe -carla-server" -ForegroundColor White

Write-Host "`n2. Launch ROS 2 Gateway:" -ForegroundColor Yellow
Write-Host "   cd ros2_gateway && ros2 launch carla_gateway gateway.launch.py" -ForegroundColor White

Write-Host "`n3. Start CARLA Client:" -ForegroundColor Yellow
Write-Host "   cd carla_client_py36 && python main.py" -ForegroundColor White

Write-Host "`n4. Train DRL Agent:" -ForegroundColor Yellow
Write-Host "   cd drl_agent && python train.py --config ../configs/training_config.yaml" -ForegroundColor White

Write-Host "`n5. Run Diagnostics:" -ForegroundColor Yellow
Write-Host "   cd tools && python diagnostics.py --mode all" -ForegroundColor White

Write-Host "`n✅ Environment setup completed!" -ForegroundColor Green
Write-Host "💡 Open VS Code with: code workspace.code-workspace" -ForegroundColor Cyan

# Return to original location
Set-Location $WorkspaceRoot
