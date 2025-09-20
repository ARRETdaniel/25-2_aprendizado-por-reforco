# CARLA Deep Reinforcement Learning Project

## 🎯 Project Overview

This project implements a Deep Reinforcement Learning pipeline for autonomous **truck navigation** using **CARLA 0.9.16** simulator and **ROS 2 Foxy**. The system is specifically designed to train a **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** agent for continuous vehicle control tasks.

## ✅ Current Implementation Status

**CARLA Client Implementation: COMPLETE** ✅
- ✅ CARLA 0.9.16 server connection established
- ✅ Vehicle spawning (truck: `vehicle.carlamotors.carlacola`)
- ✅ Real-time camera visualization (640x480 RGB + 320x240 depth)
- ✅ Memory-optimized sensor management
- ✅ Town01 map compatibility
- ✅ CV2 visualization with user controls
- ✅ Graceful resource cleanup
- ✅ Thread-safe data streaming
- ✅ Performance monitoring and validation

## 🏗️ System Architecture

```
CARLA Server (0.9.16) ←→ CarlaClient (Python API) ←→ [Future] ROS 2 Bridge ←→ [Future] TD3 Agent
     ↓                           ↓                              ↓                      ↓
   Town01              Camera/Sensor Data               geometry_msgs            Action Commands
   Vehicle              Real-time Streaming              sensor_msgs               State/Rewards
   Physics              Memory Management                std_msgs                  Training Loop
```
 Simulation              Sensors/Control (Modules)         Topics/Msgs    DRL Training

```
CARLA 0.9.16 ←→ Python API ←→ ROS 2 Bridge ←→ Gymnasium Environment ←→ TD3 Agent
```

### Core Components

- **CARLA Interface**: Direct Python API for vehicle and sensor management
- **ROS 2 Bridge**: Sensor data publishing and control commands (17 packages available)
- **Gymnasium Environment**: RL-compliant environment wrapper with custom reward functions
- **TD3 Algorithm**: Twin Delayed DDPG using Stable-Baselines3
- **Training Infrastructure**: Monitoring, logging, and model persistence

## ✅ System Status: **READY FOR DEVELOPMENT**

### Verified Dependencies
- **✅ PyTorch 2.4.1+cu118**: CUDA support for RTX 2060
- **✅ Stable-Baselines3 2.4.1**: TD3/SAC/DDPG algorithms
- **✅ CARLA 0.9.16**: Python API operational
- **✅ ROS 2 Foxy**: 17 CARLA packages built and available
- **✅ GPU**: RTX 2060 (5.6GB VRAM) with CUDA 11.8

## 🚀 Quick Start

### Environment Setup
```bash
# Source ROS 2 environment
source /opt/ros/foxy/setup.bash
source ~/carla-ros-bridge/install/setup.bash

# Set CARLA Python path (adjust path as needed)
export CARLA_ROOT=~/carla-0.9.16
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux-x86_64.egg:$CARLA_ROOT/PythonAPI/carla
```

### Running the System
```bash
# Start development
python scripts/start_development.py
```

## 📁 Project Structure

```
carla_drl_project/
├── src/                              # Source code modules
│   ├── carla_interface/              # CARLA Python API integration
│   ├── ros2_bridge/                  # ROS 2 sensor/control bridge
│   ├── environment/                  # Gymnasium RL environment
│   ├── algorithms/                   # TD3 and utility algorithms
│   └── utils/                        # Common utilities
├── config/                           # YAML configuration files
├── scripts/                          # Training and evaluation scripts
├── data/                             # Models, logs, and episodes
└── tests/                            # Unit and integration tests
```

## ⚙️ Memory Optimization (RTX 2060)

Due to 6GB VRAM limitation, the system uses:

- **Headless rendering**: `-RenderOffScreen` flag
- **Low quality settings**: `-quality-level=Low`
- **Optimized image resolution**: 320x240 (configurable)
- **Memory monitoring**: Automatic restart on overflow
- **Batch size limitation**: Max 32 samples

## 🧪 Testing System

Verify everything works:

```bash
# Test all imports
python -c "import carla, torch, stable_baselines3, rclpy; print('✅ All dependencies OK')"

# Test CUDA
python -c "import torch; print(f'✅ CUDA: {torch.cuda.is_available()}')"

# Test ROS packages
ros2 pkg list | grep carla
```

## 🎯 Development Roadmap

### **Phase 1: Core Implementation (Current)**
- [ ] CARLA client interface (`CarlaClient`)
- [ ] ROS 2 sensor publishers (`SensorPublisher`)
- [ ] Basic Gymnasium environment (`CarlaGymEnv`)
- [ ] Integration testing

### **Phase 2: DRL Training**
- [ ] TD3 agent implementation
- [ ] Reward function optimization
- [ ] Memory-efficient training pipeline
- [ ] Performance validation

### **Phase 3: Advanced Features**
- [ ] Multi-agent scenarios
- [ ] Weather/lighting variations
- [ ] Real-world transfer preparation

## 🔧 Key Configuration Files

- `config/carla_settings.yaml`: CARLA server and world parameters
- `config/training_config.yaml`: TD3 hyperparameters and training settings

## 📊 Hardware Requirements

- **OS**: Ubuntu 20.04 LTS (verified)
- **Python**: 3.8.10 (verified)
- **GPU**: NVIDIA RTX 2060 6GB (configured)
- **RAM**: 16GB recommended
- **Storage**: 50GB for models and logs

## 🐛 Common Issues & Solutions

- **CARLA server crash**: Use headless mode and low quality
- **ROS topics missing**: Ensure bridge is running and sourced
- **GPU OOM**: Reduce batch size or image resolution
- **Import errors**: Check Python path and virtual environment

## 📧 Project Context

**Objective**: Train a truck to navigate autonomously in CARLA using TD3
**Supervisor**: [Add supervisor name]
**University**: [Add university name]
**Course**: Deep Reinforcement Learning + CARLA + ROS 2

---

**Status**: ✅ Setup Complete - Ready for Core Implementation
**Next Step**: Implement CARLA client interface (see NEXT_STEPS_ANALYSIS.md)
```

### Basic Usage

1. **Train a TD3 agent:**
```bash
python scripts/train_agent.py --algorithm td3 --episodes 10000
```

2. **Evaluate trained agent:**
```bash
python scripts/evaluate_agent.py --model data/models/td3_best.zip
```

3. **Record episodes:**
```bash
python scripts/record_episodes.py --agent random --episodes 10
```

## 📁 Project Structure

```
carla_drl_project/
├── src/                              # Source code
│   ├── carla_interface/              # CARLA communication
│   ├── ros2_bridge/                  # ROS 2 integration
│   ├── environment/                  # Gym environment
│   ├── algorithms/                   # DRL implementations
│   └── utils/                        # Common utilities
├── config/                           # Configuration files
├── scripts/                          # Executable scripts
├── tests/                            # Unit and integration tests
├── data/                             # Data storage
│   ├── models/                       # Trained models
│   ├── logs/                         # Training logs
│   └── episodes/                     # Recorded episodes
└── docs/                             # Documentation
```

## ⚙️ Configuration

All system parameters are configured via YAML files in the `config/` directory:

- `carla_settings.yaml`: CARLA server parameters
- `ros2_params.yaml`: ROS 2 node configurations
- `training_config.yaml`: DRL hyperparameters
- `logging_config.yaml`: Logging settings

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Monitoring and Visualization

Training progress can be monitored using:

- **TensorBoard**: `tensorboard --logdir data/logs`
- **Weights & Biases**: Configure API key in config
- **MLflow**: For model versioning and comparison

## 🔧 Development

### Code Quality

The project uses automated code formatting and linting:

```bash
# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Lint code
flake8 src/ tests/ scripts/
mypy src/
```

### Adding New Algorithms

1. Create algorithm file in `src/algorithms/`
2. Implement required interface methods
3. Add configuration to `training_config.yaml`
4. Add tests in `tests/test_algorithms.py`

## 🐛 Troubleshooting

### Common Issues

1. **CARLA crashes with SIGSEGV**: Reduce quality settings or use headless mode
2. **ROS 2 topics not found**: Ensure `carla_ros_bridge` is running
3. **GPU memory errors**: Reduce batch size in training config
4. **Import errors**: Verify all dependencies are installed

See `docs/troubleshooting.md` for detailed solutions.

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Training Guide](docs/training_guide.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run code quality checks
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CARLA team for the excellent simulator
- ROS 2 community for the middleware framework
- Stable-Baselines3 team for the DRL implementations
- OpenAI for the Gym interface standard

## 📧 Contact

For questions and support, please open an issue or contact [your.email@example.com].

---

**Project Status**: 🚧 Under Development
**Current Version**: 0.1.0
**Last Updated**: September 2025
