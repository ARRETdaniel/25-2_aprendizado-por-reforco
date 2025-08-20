carla_drl_pipeline/
├── carla_client_py36/                 # CARLA client (Python 3.6)
│   ├── __init__.py
│   ├── main.py                        # Main CARLA client entry point
│   ├── sensor_manager.py              # Sensor setup and management
│   ├── vehicle_controller.py          # Vehicle control handling
│   ├── communication_bridge.py        # IPC bridge to ROS 2 gateway
│   └── utils.py                       # CARLA utilities
│
├── ros2_gateway/                      # ROS 2 Gateway (C++/Python 3.12)
│   ├── package.xml                    # ROS 2 package manifest
│   ├── CMakeLists.txt                 # Build configuration
│   ├── src/
│   │   ├── carla_bridge_node.cpp      # Main C++ bridge node
│   │   └── message_converters.cpp     # Message type conversions
│   ├── include/carla_bridge/
│   │   ├── carla_bridge_node.hpp
│   │   └── message_converters.hpp
│   └── launch/
│       └── carla_bridge.launch.py     # Launch file
│
├── drl_agent/                         # DRL Training Agent (Python 3.12)
│   ├── __init__.py
│   ├── train.py                       # PPO training script
│   ├── infer.py                       # Inference/evaluation script
│   ├── ppo_algorithm.py               # PPO implementation
│   ├── networks.py                    # Neural network architectures
│   ├── environment_wrapper.py         # ROS 2 environment wrapper
│   └── feature_extractors.py          # CNN feature extraction
│
├── configs/                           # Configuration files
│   ├── sim.yaml                       # Simulation configuration
│   ├── train.yaml                     # Training configuration
│   ├── scenarios/                     # Scenario definitions
│   │   ├── town01_clear.yaml
│   │   ├── town01_rain.yaml
│   │   └── town02_sunset.yaml
│   └── models/                        # Model configurations
│       ├── ppo_config.yaml
│       └── network_config.yaml
│
├── scripts/                           # Deployment and utility scripts
│   ├── setup_environment.py          # Environment setup
│   ├── launch_pipeline.py             # Complete pipeline launcher
│   ├── docker/                        # Docker configurations
│   │   ├── Dockerfile.carla
│   │   ├── Dockerfile.ros2
│   │   └── docker-compose.yml
│   └── windows/                       # Windows-specific scripts
│       ├── setup_wsl2.ps1
│       └── launch_carla.bat
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_carla_client.py          # CARLA client tests
│   ├── test_ros2_bridge.py           # ROS 2 bridge tests
│   ├── test_ppo_agent.py             # PPO algorithm tests
│   ├── integration/                   # Integration tests
│   │   ├── test_full_pipeline.py
│   │   └── test_scenarios.py
│   └── fixtures/                      # Test data and mocks
│       ├── mock_carla.py
│       └── sample_data/
│
├── docs/                              # Documentation
│   ├── README.md                      # Main documentation
│   ├── SETUP.md                       # Setup instructions
│   ├── API.md                         # API reference
│   └── TROUBLESHOOTING.md             # Common issues and solutions
│
├── requirements/                      # Dependency specifications
│   ├── carla_py36.txt                # Python 3.6 requirements
│   ├── drl_py312.txt                 # Python 3.12 requirements
│   └── ros2_deps.yaml                # ROS 2 dependencies
│
└── workspace_setup/                   # Development environment
    ├── .vscode/
    │   ├── launch.json                # VS Code debug configuration
    │   ├── tasks.json                 # Build and run tasks
    │   └── settings.json              # Project settings
    ├── .gitignore
    ├── pyproject.toml                 # Python project configuration
    └── colcon_ws/                     # ROS 2 workspace
        ├── src/
        ├── build/
        ├── install/
        └── log/
