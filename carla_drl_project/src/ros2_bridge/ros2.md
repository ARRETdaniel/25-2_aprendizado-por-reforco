CARLA Simulator ←→ CarlaClient ←→ ROS 2 Bridge ←→ TD3 Agent
     ↓                  ↓              ↓            ↓
Vehicle/Sensors → Thread-Safe Data → ROS Messages → State/Actions
Camera Images   → Memory Buffer    → Image Topics → Neural Network
Vehicle State   → Real-time Data   → Odometry     → Reward Function
Control Cmds    ← Action Commands  ← Twist Topic  ← Policy Output
