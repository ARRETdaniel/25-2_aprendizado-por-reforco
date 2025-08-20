"""
Configuration models using Pydantic for type safety and validation.
"""
from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import numpy as np


class VehicleSpawnConfig(BaseModel):
    """Vehicle spawn configuration."""
    x: float = Field(default=107.0, description="X coordinate")
    y: float = Field(default=133.0, description="Y coordinate") 
    z: float = Field(default=0.5, description="Z coordinate")
    yaw: float = Field(default=0.0, description="Yaw angle in degrees")


class WeatherConfig(BaseModel):
    """Weather simulation parameters."""
    cloudiness: float = Field(default=10.0, ge=0.0, le=100.0)
    precipitation: float = Field(default=0.0, ge=0.0, le=100.0)
    sun_altitude_angle: float = Field(default=45.0, ge=-90.0, le=90.0)
    fog_density: float = Field(default=0.0, ge=0.0, le=100.0)


class SensorConfig(BaseModel):
    """Base sensor configuration."""
    location: List[float] = Field(default=[0.0, 0.0, 0.0], min_items=3, max_items=3)
    rotation: List[float] = Field(default=[0.0, 0.0, 0.0], min_items=3, max_items=3)


class CameraConfig(SensorConfig):
    """Camera sensor specific configuration."""
    width: int = Field(default=84, gt=0)
    height: int = Field(default=84, gt=0)
    fov: float = Field(default=90.0, gt=0.0, le=180.0)
    sensor_tick: float = Field(default=0.1, gt=0.0)


class IMUConfig(SensorConfig):
    """IMU sensor configuration."""
    sensor_tick: float = Field(default=0.02, gt=0.0)


class SensorsConfig(BaseModel):
    """All sensor configurations."""
    camera: CameraConfig = Field(default_factory=CameraConfig)
    imu: IMUConfig = Field(default_factory=IMUConfig)
    collision: SensorConfig = Field(default_factory=SensorConfig)
    lane_invasion: SensorConfig = Field(default_factory=SensorConfig)


class TrafficConfig(BaseModel):
    """Traffic generation parameters."""
    num_vehicles: int = Field(default=20, ge=0)
    num_pedestrians: int = Field(default=10, ge=0)
    vehicle_spawn_radius: float = Field(default=100.0, gt=0.0)
    pedestrian_spawn_radius: float = Field(default=50.0, gt=0.0)


class CommunicationConfig(BaseModel):
    """ZeroMQ communication settings."""
    zmq_port: int = Field(default=5555, gt=1024, lt=65536)
    zmq_bind_address: str = Field(default="tcp://*:5555")
    zmq_connect_address: str = Field(default="tcp://localhost:5555")
    message_rate_hz: float = Field(default=20.0, gt=0.0)


class SimulationConfig(BaseModel):
    """Complete simulation configuration."""
    # Connection settings
    carla_host: str = Field(default="localhost")
    carla_port: int = Field(default=2000, gt=1024, lt=65536)
    timeout: float = Field(default=10.0, gt=0.0)
    sync_mode: bool = Field(default=True)
    fixed_delta_seconds: float = Field(default=0.02, gt=0.0)
    
    # World settings
    town: str = Field(default="Town02", regex=r"^Town\d{2}$")
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    
    # Vehicle settings
    blueprint: str = Field(default="vehicle.tesla.model3")
    spawn_point: VehicleSpawnConfig = Field(default_factory=VehicleSpawnConfig)
    autopilot: bool = Field(default=False)
    
    # Sensors
    sensors: SensorsConfig = Field(default_factory=SensorsConfig)
    
    # Traffic
    traffic: TrafficConfig = Field(default_factory=TrafficConfig)
    
    # Communication
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    
    # Reproducibility
    random_seed: int = Field(default=42, ge=0)
    numpy_seed: int = Field(default=42, ge=0)


class RewardConfig(BaseModel):
    """Reward function configuration."""
    speed_reward_weight: float = Field(default=1.0)
    lane_keeping_weight: float = Field(default=2.0)
    collision_penalty: float = Field(default=-100.0, le=0.0)
    lane_invasion_penalty: float = Field(default=-10.0, le=0.0)
    progress_reward_weight: float = Field(default=5.0)
    comfort_weight: float = Field(default=0.5)


class ActionBounds(BaseModel):
    """Action space bounds."""
    throttle: Tuple[float, float] = Field(default=(0.0, 1.0))
    steering: Tuple[float, float] = Field(default=(-1.0, 1.0))
    
    @validator('throttle', 'steering')
    def validate_bounds(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Lower bound must be less than upper bound")
        return v


class ActionSpaceConfig(BaseModel):
    """Action space configuration."""
    continuous: bool = Field(default=True)
    action_bounds: ActionBounds = Field(default_factory=ActionBounds)


class ObservationSpaceConfig(BaseModel):
    """Observation space configuration."""
    image_size: Tuple[int, int, int] = Field(default=(84, 84, 3))
    normalize_images: bool = Field(default=True)
    include_velocity: bool = Field(default=True)
    include_angular_velocity: bool = Field(default=True)
    include_position: bool = Field(default=False)


class EpisodeConfig(BaseModel):
    """Episode configuration."""
    max_steps: int = Field(default=2000, gt=0)
    timeout_seconds: float = Field(default=100.0, gt=0.0)
    success_distance_threshold: float = Field(default=5.0, gt=0.0)


class EnvironmentConfig(BaseModel):
    """Environment configuration."""
    reward_function: RewardConfig = Field(default_factory=RewardConfig)
    observation_space: ObservationSpaceConfig = Field(default_factory=ObservationSpaceConfig)
    action_space: ActionSpaceConfig = Field(default_factory=ActionSpaceConfig)
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)


class NetworkArchitectureConfig(BaseModel):
    """Neural network architecture configuration."""
    features_extractor: Dict[str, Any] = Field(default={
        "cnn_layers": [[32, 8, 4], [64, 4, 2], [64, 3, 1]],
        "mlp_layers": [512, 512],
        "activation": "relu"
    })
    policy_head: Dict[str, Any] = Field(default={
        "layers": [256],
        "activation": "tanh"
    })
    value_head: Dict[str, Any] = Field(default={
        "layers": [256], 
        "activation": "relu"
    })


class HyperparametersConfig(BaseModel):
    """PPO hyperparameters."""
    learning_rate: float = Field(default=3e-4, gt=0.0)
    n_steps: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=64, gt=0)
    n_epochs: int = Field(default=10, gt=0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_range: float = Field(default=0.2, gt=0.0)
    clip_range_vf: Optional[float] = None
    ent_coef: float = Field(default=0.01, ge=0.0)
    vf_coef: float = Field(default=0.5, ge=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    target_kl: Optional[float] = None


class LoggingConfig(BaseModel):
    """Logging configuration."""
    tensorboard_log: str = Field(default="./logs/tensorboard")
    checkpoint_dir: str = Field(default="./models/checkpoints")
    log_interval: int = Field(default=1, gt=0)
    verbose: int = Field(default=1, ge=0, le=2)
    save_replay_buffer: bool = Field(default=False)


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""
    total_timesteps: int = Field(default=1000000, gt=0)
    eval_freq: int = Field(default=10000, gt=0)
    n_eval_episodes: int = Field(default=5, gt=0)
    eval_deterministic: bool = Field(default=True)
    save_freq: int = Field(default=50000, gt=0)


class AlgorithmConfig(BaseModel):
    """Complete training algorithm configuration."""
    name: str = Field(default="PPO")
    policy: str = Field(default="CnnPolicy")
    hyperparameters: HyperparametersConfig = Field(default_factory=HyperparametersConfig)
    network_architecture: NetworkArchitectureConfig = Field(default_factory=NetworkArchitectureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Reproducibility
    random_seed: int = Field(default=42, ge=0)
    torch_deterministic: bool = Field(default=True)
    cuda_deterministic: bool = Field(default=False)


def load_sim_config(config_path: str) -> SimulationConfig:
    """Load simulation configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return SimulationConfig(**config_dict)


def load_train_config(config_path: str) -> AlgorithmConfig:
    """Load training configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return AlgorithmConfig(**config_dict)
