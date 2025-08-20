"""
Configuration models for CARLA DRL Pipeline using Pydantic.

This module provides typed configuration classes that validate and parse
YAML configuration files for the CARLA DRL training pipeline.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, validator
from pathlib import Path
import yaml


class SensorConfig(BaseModel):
    """Configuration for CARLA sensors."""
    enabled: bool = True
    width: int = 800
    height: int = 600
    fov: float = 90.0
    position: List[float] = Field(default=[2.0, 0.0, 1.4], min_items=3, max_items=3)
    rotation: List[float] = Field(default=[0.0, 0.0, 0.0], min_items=3, max_items=3)
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v <= 0 or v > 4096:
            raise ValueError('Image dimensions must be between 1 and 4096')
        return v
    
    @validator('fov')
    def validate_fov(cls, v):
        if v <= 0 or v > 180:
            raise ValueError('FOV must be between 0 and 180 degrees')
        return v


class LidarConfig(BaseModel):
    """Configuration for CARLA LiDAR sensor."""
    enabled: bool = False
    channels: int = 32
    range: float = 50.0
    points_per_second: int = 56000
    rotation_frequency: float = 10.0
    upper_fov_limit: float = 10.0
    lower_fov_limit: float = -30.0
    position: List[float] = Field(default=[0.0, 0.0, 2.5], min_items=3, max_items=3)
    rotation: List[float] = Field(default=[0.0, 0.0, 0.0], min_items=3, max_items=3)


class SensorsConfig(BaseModel):
    """Configuration for all CARLA sensors."""
    camera_rgb: SensorConfig = SensorConfig()
    camera_depth: SensorConfig = SensorConfig()
    lidar: LidarConfig = LidarConfig()


class ServerConfig(BaseModel):
    """Configuration for CARLA server connection."""
    host: str = "localhost"
    port: int = Field(default=2000, ge=1000, le=65535)
    timeout: float = Field(default=10.0, gt=0)
    quality_level: str = Field(default="Low", regex="^(Low|Epic)$")
    synchronous_mode: bool = True
    fixed_delta_seconds: float = Field(default=0.033, gt=0, le=1.0)


class EnvironmentConfig(BaseModel):
    """Configuration for CARLA environment."""
    town: str = Field(default="Town01", regex="^Town0[1-2]$")
    weather_id: int = Field(default=1, ge=1, le=14)
    random_weather: bool = False
    seed_vehicles: int = Field(default=42, ge=0)
    seed_pedestrians: int = Field(default=42, ge=0)
    number_of_vehicles: int = Field(default=15, ge=0, le=100)
    number_of_pedestrians: int = Field(default=10, ge=0, le=100)


class VehicleConfig(BaseModel):
    """Configuration for player vehicle."""
    player_start_index: int = Field(default=0, ge=-1)
    autopilot_enabled: bool = False
    vehicle_type: str = "vehicle.tesla.model3"


class SimulationConfig(BaseModel):
    """Complete simulation configuration."""
    server: ServerConfig = ServerConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    vehicle: VehicleConfig = VehicleConfig()
    sensors: SensorsConfig = SensorsConfig()


class QoSConfig(BaseModel):
    """ROS 2 QoS configuration."""
    reliability: str = Field(default="RELIABLE", regex="^(RELIABLE|BEST_EFFORT)$")
    history: str = Field(default="KEEP_LAST", regex="^(KEEP_LAST|KEEP_ALL)$")
    depth: int = Field(default=10, ge=1)
    durability: str = Field(default="VOLATILE", regex="^(VOLATILE|TRANSIENT_LOCAL)$")


class TopicsConfig(BaseModel):
    """ROS 2 topic configuration."""
    camera_image: str = "/carla/camera/image"
    camera_depth: str = "/carla/camera/depth"
    vehicle_state: str = "/carla/vehicle/state"
    vehicle_pose: str = "/carla/vehicle/pose"
    environment_reward: str = "/carla/environment/reward"
    environment_done: str = "/carla/environment/done"
    environment_info: str = "/carla/environment/info"
    vehicle_control: str = "/carla/vehicle/control"
    environment_reset: str = "/carla/environment/reset"


class CommunicationConfig(BaseModel):
    """ROS 2 communication settings."""
    use_compression: bool = True
    max_message_size: int = Field(default=10485760, gt=0)  # 10MB
    publish_rate: float = Field(default=30.0, gt=0)


class ROS2Config(BaseModel):
    """ROS 2 bridge configuration."""
    node_name: str = "carla_bridge"
    namespace: str = "/carla"
    qos: QoSConfig = QoSConfig()
    topics: TopicsConfig = TopicsConfig()
    communication: CommunicationConfig = CommunicationConfig()


class DevelopmentConfig(BaseModel):
    """Development and debugging configuration."""
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR)$")
    log_to_file: bool = True
    log_directory: str = "./logs"
    display_camera: bool = True
    display_sensors: bool = True
    save_images: bool = False
    image_save_path: str = "./data/images"
    profile_performance: bool = False
    monitor_memory: bool = True
    monitor_fps: bool = True


class RandomSeedsConfig(BaseModel):
    """Random seeds for reproducibility."""
    carla: int = 42
    numpy: int = 42
    python: int = 42


class EpisodeConfig(BaseModel):
    """Episode configuration."""
    max_steps: int = Field(default=1000, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    auto_reset: bool = True


class CarlaSimConfig(BaseModel):
    """Complete CARLA simulation configuration."""
    simulation: SimulationConfig = SimulationConfig()
    ros2: ROS2Config = ROS2Config()
    development: DevelopmentConfig = DevelopmentConfig()
    random_seeds: RandomSeedsConfig = RandomSeedsConfig()
    episode: EpisodeConfig = EpisodeConfig()

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "CarlaSimConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class PPOConfig(BaseModel):
    """PPO algorithm configuration."""
    learning_rate: float = Field(default=3e-4, gt=0)
    n_steps: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=64, gt=0)
    n_epochs: int = Field(default=10, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_range: float = Field(default=0.2, gt=0, le=1)
    clip_range_vf: Optional[float] = None
    normalize_advantage: bool = True
    ent_coef: float = Field(default=0.0, ge=0)
    vf_coef: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    target_kl: Optional[float] = None


class NetworkLayerConfig(BaseModel):
    """Neural network layer configuration."""
    filters: Optional[int] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    activation: str = "relu"
    units: Optional[int] = None


class FeatureExtractorConfig(BaseModel):
    """Feature extractor configuration."""
    type: str = "CNNExtractor"
    cnn_layers: List[NetworkLayerConfig] = Field(default_factory=list)
    flatten_dim: int = 64


class PolicyNetworkConfig(BaseModel):
    """Policy network configuration."""
    type: str = "MultimodalPolicy"
    cnn_features_dim: int = 512
    mlp_hidden_dims: List[int] = Field(default=[256, 256])
    activation: str = "relu"
    dropout_rate: float = Field(default=0.1, ge=0, le=1)
    share_features_extractor: bool = True


class ValueNetworkConfig(BaseModel):
    """Value network configuration."""
    type: str = "MultimodalValue"
    cnn_features_dim: int = 512
    mlp_hidden_dims: List[int] = Field(default=[256, 256])
    activation: str = "relu"
    dropout_rate: float = Field(default=0.1, ge=0, le=1)


class NetworksConfig(BaseModel):
    """Neural networks configuration."""
    policy: PolicyNetworkConfig = PolicyNetworkConfig()
    value: ValueNetworkConfig = ValueNetworkConfig()
    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()


class AlgorithmConfig(BaseModel):
    """Complete algorithm configuration."""
    name: str = "PPO"
    version: str = "2.0"
    ppo: PPOConfig = PPOConfig()
    networks: NetworksConfig = NetworksConfig()


class SpaceConfig(BaseModel):
    """Environment space configuration."""
    type: str
    shape: Optional[List[int]] = None
    dtype: str = "float32"
    low: Optional[Union[float, List[float]]] = None
    high: Optional[Union[float, List[float]]] = None
    spaces: Optional[Dict[str, "SpaceConfig"]] = None


class SpacesConfig(BaseModel):
    """Observation and action spaces configuration."""
    observation: SpaceConfig
    action: SpaceConfig


class RewardComponentConfig(BaseModel):
    """Individual reward component configuration."""
    enabled: bool = True
    weight: float = 1.0


class RewardConfig(BaseModel):
    """Reward function configuration."""
    components: Dict[str, RewardComponentConfig] = Field(default_factory=dict)
    reward_scale: float = 1.0
    normalize_rewards: bool = False


class TrainingConfig(BaseModel):
    """Training configuration."""
    total_timesteps: int = Field(default=1000000, gt=0)
    eval_freq: int = Field(default=10000, gt=0)
    eval_episodes: int = Field(default=5, gt=0)
    save_freq: int = Field(default=25000, gt=0)
    n_envs: int = Field(default=1, gt=0)
    verbose: int = Field(default=1, ge=0, le=2)
    tensorboard_log: str = "./logs/tensorboard"
    log_interval: int = Field(default=1, gt=0)
    save_replay_buffer: bool = False
    checkpoint_dir: str = "./checkpoints"
    model_save_path: str = "./models"
    load_checkpoint: Optional[str] = None


class CarlaTrainConfig(BaseModel):
    """Complete CARLA training configuration."""
    algorithm: AlgorithmConfig = AlgorithmConfig()
    training: TrainingConfig = TrainingConfig()
    spaces: SpacesConfig
    reward: RewardConfig = RewardConfig()

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "CarlaTrainConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# Update forward references
SpaceConfig.model_rebuild()


def load_sim_config(config_path: Union[str, Path]) -> CarlaSimConfig:
    """Load simulation configuration from YAML file."""
    return CarlaSimConfig.from_yaml(config_path)


def load_train_config(config_path: Union[str, Path]) -> CarlaTrainConfig:
    """Load training configuration from YAML file."""
    return CarlaTrainConfig.from_yaml(config_path)
