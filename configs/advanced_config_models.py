"""
Advanced configuration models for CARLA DRL Pipeline.
Extends base configuration with production features like curriculum learning,
multi-agent training, and monitoring integration.
"""

from typing import Dict, List, Optional, Union, Any, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import yaml
from pathlib import Path
import numpy as np

class AdvancedCarlaConfig(BaseModel):
    """Enhanced CARLA configuration with advanced features."""
    
    # Base CARLA settings (inherited from existing config)
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    
    # Enhanced simulation settings
    quality_level: Literal["Low", "Medium", "High", "Epic"] = "Medium"
    no_rendering_mode: bool = False
    offscreen_rendering: bool = False
    
    # Multi-scenario configuration
    towns: List[str] = Field(default=["Town01", "Town02", "Town03"])
    current_town: str = "Town01"
    auto_town_rotation: bool = False
    town_rotation_episodes: int = 100
    
    # Advanced weather system
    weather_presets: List[Dict[str, float]] = Field(default_factory=lambda: [
        {"cloudiness": 0.0, "precipitation": 0.0, "sun_altitude": 70.0, "fog_density": 0.0},
        {"cloudiness": 50.0, "precipitation": 0.0, "sun_altitude": 45.0, "fog_density": 0.0},
        {"cloudiness": 80.0, "precipitation": 30.0, "sun_altitude": 30.0, "fog_density": 5.0},
        {"cloudiness": 90.0, "precipitation": 60.0, "sun_altitude": 15.0, "fog_density": 15.0}
    ])
    dynamic_weather: bool = False
    weather_change_episodes: int = 50
    
    # Traffic configuration
    traffic_config: Dict[str, Any] = Field(default_factory=lambda: {
        "num_vehicles": 20,
        "num_pedestrians": 10,
        "vehicle_spawn_radius": 200.0,
        "pedestrian_spawn_radius": 100.0,
        "aggressive_driving_ratio": 0.1,
        "traffic_light_compliance": 0.9
    })
    
    # Sensor configuration
    sensors: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "rgb_camera": {
            "enabled": True,
            "image_size_x": 800,
            "image_size_y": 600,
            "fov": 90,
            "location": [0.3, 0.0, 1.3],
            "rotation": [0.0, 0.0, 0.0]
        },
        "depth_camera": {
            "enabled": True,
            "image_size_x": 800,
            "image_size_y": 600,
            "fov": 90,
            "location": [0.3, 0.0, 1.3],
            "rotation": [0.0, 0.0, 0.0]
        },
        "semantic_camera": {
            "enabled": False,
            "image_size_x": 800,
            "image_size_y": 600,
            "fov": 90,
            "location": [0.3, 0.0, 1.3],
            "rotation": [0.0, 0.0, 0.0]
        },
        "lidar": {
            "enabled": False,
            "channels": 32,
            "range": 50.0,
            "points_per_second": 100000,
            "rotation_frequency": 10,
            "location": [0.0, 0.0, 2.5]
        },
        "imu": {
            "enabled": True,
            "location": [0.0, 0.0, 0.0]
        },
        "gnss": {
            "enabled": True,
            "location": [0.0, 0.0, 0.0]
        }
    })
    
    # Performance monitoring
    monitoring: Dict[str, Any] = Field(default_factory=lambda: {
        "enable_metrics": True,
        "metrics_port": 8080,
        "log_sensor_data": True,
        "log_vehicle_state": True,
        "frame_rate_target": 20.0,
        "performance_warnings": True
    })

class CurriculumConfig(BaseModel):
    """Configuration for curriculum learning."""
    
    enabled: bool = True
    
    # Curriculum stages
    stages: List[Dict[str, Any]] = Field(default_factory=lambda: [
        {
            "name": "basic_lane_following",
            "episodes": 500,
            "town": "Town01",
            "weather_preset": 0,
            "traffic_density": 0.1,
            "max_speed": 30.0,
            "success_threshold": 0.8,
            "objectives": ["lane_keeping", "speed_control"]
        },
        {
            "name": "urban_navigation",
            "episodes": 1000,
            "town": "Town02", 
            "weather_preset": 1,
            "traffic_density": 0.3,
            "max_speed": 50.0,
            "success_threshold": 0.7,
            "objectives": ["intersection_navigation", "traffic_light_compliance"]
        },
        {
            "name": "complex_scenarios",
            "episodes": 1500,
            "town": "Town03",
            "weather_preset": 2,
            "traffic_density": 0.5,
            "max_speed": 60.0,
            "success_threshold": 0.6,
            "objectives": ["obstacle_avoidance", "emergency_braking", "lane_changes"]
        },
        {
            "name": "adverse_conditions",
            "episodes": 2000,
            "town": "random",
            "weather_preset": "random",
            "traffic_density": 0.7,
            "max_speed": 70.0,
            "success_threshold": 0.5,
            "objectives": ["all_weather_driving", "high_traffic_navigation"]
        }
    ])
    
    # Adaptation settings
    auto_progression: bool = True
    progression_threshold: float = 0.8
    regression_threshold: float = 0.3
    stage_evaluation_episodes: int = 50
    
    # Dynamic difficulty adjustment
    dynamic_difficulty: bool = True
    difficulty_window: int = 100  # Episodes to consider for difficulty adjustment
    difficulty_adaptation_rate: float = 0.1

class AdvancedTrainingConfig(BaseModel):
    """Enhanced training configuration with advanced features."""
    
    # Base training settings
    algorithm: Literal["PPO", "SAC", "TD3", "A2C"] = "PPO"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    
    # PPO specific settings
    ppo: Dict[str, Any] = Field(default_factory=lambda: {
        "n_steps": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "use_sde": False,
        "sde_sample_freq": -1
    })
    
    # SAC specific settings  
    sac: Dict[str, Any] = Field(default_factory=lambda: {
        "buffer_size": 1_000_000,
        "learning_starts": 10000,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_entropy": "auto",
        "use_sde": False,
        "sde_sample_freq": -1
    })
    
    # Multi-agent training
    multi_agent: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,
        "num_agents": 1,
        "coordination_reward": 0.1,
        "competition_reward": -0.05,
        "shared_experience": True,
        "central_critic": False
    })
    
    # Experience replay enhancements
    replay_buffer: Dict[str, Any] = Field(default_factory=lambda: {
        "prioritized": True,
        "alpha": 0.6,
        "beta": 0.4,
        "beta_annealing_steps": 100000,
        "epsilon": 1e-6,
        "n_step": 3,
        "hindsight_experience_replay": False
    })
    
    # Network architecture
    policy_network: Dict[str, Any] = Field(default_factory=lambda: {
        "net_arch": [256, 256],
        "activation_fn": "relu",
        "ortho_init": True,
        "use_expln": False,
        "log_std_init": 0.0,
        "full_std": True,
        "use_sde": False,
        "squash_output": False
    })
    
    # Feature extraction
    features_extractor: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "multimodal_cnn",
        "cnn_features_dim": 512,
        "normalize_images": True,
        "attention_mechanism": True,
        "temporal_context": 4,
        "feature_fusion": "concatenate"  # "concatenate", "attention", "gating"
    })
    
    # Curriculum learning integration
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    
    # Regularization and stability
    regularization: Dict[str, Any] = Field(default_factory=lambda: {
        "weight_decay": 1e-5,
        "dropout_rate": 0.1,
        "batch_norm": False,
        "layer_norm": True,
        "spectral_norm": False,
        "gradient_clipping": True,
        "grad_clip_value": 0.5
    })
    
    # Training monitoring
    logging: Dict[str, Any] = Field(default_factory=lambda: {
        "tensorboard": True,
        "wandb": False,
        "mlflow": False,
        "log_interval": 10,
        "eval_interval": 1000,
        "save_interval": 5000,
        "video_interval": 10000,
        "detailed_logging": True
    })
    
    # Checkpointing and model management
    checkpointing: Dict[str, Any] = Field(default_factory=lambda: {
        "save_best_model": True,
        "save_last_model": True,
        "save_replay_buffer": True,
        "checkpoint_frequency": 10000,
        "max_checkpoints": 5,
        "model_versioning": True
    })

class MonitoringConfig(BaseModel):
    """Configuration for system monitoring and observability."""
    
    # Prometheus metrics
    prometheus: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "port": 8080,
        "metrics_prefix": "carla_drl",
        "scrape_interval": "5s",
        "custom_metrics": [
            "episode_reward",
            "collision_rate", 
            "lane_invasion_rate",
            "success_rate",
            "average_speed",
            "training_loss",
            "policy_entropy"
        ]
    })
    
    # Grafana dashboards
    grafana: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "port": 3000,
        "auto_dashboards": True,
        "alert_channels": ["slack", "email"],
        "dashboard_refresh": "5s"
    })
    
    # Health monitoring
    health_checks: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "check_interval": 10.0,
        "timeout": 5.0,
        "checks": [
            "carla_server_connection",
            "ros2_node_status", 
            "drl_training_progress",
            "system_resources",
            "gpu_utilization"
        ]
    })
    
    # Alerting
    alerting: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "channels": {
            "slack": {
                "webhook_url": "",
                "channel": "#carla-drl-alerts"
            },
            "email": {
                "smtp_server": "",
                "recipients": []
            }
        },
        "alert_rules": [
            {
                "name": "high_collision_rate",
                "condition": "collision_rate > 0.1",
                "severity": "warning",
                "description": "Collision rate is above 10%"
            },
            {
                "name": "training_stalled",
                "condition": "training_progress_stalled > 3600",
                "severity": "critical", 
                "description": "Training has not progressed for 1 hour"
            },
            {
                "name": "low_gpu_utilization",
                "condition": "gpu_utilization < 0.3",
                "severity": "info",
                "description": "GPU utilization is below 30%"
            }
        ]
    })

class AdvancedPipelineConfig(BaseModel):
    """Master configuration combining all components."""
    
    # Component configurations
    carla: AdvancedCarlaConfig = Field(default_factory=AdvancedCarlaConfig)
    training: AdvancedTrainingConfig = Field(default_factory=AdvancedTrainingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Pipeline settings
    pipeline: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "carla_drl_pipeline",
        "version": "2.0.0",
        "description": "Advanced CARLA DRL Pipeline with Production Features",
        "random_seed": 42,
        "deterministic": True,
        "reproducible": True
    })
    
    # Resource management
    resources: Dict[str, Any] = Field(default_factory=lambda: {
        "cpu_cores": 4,
        "memory_gb": 8,
        "gpu_memory_gb": 6,
        "disk_space_gb": 50,
        "network_bandwidth_mbps": 100
    })
    
    # Environment management
    environment: Dict[str, Any] = Field(default_factory=lambda: {
        "python_version": "3.12",
        "carla_python_version": "3.6",
        "ros2_distro": "humble",
        "cuda_version": "11.8",
        "conda_env": "carla_drl",
        "virtual_env": "./venv"
    })
    
    @validator('pipeline')
    def validate_pipeline_config(cls, v):
        """Validate pipeline configuration."""
        required_fields = ['name', 'version', 'description']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Pipeline config missing required field: {field}")
        return v
    
    @validator('resources')
    def validate_resource_requirements(cls, v):
        """Validate resource requirements are sufficient."""
        min_requirements = {
            'cpu_cores': 2,
            'memory_gb': 4,
            'gpu_memory_gb': 2,
            'disk_space_gb': 20
        }
        
        for resource, min_value in min_requirements.items():
            if v.get(resource, 0) < min_value:
                raise ValueError(f"Insufficient {resource}: {v.get(resource)} < {min_value}")
        
        return v

# Configuration factory functions
def load_config(config_path: Union[str, Path]) -> AdvancedPipelineConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return AdvancedPipelineConfig(**config_data)

def save_config(config: AdvancedPipelineConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, indent=2)

def create_default_config() -> AdvancedPipelineConfig:
    """Create default configuration with sensible defaults."""
    return AdvancedPipelineConfig()

# Export configuration models
__all__ = [
    'AdvancedCarlaConfig',
    'CurriculumConfig', 
    'AdvancedTrainingConfig',
    'MonitoringConfig',
    'AdvancedPipelineConfig',
    'load_config',
    'save_config',
    'create_default_config'
]
