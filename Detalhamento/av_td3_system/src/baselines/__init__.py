"""Classical baseline agents for autonomous vehicle control."""

# Import only modules that don't require CARLA at import time
from src.baselines.pid_controller import PIDController
from src.baselines.pure_pursuit_controller import PurePursuitController
from src.baselines.baseline_controller import BaselineController

# IDMMOBILBaseline requires CARLA to be installed
# Import conditionally if needed:
# from src.baselines.idm_mobil import IDMMOBILBaseline

__all__ = ['PIDController', 'PurePursuitController', 'BaselineController']
