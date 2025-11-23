"""
Combined Baseline Controller for Autonomous Vehicle Navigation.

This module integrates PID (longitudinal) and Pure Pursuit (lateral) controllers
to provide complete vehicle control. It interfaces with CARLA 0.9.16 Python API
and can be used as a baseline for comparison with deep reinforcement learning agents.

Author: GitHub Copilot Agent
Date: 2025-01-20
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import carla

from src.baselines.pid_controller import PIDController
from src.baselines.pure_pursuit_controller import PurePursuitController


class BaselineController:
    """
    Combined PID + Pure Pursuit controller for autonomous vehicle navigation.
    
    This controller provides complete vehicle control by combining:
    - PID Controller: Longitudinal control (speed tracking)
    - Pure Pursuit Controller: Lateral control (path following)
    
    The controller takes vehicle state and waypoints as input and outputs
    a carla.VehicleControl command that can be applied to the vehicle.
    
    Attributes:
        pid_controller (PIDController): Longitudinal controller
        pure_pursuit_controller (PurePursuitController): Lateral controller
        target_speed (float): Default target speed in m/s
    """
    
    def __init__(
        self,
        # PID parameters
        pid_kp: float = 0.50,
        pid_ki: float = 0.30,
        pid_kd: float = 0.13,
        integrator_min: float = 0.0,
        integrator_max: float = 10.0,
        # Pure Pursuit parameters
        lookahead_distance: float = 2.0,
        kp_heading: float = 8.00,
        k_speed_crosstrack: float = 0.00,
        cross_track_deadband: float = 0.01,
        # General parameters
        target_speed: float = 30.0  # km/h (will be converted to m/s)
    ):
        """
        Initialize the combined baseline controller.
        
        Args:
            pid_kp: PID proportional gain (default: 0.50 from controller2d.py)
            pid_ki: PID integral gain (default: 0.30 from controller2d.py)
            pid_kd: PID derivative gain (default: 0.13 from controller2d.py)
            integrator_min: Minimum PID integrator value
            integrator_max: Maximum PID integrator value
            lookahead_distance: Pure Pursuit lookahead in meters (default: 2.0)
            kp_heading: Pure Pursuit heading gain (default: 8.00)
            k_speed_crosstrack: Pure Pursuit speed-dependent crosstrack gain
            cross_track_deadband: Minimum crosstrack error to react to
            target_speed: Default target speed in km/h (converted to m/s internally)
        """
        # Initialize controllers
        self.pid_controller = PIDController(
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            integrator_min=integrator_min,
            integrator_max=integrator_max
        )
        
        self.pure_pursuit_controller = PurePursuitController(
            lookahead_distance=lookahead_distance,
            kp_heading=kp_heading,
            k_speed_crosstrack=k_speed_crosstrack,
            cross_track_deadband=cross_track_deadband
        )
        
        # Convert target speed from km/h to m/s
        self.target_speed = target_speed / 3.6  # km/h -> m/s
    
    def reset(self) -> None:
        """
        Reset both controllers' state.
        
        Should be called at the start of each episode.
        """
        self.pid_controller.reset()
        # Pure Pursuit is stateless, no reset needed
    
    def compute_control(
        self,
        vehicle: carla.Vehicle,
        waypoints: List[Tuple[float, float, float]],
        dt: float,
        target_speed: Optional[float] = None
    ) -> carla.VehicleControl:
        """
        Compute vehicle control commands from current state and waypoints.
        
        This method:
        1. Extracts vehicle state from CARLA vehicle object
        2. Computes throttle/brake using PID controller
        3. Computes steering using Pure Pursuit controller
        4. Returns a carla.VehicleControl command
        
        Args:
            vehicle: CARLA vehicle actor (provides get_transform(), get_velocity())
            waypoints: List of (x, y, speed) tuples defining the reference path
            dt: Time step in seconds (should match CARLA's fixed_delta_seconds)
            target_speed: Optional override for target speed in m/s
                         (if None, uses self.target_speed)
        
        Returns:
            carla.VehicleControl with throttle, steer, brake commands
        
        Example:
            >>> controller = BaselineController()
            >>> waypoints = [(0, 0, 8.33), (10, 0, 8.33), (20, 5, 8.33)]
            >>> control = controller.compute_control(vehicle, waypoints, dt=0.05)
            >>> vehicle.apply_control(control)
        """
        # Extract vehicle state from CARLA API
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        # Position (meters)
        current_x = transform.location.x
        current_y = transform.location.y
        
        # Yaw (convert from degrees to radians for controller)
        current_yaw = np.radians(transform.rotation.yaw)
        
        # Speed (convert from Vector3D to scalar m/s)
        current_speed = np.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )
        
        # Use provided target speed or default
        if target_speed is None:
            target_speed = self.target_speed
        
        # Compute longitudinal control (throttle/brake)
        throttle, brake = self.pid_controller.update(
            current_speed=current_speed,
            target_speed=target_speed,
            dt=dt
        )
        
        # Compute lateral control (steering)
        steer = self.pure_pursuit_controller.update(
            current_x=current_x,
            current_y=current_y,
            current_yaw=current_yaw,
            current_speed=current_speed,
            waypoints=waypoints
        )
        
        # Create CARLA VehicleControl command
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0
        )
        
        return control
    
    def get_debug_info(self, vehicle: carla.Vehicle) -> Dict[str, Any]:
        """
        Get debug information about current controller state.
        
        Useful for logging and debugging.
        
        Args:
            vehicle: CARLA vehicle actor
        
        Returns:
            Dictionary with debug information
        """
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        return {
            'position': {
                'x': transform.location.x,
                'y': transform.location.y,
                'z': transform.location.z
            },
            'rotation': {
                'pitch': transform.rotation.pitch,
                'yaw': transform.rotation.yaw,
                'roll': transform.rotation.roll
            },
            'speed_m_s': np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
            'pid_state': {
                'integral': self.pid_controller.v_error_integral,
                'prev_error': self.pid_controller.v_error_prev
            },
            'target_speed_m_s': self.target_speed,
            'target_speed_km_h': self.target_speed * 3.6
        }
    
    def __repr__(self) -> str:
        """String representation of the controller."""
        return (
            f"BaselineController(\n"
            f"  PID: {self.pid_controller},\n"
            f"  PurePursuit: {self.pure_pursuit_controller},\n"
            f"  target_speed={self.target_speed:.2f} m/s ({self.target_speed * 3.6:.2f} km/h)\n"
            f")"
        )
