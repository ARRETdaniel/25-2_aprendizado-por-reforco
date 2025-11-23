"""
PID Controller for Longitudinal Vehicle Control.

This module implements a Proportional-Integral-Derivative (PID) controller
for tracking desired vehicle speed. The implementation is adapted from the
TCC controller2d.py for compatibility with CARLA 0.9.16 Python API.

Author: GitHub Copilot Agent
Date: 2025-01-20
Based on: related_works/exemple-carlaClient-openCV-YOLO-town01-waypoints/controller2d.py
"""

from typing import Tuple
import numpy as np


class PIDController:
    """
    PID Controller for longitudinal (speed) control.

    This controller computes throttle and brake commands to track a desired speed.
    It uses the standard PID formula with anti-windup integrator saturation.

    Attributes:
        kp (float): Proportional gain
        ki (float): Integral gain
        kd (float): Derivative gain
        integrator_min (float): Minimum integrator value (anti-windup)
        integrator_max (float): Maximum integrator value (anti-windup)
        v_error_integral (float): Accumulated speed error (integral term)
        v_error_prev (float): Previous speed error (for derivative term)
    """

    def __init__(
        self,
        kp: float = 0.50,
        ki: float = 0.30,
        kd: float = 0.13,
        integrator_min: float = 0.0,
        integrator_max: float = 10.0
    ):
        """
        Initialize PID controller with specified gains.

        Args:
            kp: Proportional gain (default: 0.50 from controller2d.py)
            ki: Integral gain (default: 0.30 from controller2d.py)
            kd: Derivative gain (default: 0.13 from controller2d.py)
            integrator_min: Minimum integrator value for anti-windup (default: 0.0)
            integrator_max: Maximum integrator value for anti-windup (default: 10.0)
        """
        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Integrator limits (anti-windup)
        self.integrator_min = integrator_min
        self.integrator_max = integrator_max

        # State variables
        self.v_error_integral = 0.0
        self.v_error_prev = 0.0

    def reset(self) -> None:
        """
        Reset controller state (integral and derivative terms).

        This should be called at the start of each episode or when the
        controller needs to be reinitialized.
        """
        self.v_error_integral = 0.0
        self.v_error_prev = 0.0

    def update(
        self,
        current_speed: float,
        target_speed: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Compute throttle and brake commands based on speed error.

        The PID formula used is:
            control = kp * error + ki * integral(error) + kd * d(error)/dt

        Positive control output maps to throttle, negative to brake.

        Args:
            current_speed: Current vehicle speed in m/s
            target_speed: Desired vehicle speed in m/s
            dt: Time step in seconds (should match CARLA fixed_delta_seconds)

        Returns:
            Tuple[throttle, brake] where:
                - throttle: Throttle command in [0.0, 1.0]
                - brake: Brake command in [0.0, 1.0]

        Example:
            >>> controller = PIDController()
            >>> throttle, brake = controller.update(current_speed=5.0, target_speed=10.0, dt=0.05)
            >>> print(f"Throttle: {throttle:.2f}, Brake: {brake:.2f}")
        """
        # Compute speed error
        v_error = target_speed - current_speed

        # Integral term with anti-windup
        self.v_error_integral += v_error * dt
        self.v_error_integral = np.clip(
            self.v_error_integral,
            self.integrator_min,
            self.integrator_max
        )

        # Derivative term
        if dt > 0:
            v_error_derivative = (v_error - self.v_error_prev) / dt
        else:
            v_error_derivative = 0.0

        # PID control law
        control_output = (
            self.kp * v_error +
            self.ki * self.v_error_integral +
            self.kd * v_error_derivative
        )

        # Split control into throttle and brake
        # Positive control -> throttle, Negative control -> brake
        if control_output >= 0:
            throttle = np.clip(control_output, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-control_output, 0.0, 1.0)

        # Update previous error for next iteration
        self.v_error_prev = v_error

        return throttle, brake

    def __repr__(self) -> str:
        """String representation of the controller."""
        return (
            f"PIDController(kp={self.kp}, ki={self.ki}, kd={self.kd}, "
            f"integrator_range=[{self.integrator_min}, {self.integrator_max}])"
        )
