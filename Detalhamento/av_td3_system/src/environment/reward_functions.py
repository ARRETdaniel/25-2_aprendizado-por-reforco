"""
Multi-Component Reward Function for Autonomous Vehicle Navigation

Implements the reward function from the IEEE paper:
- Efficiency: Maintain target speed
- Lane Keeping: Minimize lateral deviation and heading error
- Comfort: Minimize jerk (smooth acceleration)
- Safety: Large penalty for collisions and off-road events

Paper Reference: "End-to-End Visual Autonomous Navigation with Twin Delayed DDPG"
Section III.B: Reward Function R(s_t, a_t)
"""

import numpy as np
from typing import Dict
import logging


class RewardCalculator:
    """
    Computes multi-component reward for driving policy training.
    
    Reward components (configurable weights):
    1. Efficiency: Reward for target speed tracking
    2. Lane Keeping: Reward for staying in lane
    3. Comfort: Penalty for high jerk
    4. Safety: Large penalty for collisions/off-road
    """

    def __init__(self, config: Dict):
        """
        Initialize reward calculator with configuration.

        Args:
            config: Dictionary with reward parameters from training_config.yaml
                Expected keys:
                - reward.weights: {efficiency, lane_keeping, comfort, safety}
                - reward.efficiency: {target_speed, speed_tolerance, overspeed_penalty_scale}
                - reward.lane_keeping: {lateral_tolerance, heading_tolerance}
                - reward.comfort: {jerk_threshold}
                - reward.safety: {collision_penalty, offroad_penalty, wrong_way_penalty}
        """
        self.logger = logging.getLogger(__name__)

        # Extract weights
        self.weights = config.get("weights", {
            "efficiency": 1.0,
            "lane_keeping": 2.0,
            "comfort": 0.5,
            "safety": -100.0,
        })

        # Efficiency parameters
        self.target_speed = config.get("efficiency", {}).get("target_speed", 10.0)  # m/s
        self.speed_tolerance = config.get("efficiency", {}).get("speed_tolerance", 2.0)
        self.overspeed_penalty_scale = config.get("efficiency", {}).get(
            "overspeed_penalty_scale", 2.0
        )

        # Lane keeping parameters
        self.lateral_tolerance = config.get("lane_keeping", {}).get(
            "lateral_tolerance", 0.5
        )
        self.heading_tolerance = config.get("lane_keeping", {}).get(
            "heading_tolerance", 0.1
        )

        # Comfort parameters
        self.jerk_threshold = config.get("comfort", {}).get("jerk_threshold", 3.0)

        # Safety parameters
        self.collision_penalty = config.get("safety", {}).get("collision_penalty", -1000.0)
        self.offroad_penalty = config.get("safety", {}).get("offroad_penalty", -500.0)
        self.wrong_way_penalty = config.get("safety", {}).get(
            "wrong_way_penalty", -200.0
        )

        # State tracking for jerk calculation
        self.prev_acceleration = 0.0
        self.prev_acceleration_lateral = 0.0

    def calculate(
        self,
        velocity: float,
        lateral_deviation: float,
        heading_error: float,
        acceleration: float,
        acceleration_lateral: float,
        collision_detected: bool,
        offroad_detected: bool,
        wrong_way: bool = False,
        distance_to_goal: float = 0.0,
    ) -> Dict:
        """
        Calculate multi-component reward.

        Args:
            velocity: Current vehicle velocity (m/s)
            lateral_deviation: Distance from lane center (m, positive=right)
            heading_error: Heading error w.r.t. lane direction (radians)
            acceleration: Current longitudinal acceleration (m/s²)
            acceleration_lateral: Current lateral acceleration (m/s²)
            collision_detected: Whether collision occurred this step
            offroad_detected: Whether vehicle went off-road
            wrong_way: Whether vehicle is driving in wrong direction
            distance_to_goal: Distance remaining to destination (m)

        Returns:
            Dictionary with:
            - total: Total reward
            - efficiency: Efficiency component
            - lane_keeping: Lane keeping component
            - comfort: Comfort component
            - safety: Safety component
            - breakdown: Detailed breakdown of all components
        """
        reward_dict = {}

        # 1. EFFICIENCY REWARD: Target speed tracking
        efficiency = self._calculate_efficiency_reward(velocity)
        reward_dict["efficiency"] = efficiency

        # 2. LANE KEEPING REWARD: Minimize lateral deviation and heading error
        lane_keeping = self._calculate_lane_keeping_reward(
            lateral_deviation, heading_error
        )
        reward_dict["lane_keeping"] = lane_keeping

        # 3. COMFORT PENALTY: Minimize jerk
        comfort = self._calculate_comfort_reward(acceleration, acceleration_lateral)
        reward_dict["comfort"] = comfort

        # Update previous accelerations for next step
        self.prev_acceleration = acceleration
        self.prev_acceleration_lateral = acceleration_lateral

        # 4. SAFETY PENALTY: Large penalty for dangerous events
        safety = self._calculate_safety_reward(
            collision_detected, offroad_detected, wrong_way
        )
        reward_dict["safety"] = safety

        # Calculate total weighted reward
        total_reward = (
            self.weights["efficiency"] * efficiency
            + self.weights["lane_keeping"] * lane_keeping
            + self.weights["comfort"] * comfort
            + self.weights["safety"] * safety
        )

        reward_dict["total"] = total_reward
        reward_dict["breakdown"] = {
            "efficiency": (
                self.weights["efficiency"],
                efficiency,
                self.weights["efficiency"] * efficiency,
            ),
            "lane_keeping": (
                self.weights["lane_keeping"],
                lane_keeping,
                self.weights["lane_keeping"] * lane_keeping,
            ),
            "comfort": (
                self.weights["comfort"],
                comfort,
                self.weights["comfort"] * comfort,
            ),
            "safety": (
                self.weights["safety"],
                safety,
                self.weights["safety"] * safety,
            ),
        }

        return reward_dict

    def _calculate_efficiency_reward(self, velocity: float) -> float:
        """
        Calculate efficiency reward for target speed tracking.

        Reward increases if velocity is close to target speed.
        Penalizes both underspeeding and overspeeding (with higher penalty for overspeeding).

        Args:
            velocity: Current velocity (m/s)

        Returns:
            Efficiency reward (typically 0 to 1)
        """
        speed_diff = abs(velocity - self.target_speed)

        if speed_diff <= self.speed_tolerance:
            # Within tolerance: positive reward
            efficiency = 1.0 - (speed_diff / self.speed_tolerance) * 0.5
        else:
            # Outside tolerance: negative reward
            excess = speed_diff - self.speed_tolerance

            if velocity > self.target_speed:
                # Overspeeding: higher penalty
                efficiency = -excess / self.target_speed * self.overspeed_penalty_scale
            else:
                # Underspeeding: lower penalty
                efficiency = -excess / self.target_speed * 0.5

        return float(np.clip(efficiency, -1.0, 1.0))

    def _calculate_lane_keeping_reward(
        self, lateral_deviation: float, heading_error: float
    ) -> float:
        """
        Calculate lane keeping reward.

        Reward is high when vehicle stays centered in lane with correct heading.

        Args:
            lateral_deviation: Perpendicular distance from lane center (m)
            heading_error: Heading error w.r.t. lane direction (radians)

        Returns:
            Lane keeping reward (typically -1 to 1)
        """
        # Lateral deviation component (normalized)
        lat_error = min(abs(lateral_deviation) / self.lateral_tolerance, 1.0)
        lat_reward = 1.0 - lat_error * 0.7

        # Heading error component (normalized)
        head_error = min(abs(heading_error) / self.heading_tolerance, 1.0)
        head_reward = 1.0 - head_error * 0.3

        # Combined reward (penalize both errors but heading less critical)
        lane_keeping = (lat_reward + head_reward) / 2.0 - 0.5  # Range: [-0.5, 0.5]

        return float(np.clip(lane_keeping, -1.0, 1.0))

    def _calculate_comfort_reward(
        self, acceleration: float, acceleration_lateral: float
    ) -> float:
        """
        Calculate comfort reward (penalize high jerk).

        Jerk = rate of change of acceleration. High jerk indicates jerky/uncomfortable driving.

        Args:
            acceleration: Current longitudinal acceleration (m/s²)
            acceleration_lateral: Current lateral acceleration (m/s²)

        Returns:
            Comfort reward (typically -1 to 0)
        """
        # Calculate jerk (change in acceleration)
        jerk_long = abs(acceleration - self.prev_acceleration)
        jerk_lat = abs(acceleration_lateral - self.prev_acceleration_lateral)

        # Combined jerk magnitude
        total_jerk = np.sqrt(jerk_long**2 + jerk_lat**2)

        if total_jerk <= self.jerk_threshold:
            # Below threshold: small positive reward for smoothness
            comfort = (1.0 - total_jerk / self.jerk_threshold) * 0.3
        else:
            # Above threshold: negative reward for jerky motion
            excess_jerk = total_jerk - self.jerk_threshold
            comfort = -excess_jerk / self.jerk_threshold

        return float(np.clip(comfort, -1.0, 0.3))

    def _calculate_safety_reward(
        self, collision_detected: bool, offroad_detected: bool, wrong_way: bool
    ) -> float:
        """
        Calculate safety reward.

        Large negative penalties for unsafe events (collisions, off-road, wrong-way driving).

        Args:
            collision_detected: Whether collision occurred
            offroad_detected: Whether vehicle went off-road
            wrong_way: Whether vehicle is driving wrong direction

        Returns:
            Safety reward (0 if safe, very negative if unsafe)
        """
        safety = 0.0

        if collision_detected:
            safety += self.collision_penalty
        if offroad_detected:
            safety += self.offroad_penalty
        if wrong_way:
            safety += self.wrong_way_penalty

        return float(safety)

    def reset(self):
        """Reset internal state for new episode."""
        self.prev_acceleration = 0.0
        self.prev_acceleration_lateral = 0.0
