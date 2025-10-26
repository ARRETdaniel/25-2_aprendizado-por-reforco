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
                - reward.weights: {efficiency, lane_keeping, comfort, safety, progress}
                - reward.efficiency: {target_speed, speed_tolerance, overspeed_penalty_scale}
                - reward.lane_keeping: {lateral_tolerance, heading_tolerance}
                - reward.comfort: {jerk_threshold}
                - reward.safety: {collision_penalty, offroad_penalty, wrong_way_penalty}
                - reward.progress: {waypoint_bonus, distance_scale}
        """
        self.logger = logging.getLogger(__name__)

        # Extract weights
        self.weights = config.get("weights", {
            "efficiency": 1.0,
            "lane_keeping": 2.0,
            "comfort": 0.5,
            "safety": -100.0,
            "progress": 5.0,  # NEW: High weight for goal-directed progress
        })

        # Efficiency parameters
        self.target_speed = config.get("efficiency", {}).get("target_speed", 8.33)  # m/s (default: 30 km/h)
        self.speed_tolerance = config.get("efficiency", {}).get("speed_tolerance", 1.39)  # m/s (default: 5 km/h)
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

        # Progress parameters (NEW: Goal-directed navigation rewards)
        self.waypoint_bonus = config.get("progress", {}).get("waypoint_bonus", 10.0)
        self.distance_scale = config.get("progress", {}).get("distance_scale", 0.1)
        self.goal_reached_bonus = config.get("progress", {}).get("goal_reached_bonus", 100.0)

        # State tracking for jerk calculation
        self.prev_acceleration = 0.0
        self.prev_acceleration_lateral = 0.0

        # State tracking for progress calculation
        self.prev_distance_to_goal = None  # Will be set on first step

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
        waypoint_reached: bool = False,
        goal_reached: bool = False,
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
            waypoint_reached: Whether agent reached a waypoint this step (NEW)
            goal_reached: Whether agent reached final goal this step (NEW)

        Returns:
            Dictionary with:
            - total: Total reward
            - efficiency: Efficiency component
            - lane_keeping: Lane keeping component
            - comfort: Comfort component
            - safety: Safety component
            - progress: Progress component (NEW)
            - breakdown: Detailed breakdown of all components
        """
        reward_dict = {}

        # 1. EFFICIENCY REWARD: Target speed tracking
        efficiency = self._calculate_efficiency_reward(velocity)
        reward_dict["efficiency"] = efficiency

        # 2. LANE KEEPING REWARD: Minimize lateral deviation and heading error
        # CRITICAL FIX: Pass velocity to gate reward - no reward if stationary
        lane_keeping = self._calculate_lane_keeping_reward(
            lateral_deviation, heading_error, velocity
        )
        reward_dict["lane_keeping"] = lane_keeping

        # 3. COMFORT PENALTY: Minimize jerk
        # CRITICAL FIX: Pass velocity to gate reward - no reward if stationary
        comfort = self._calculate_comfort_reward(acceleration, acceleration_lateral, velocity)
        reward_dict["comfort"] = comfort

        # Update previous accelerations for next step
        self.prev_acceleration = acceleration
        self.prev_acceleration_lateral = acceleration_lateral

        # 4. SAFETY PENALTY: Large penalty for dangerous events
        # CRITICAL FIX: Pass velocity and distance_to_goal to penalize unnecessary stopping
        safety = self._calculate_safety_reward(
            collision_detected, offroad_detected, wrong_way, velocity, distance_to_goal
        )
        reward_dict["safety"] = safety

        # 5. PROGRESS REWARD: Reward forward progress toward goal (NEW)
        progress = self._calculate_progress_reward(
            distance_to_goal, waypoint_reached, goal_reached
        )
        reward_dict["progress"] = progress

        # Calculate total weighted reward
        total_reward = (
            self.weights["efficiency"] * efficiency
            + self.weights["lane_keeping"] * lane_keeping
            + self.weights["comfort"] * comfort
            + self.weights["safety"] * safety
            + self.weights["progress"] * progress
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
            "progress": (
                self.weights["progress"],
                progress,
                self.weights["progress"] * progress,
            ),
        }

        return reward_dict

    def _calculate_efficiency_reward(self, velocity: float) -> float:
        """
        Calculate efficiency reward for target speed tracking.

        CRITICAL: Agent must be incentivized to MOVE. Following Pérez-Gil et al. (2022):
        - Reward longitudinal velocity (forward movement)
        - Heavily penalize staying still or moving too slow
        - Penalize excessive speed

        Paper formula: R = Σ|v_t * cos(φ_t)| - |v_t * sin(φ_t)| - |v_t| * |d_t|

        Args:
            velocity: Current velocity (m/s)

        Returns:
            Efficiency reward (range: -1.0 to 1.0, BUT heavily negative when not moving)
        """
        # Normalize velocity to [0, 1] where 1.0 = target speed
        velocity_normalized = velocity / self.target_speed

        if velocity < 1.0:  # Below 1 m/s (3.6 km/h) = essentially stopped
            # STRONG penalty for not moving - agent must learn to accelerate
            efficiency = -1.0
        elif velocity < self.target_speed * 0.5:  # Below half target speed
            # Moderate penalty for moving too slow
            efficiency = -0.5 + (velocity_normalized * 0.5)
        elif abs(velocity - self.target_speed) <= self.speed_tolerance:
            # Within tolerance: positive reward (optimal range)
            speed_diff = abs(velocity - self.target_speed)
            efficiency = 1.0 - (speed_diff / self.speed_tolerance) * 0.3
        else:
            # Outside tolerance
            if velocity > self.target_speed:
                # Overspeeding: penalty but less than underspeeding
                excess = velocity - self.target_speed
                efficiency = 0.7 - (excess / self.target_speed) * self.overspeed_penalty_scale
            else:
                # Underspeeding: penalty
                deficit = self.target_speed - velocity
                efficiency = -0.3 - (deficit / self.target_speed) * 0.3

        return float(np.clip(efficiency, -1.0, 1.0))

    def _calculate_lane_keeping_reward(
        self, lateral_deviation: float, heading_error: float, velocity: float
    ) -> float:
        """
        Calculate lane keeping reward.

        CRITICAL FIX: Reward is ONLY given when vehicle is MOVING.
        Agent should not be rewarded for staying centered while stationary.

        Args:
            lateral_deviation: Perpendicular distance from lane center (m)
            heading_error: Heading error w.r.t. lane direction (radians)
            velocity: Current velocity (m/s) - REQUIRED to gate the reward

        Returns:
            Lane keeping reward (0 if stationary, -1 to 1 if moving)
        """
        # CRITICAL: No lane keeping reward if not moving!
        # Vehicle must be moving at least 1 m/s (3.6 km/h) to get this reward
        if velocity < 1.0:
            return 0.0  # Zero reward for staying centered while stationary

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
        self, acceleration: float, acceleration_lateral: float, velocity: float
    ) -> float:
        """
        Calculate comfort reward (penalize high jerk).

        CRITICAL FIX: Comfort reward is ONLY given when vehicle is MOVING.
        Agent should not be rewarded for smoothness while stationary.

        Jerk = rate of change of acceleration. High jerk indicates jerky/uncomfortable driving.

        Args:
            acceleration: Current longitudinal acceleration (m/s²)
            acceleration_lateral: Current lateral acceleration (m/s²)
            velocity: Current velocity (m/s) - REQUIRED to gate the reward

        Returns:
            Comfort reward (0 if stationary, -1 to 0.3 if moving)
        """
        # CRITICAL: No comfort reward if not moving!
        # Vehicle must be moving at least 1 m/s (3.6 km/h) to get this reward
        if velocity < 1.0:
            return 0.0  # Zero reward for smoothness while stationary

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
        self,
        collision_detected: bool,
        offroad_detected: bool,
        wrong_way: bool,
        velocity: float,
        distance_to_goal: float
    ) -> float:
        """
        Calculate safety reward.

        CRITICAL FIX: Added stationary vehicle penalty.
        It is UNSAFE to stop in the middle of the road when:
        - No collision/obstruction
        - Not off-road
        - Still have distance to cover to goal

        Large negative penalties for unsafe events (collisions, off-road, wrong-way, stopping unnecessarily).

        Args:
            collision_detected: Whether collision occurred
            offroad_detected: Whether vehicle went off-road
            wrong_way: Whether vehicle is driving wrong direction
            velocity: Current velocity (m/s) - to detect stationary behavior
            distance_to_goal: Distance to destination (m) - to verify goal not reached

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

        # REMOVED: Overly aggressive stopping penalty during exploration
        # The agent needs time to learn how to move forward without being
        # constantly penalized. The efficiency reward already handles this by
        # rewarding target speed achievement. Let the agent explore!

        # OLD CODE (disabled):
        # if velocity < 0.5 and distance_to_goal > 5.0 and not collision_detected and not offroad_detected:
        #     safety += -1.0  # This was preventing exploration

        return float(safety)

    def _calculate_progress_reward(
        self,
        distance_to_goal: float,
        waypoint_reached: bool,
        goal_reached: bool,
    ) -> float:
        """
        Calculate progress reward for goal-directed navigation.

        Implements dense reward shaping based on:
        1. Distance reduction to goal (negative reward if moving away)
        2. Waypoint milestone bonuses
        3. Goal reached bonus

        This addresses the sparse reward problem cited in arXiv:2408.10215:
        "Sparse and delayed nature of rewards in many real-world scenarios can hinder learning progress"

        Args:
            distance_to_goal: Current distance to final goal waypoint (meters)
            waypoint_reached: Whether agent passed a waypoint this step
            goal_reached: Whether agent reached the final destination

        Returns:
            Progress reward (positive for forward progress, negative for backward)
        """
        progress = 0.0

        # Component 1: Distance-based reward (dense, continuous)
        # Reward = (prev_distance - current_distance) * scale
        # Positive when moving toward goal, negative when moving away
        if self.prev_distance_to_goal is not None:
            distance_delta = self.prev_distance_to_goal - distance_to_goal
            # Normalize by distance scale for better reward magnitude
            progress += distance_delta * self.distance_scale

        # Update tracking
        self.prev_distance_to_goal = distance_to_goal

        # Component 2: Waypoint milestone bonus (sparse but frequent)
        # Encourage agent to reach intermediate waypoints
        if waypoint_reached:
            progress += self.waypoint_bonus
            self.logger.info(f"🎯 Waypoint reached! Bonus: +{self.waypoint_bonus:.1f}")

        # Component 3: Goal reached bonus (sparse but terminal)
        # Large reward for completing the route
        if goal_reached:
            progress += self.goal_reached_bonus
            self.logger.info(f"🏁 Goal reached! Bonus: +{self.goal_reached_bonus:.1f}")

        return float(np.clip(progress, -10.0, 110.0))  # Clip to reasonable range

    def reset(self):
        """Reset internal state for new episode."""
        self.prev_acceleration = 0.0
        self.prev_acceleration_lateral = 0.0
        self.prev_distance_to_goal = None  # Reset progress tracking
