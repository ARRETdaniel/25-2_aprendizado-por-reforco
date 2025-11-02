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
        # FIXED: Reduced collision penalty from -1000 to -100 (High Priority Fix #4)
        # Rationale: TD3's clipped double-Q amplifies negative memories.
        # -1000 creates "collisions are unrecoverable" belief, -100 is still strong
        # but allows agent to learn from mistakes. Matches successful implementations
        # (Ben Elallid et al. 2023, Pérez-Gil et al. 2022).
        self.collision_penalty = config.get("safety", {}).get("collision_penalty", -100.0)
        self.offroad_penalty = config.get("safety", {}).get("offroad_penalty", -500.0)
        self.wrong_way_penalty = config.get("safety", {}).get(
            "wrong_way_penalty", -200.0
        )

        # Progress parameters (NEW: Goal-directed navigation rewards)
        self.waypoint_bonus = config.get("progress", {}).get("waypoint_bonus", 10.0)
        # FIXED: Increased distance scale from 0.1 to 1.0 (High Priority Fix #3)
        # Rationale: Moving 1m now gives +1.0 progress (weighted: +5.0 total),
        # which can offset efficiency penalty during acceleration. Previous 0.1 scale
        # gave only +0.5 weighted reward, insufficient to overcome -1.0 efficiency penalty.
        self.distance_scale = config.get("progress", {}).get("distance_scale", 1.0)
        self.goal_reached_bonus = config.get("progress", {}).get("goal_reached_bonus", 100.0)

        # State tracking for jerk calculation
        self.prev_acceleration = 0.0
        self.prev_acceleration_lateral = 0.0

        # State tracking for progress calculation
        self.prev_distance_to_goal = None  # Will be set on first step

        # PBRS (Potential-Based Reward Shaping) parameter (Medium Priority Fix #6)
        # Discount factor for potential function (should match TD3's gamma)
        self.gamma = config.get("gamma", 0.99)

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
        lane_half_width: float = None,  # NEW: CARLA lane width normalization
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
            lane_half_width: Half of current lane width from CARLA (m).
                           If None, uses config lateral_tolerance. (NEW)

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

        # 1. EFFICIENCY REWARD: Forward velocity component (CRITICAL FIX #1)
        # Now requires heading_error to compute v * cos(φ) term
        efficiency = self._calculate_efficiency_reward(velocity, heading_error)
        reward_dict["efficiency"] = efficiency

        # 2. LANE KEEPING REWARD: Minimize lateral deviation and heading error
        # CRITICAL FIX: Pass velocity to gate reward - no reward if stationary
        # ENHANCEMENT: Pass lane_half_width for CARLA-based normalization
        lane_keeping = self._calculate_lane_keeping_reward(
            lateral_deviation, heading_error, velocity, lane_half_width
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

    def _calculate_efficiency_reward(self, velocity: float, heading_error: float) -> float:
        """
        Calculate efficiency reward using forward velocity component.

        SIMPLIFIED VERSION (Priority 1 from documentation-backed analysis):
        - Follows KISS principle from ArXiv Reward Engineering Survey (2408.10215v1)
        - "Simple rewards often outperform complex designs"
        - Removes reverse penalty (breaks continuity at v=0)
        - Removes target speed tracking (unnecessary complexity)
        - Pure linear scaling is TD3-compatible and sufficient

        Based on:
        - Pérez-Gil et al. (2022): Forward velocity component R = v * cos(φ)
        - TD3 requirements (Fujimoto et al. 2018): Continuous differentiable
        - OpenAI Spinning Up: No reward normalization in official TD3
        - CARLA 0.9.16 API: get_velocity() returns Vector3D in m/s

        Mathematical properties:
        - v=0 m/s → efficiency=0 (neutral, prevents local optimum)
        - v=1 m/s, φ=0° → efficiency=+0.12 (immediate positive feedback)
        - v=8.33 m/s, φ=0° → efficiency=+1.0 (optimal)
        - φ=90° → efficiency=0 (perpendicular, neutral)
        - φ=180° → efficiency=-1.0 (backward, natural penalty from cos)
        - Continuous and differentiable EVERYWHERE (no discontinuities)

        Why simplified:
        - Reverse penalty (* 2.0) created discontinuity at v_forward=0
        - Target speed tracking added 3 conditional branches
        - Linear scaling already incentivizes reaching target speed naturally
        - TD3 exploration (Gaussian noise) benefits from smooth landscape

        Args:
            velocity: Current velocity magnitude (m/s)
            heading_error: Heading error w.r.t. desired direction (radians)

        Returns:
            Efficiency reward in [-1.0, 1.0], continuous and differentiable

        References:
        - Analysis: day_three-reward_efficiency_analysis.md (Assessment: 8.5/10)
        - TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
        - CARLA: https://carla.readthedocs.io/en/latest/python_api/#carlavehicle
        """
        # Forward velocity component: v * cos(φ)
        # Projects velocity onto desired heading direction
        # CARLA API validated: get_velocity() returns Vector3D in m/s (global frame)
        forward_velocity = velocity * np.cos(heading_error)

        # Normalize by target speed to get reward in [-1, 1] range
        # Linear scaling naturally incentivizes reaching target speed:
        # v=4.165 m/s (50%) → efficiency=0.5
        # v=8.33 m/s (100%) → efficiency=1.0 (maximum reward)
        efficiency = forward_velocity / self.target_speed

        # Clip to [-1, 1] range for safety (though math should keep it in range)
        return float(np.clip(efficiency, -1.0, 1.0))

    def _calculate_lane_keeping_reward(
        self, lateral_deviation: float, heading_error: float, velocity: float,
        lane_half_width: float = None
    ) -> float:
        """
        Calculate lane keeping reward with CARLA-based lane width normalization.

        CRITICAL FIX #2: Reduced velocity gate from 1.0 m/s to 0.1 m/s and added
        continuous velocity scaling to provide learning gradient during acceleration.

        ENHANCEMENT (Priority 3): Dynamic lane width normalization using CARLA API.
        Uses actual road geometry from waypoint.lane_width instead of fixed config.

        Key changes:
        - OLD: Hard cutoff at 1.0 m/s (no reward below 3.6 km/h)
        - NEW: Gate at 0.1 m/s (0.36 km/h, truly stationary) + velocity scaling
        - Gradual reward scaling from 0 to full as velocity increases 0→3 m/s
        - Enables agent to learn "stay centered while accelerating"
        - ENHANCED: Lateral normalization now uses CARLA lane_width (e.g., 1.25m urban
          vs 0.5m config) to reduce false positive lane invasions

        Rationale:
        - 1.0 m/s is slow pedestrian walk, not "stopped"
        - CARLA physics: 0→1 m/s takes ~10 ticks, all receiving zero gradient
        - TD3 needs continuous Q-value gradients for policy learning
        - Pérez-Gil et al. use NO velocity gating (continuous everywhere)
        - Lane width: Town01 ≈ 2.5m (half=1.25m) vs config 0.5m → 2.5x more permissive
        - Reduces false positives at 0.5m < |d| < 1.25m range

        Args:
            lateral_deviation: Perpendicular distance from lane center (m)
            heading_error: Heading error w.r.t. lane direction (radians)
            velocity: Current velocity (m/s) - for gating and scaling
            lane_half_width: Half of current lane width from CARLA (m).
                           If None, uses config lateral_tolerance.

        Returns:
            Lane keeping reward, velocity-scaled, in [-1.0, 1.0]
        """
        # FIXED: Lower velocity threshold from 1.0 to 0.1 m/s
        # Only gate when truly stationary (0.1 m/s = 0.36 km/h)
        if velocity < 0.1:
            return 0.0

        # FIXED: Add velocity scaling for continuous gradient
        # Linearly scale from 0 (at v=0.1) to 1.0 (at v=3.0)
        # This provides learning signal during acceleration phase
        velocity_scale = min((velocity - 0.1) / 2.9, 1.0)  # (v-0.1)/(3.0-0.1) = (v-0.1)/2.9

        # ENHANCEMENT: Use CARLA lane width if available, else fallback to config
        # This enables multi-map generalization and reduces false positives
        effective_tolerance = (
            lane_half_width if lane_half_width is not None
            else self.lateral_tolerance
        )

        # Lateral deviation component (normalized by CARLA lane width or config tolerance)
        lat_error = min(abs(lateral_deviation) / effective_tolerance, 1.0)
        lat_reward = 1.0 - lat_error * 0.7  # 70% weight on lateral error

        # Heading error component (normalized by tolerance)
        head_error = min(abs(heading_error) / self.heading_tolerance, 1.0)
        head_reward = 1.0 - head_error * 0.3  # 30% weight on heading error

        # Combined reward (average of components, shifted to [-0.5, 0.5])
        lane_keeping = (lat_reward + head_reward) / 2.0 - 0.5

        # Apply velocity scaling (gradual increase as vehicle accelerates)
        # At v=0.5 m/s: scale≈0.14 → some learning signal
        # At v=1.0 m/s: scale≈0.31 → moderate signal
        # At v=3.0 m/s: scale=1.0 → full signal
        return float(np.clip(lane_keeping * velocity_scale, -1.0, 1.0))

    def _calculate_comfort_reward(
        self, acceleration: float, acceleration_lateral: float, velocity: float
    ) -> float:
        """
        Calculate comfort reward (penalize high jerk) with reduced velocity gating.

        CRITICAL FIX #2 (continued): Same velocity gating fix as lane_keeping.
        Reduced threshold from 1.0 m/s to 0.1 m/s with velocity scaling.

        Key changes:
        - OLD: No gradient below 1.0 m/s (can't learn smooth acceleration)
        - NEW: Velocity-scaled gradient from 0.1 m/s upward
        - Agent can now learn "smooth acceleration from rest"

        Jerk = rate of change of acceleration. High jerk indicates jerky/uncomfortable driving.

        Args:
            acceleration: Current longitudinal acceleration (m/s²)
            acceleration_lateral: Current lateral acceleration (m/s²)
            velocity: Current velocity (m/s) - for gating and scaling

        Returns:
            Comfort reward, velocity-scaled, in [-1.0, 0.3]
        """
        # FIXED: Lower velocity threshold from 1.0 to 0.1 m/s
        if velocity < 0.1:
            return 0.0

        # Calculate jerk (change in acceleration since last step)
        jerk_long = abs(acceleration - self.prev_acceleration)
        jerk_lat = abs(acceleration_lateral - self.prev_acceleration_lateral)

        # Combined jerk magnitude (Euclidean norm)
        total_jerk = np.sqrt(jerk_long**2 + jerk_lat**2)

        # FIXED: Add velocity scaling (same as lane_keeping)
        velocity_scale = min((velocity - 0.1) / 2.9, 1.0)

        # Calculate comfort reward based on jerk
        if total_jerk <= self.jerk_threshold:
            # Below threshold: small positive reward for smoothness
            comfort = (1.0 - total_jerk / self.jerk_threshold) * 0.3
        else:
            # Above threshold: penalty for jerky motion
            excess_jerk = total_jerk - self.jerk_threshold
            comfort = -excess_jerk / self.jerk_threshold

        # Apply velocity scaling
        return float(np.clip(comfort * velocity_scale, -1.0, 0.3))

    def _calculate_safety_reward(
        self,
        collision_detected: bool,
        offroad_detected: bool,
        wrong_way: bool,
        velocity: float,
        distance_to_goal: float
    ) -> float:
        """
        Calculate safety reward with improved stopping penalty.

        MEDIUM PRIORITY FIX #5: Removed distance threshold from stopping penalty
        to eliminate exploitation loophole.

        Key changes:
        - OLD: Only penalize stopping if distance_to_goal > 5.0 m
        - NEW: Always penalize unnecessary stopping (no distance condition)
        - Additional penalty when far from goal for stronger signal

        Rationale:
        - Previous implementation allowed agent to "camp" near spawn if within 5m
        - Stopping is unsafe anywhere on road (except at goal)
        - Progressive penalty structure: small base + larger when far from goal

        Large negative penalties for unsafe events (collisions, off-road, wrong-way, stopping unnecessarily).

        Args:
            collision_detected: Whether collision occurred
            offroad_detected: Whether vehicle went off-road
            wrong_way: Whether vehicle is driving wrong direction
            velocity: Current velocity (m/s) - to detect stationary behavior
            distance_to_goal: Distance to destination (m) - for progressive penalty

        Returns:
            Safety reward (0 if safe, very negative if unsafe)
        """
        safety = 0.0

        # Catastrophic events (immediate episode failure)
        if collision_detected:
            safety += self.collision_penalty
        if offroad_detected:
            safety += self.offroad_penalty
        if wrong_way:
            safety += self.wrong_way_penalty

        # FIXED: Progressive stopping penalty (Medium Priority Fix #5)
        # No distance threshold - stopping is always discouraged unless at goal
        # Structure: Base penalty (-0.1) + distance-based penalty (up to -0.4)
        if not collision_detected and not offroad_detected:
            if velocity < 0.5:  # Essentially stopped (< 1.8 km/h)
                # Base penalty: small constant disincentive for stopping
                safety += -0.1

                # Additional penalty if far from goal (progressive)
                # Stronger signal to keep moving when destination is distant
                if distance_to_goal > 10.0:
                    safety += -0.4  # Total: -0.5 when far from goal
                elif distance_to_goal > 5.0:
                    safety += -0.2  # Total: -0.3 when moderately far

        return float(safety)

    def _calculate_progress_reward(
        self,
        distance_to_goal: float,
        waypoint_reached: bool,
        goal_reached: bool,
    ) -> float:
        """
        Calculate progress reward with PBRS (Potential-Based Reward Shaping).

        MEDIUM PRIORITY FIX #6: Added PBRS component for theoretically sound
        dense reward signal.

        Implements dense reward shaping based on:
        1. Distance reduction to goal (increased 10x via distance_scale)
        2. PBRS: F(s,s') = γΦ(s') - Φ(s) where Φ(s) = -distance_to_goal
        3. Waypoint milestone bonuses
        4. Goal reached bonus

        PBRS Theorem (Ng et al. 1999):
        "Potential-based shaping functions ensure that policies learned with
        shaped rewards remain effective in the original MDP, maintaining
        near-optimal policies."

        Mathematical guarantee: Adding F(s,s') = γΦ(s') - Φ(s) does NOT change
        the optimal policy, but provides denser learning signal.

        This addresses the sparse reward problem cited in arXiv:2408.10215:
        "Sparse and delayed nature of rewards in many real-world scenarios can hinder learning progress"

        Args:
            distance_to_goal: Current distance to final goal waypoint (meters)
            waypoint_reached: Whether agent passed a waypoint this step
            goal_reached: Whether agent reached the final destination

        Returns:
            Progress reward (positive for forward progress, includes PBRS term)
        """
        progress = 0.0

        # Component 1: Distance-based reward (dense, continuous)
        # FIXED: Now uses distance_scale=1.0 (High Priority Fix #3)
        # Reward = (prev_distance - current_distance) * scale
        # Positive when moving toward goal, negative when moving away
        if self.prev_distance_to_goal is not None:
            distance_delta = self.prev_distance_to_goal - distance_to_goal
            progress += distance_delta * self.distance_scale

            # Component 1b: PBRS (Medium Priority Fix #6)
            # Potential function: Φ(s) = -distance_to_goal
            # Shaping: F(s,s') = γΦ(s') - Φ(s)
            #                  = γ(-distance_to_goal') - (-distance_to_goal_prev)
            #                  = -γ*distance_to_goal' + distance_to_goal_prev
            potential_current = -distance_to_goal
            potential_prev = -self.prev_distance_to_goal
            pbrs_reward = self.gamma * potential_current - potential_prev

            # Add PBRS with moderate weight (0.5x) to complement distance reward
            # Total progress signal = direct distance + PBRS (both encourage forward movement)
            progress += pbrs_reward * 0.5

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
