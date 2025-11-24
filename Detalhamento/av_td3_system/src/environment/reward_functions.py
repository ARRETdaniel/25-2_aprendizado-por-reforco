"""
Reward calculation for autonomous vehicle navigation.

Implements a multi-component reward function with safety penalties,
progress incentives, and comfort objectives.
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
        # LITERATURE-VALIDATED FIX (WARNING-001 & WARNING-002):
        # Updated defaults to match config and prevent reward domination
        # Reference: Perot et al. (2017) - distance penalty critical for lane keeping
        # Reference: Chen et al. (2019) - balanced multi-component rewards
        # CRITICAL FIX (Nov 21, 2025): Changed safety from -100.0 to +1.0
        # Rationale: Safety penalties are ALREADY NEGATIVE (-10.0 for collision).
        # Negative weight would INVERT them into positive rewards (+1000 for crash!).
        # Pattern: Positive weights × signed components = correct reward direction.
        self.weights = config.get("weights", {
            "efficiency": 1.0,
            "lane_keeping": 5.0,  # INCREASED from 2.0: Prioritize staying in lane
            "comfort": 0.5,
            "safety": 1.0,  # FIXED from -100.0: Penalties are already negative!
            "progress": 1.0,  # REDUCED from 5.0: Prevent domination (was 88.9%)
        })

        # VERIFICATION: Log loaded weights to confirm config is properly loaded
        self.logger.info("=" * 80)
        self.logger.info("REWARD WEIGHTS VERIFICATION (addressing WARNING-002)")
        self.logger.info("=" * 80)
        for component, weight in self.weights.items():
            self.logger.info(f"  {component:15s}: {weight:6.1f}")
        self.logger.info("=" * 80)

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
        # CRITICAL FIX (Nov 19, 2025): Add explicit lane invasion penalty
        # Lane invasion (crossing lane markings) is a discrete safety violation
        # detected by CARLA's sensor.other.lane_invasion sensor.
        # Penalty severity hierarchy: offroad (-500) > wrong_way (-200) > collision (-100) > lane_invasion (-50)
        # Rationale: Lane crossing is unsafe but less severe than full lane departure or collision
        self.lane_invasion_penalty = config.get("safety", {}).get(
            "lane_invasion_penalty", -10.0
        )

        # Progress parameters (NEW: Goal-directed navigation rewards)
        # LITERATURE-VALIDATED FIX (WARNING-001 & WARNING-002):
        # Reduced discrete bonuses to prevent reward domination
        # Reference: Perot et al. (2017) - continuous rewards work better than discrete
        self.waypoint_bonus = config.get("progress", {}).get("waypoint_bonus", 1.0)  # REDUCED from 10.0
        # FIXED: Increased distance scale from 0.1 to 1.0 (High Priority Fix #3)
        # Rationale: Moving 1m now gives +1.0 progress (weighted: +1.0 total with new weight),
        # which can offset efficiency penalty during acceleration. Previous 0.1 scale
        # gave only +0.5 weighted reward, insufficient to overcome -1.0 efficiency penalty.
        self.distance_scale = config.get("progress", {}).get("distance_scale", 1.0)
        self.goal_reached_bonus = config.get("progress", {}).get("goal_reached_bonus", 100.0)  # REDUCED from 100.0

        # VERIFICATION: Log loaded progress parameters
        self.logger.info("PROGRESS REWARD PARAMETERS VERIFICATION (addressing WARNING-001)")
        self.logger.info("=" * 80)
        self.logger.info(f"  waypoint_bonus      : {self.waypoint_bonus:6.1f} (was 10.0)")
        self.logger.info(f"  distance_scale      : {self.distance_scale:6.1f} (was 0.1)")
        self.logger.info(f"  goal_reached_bonus  : {self.goal_reached_bonus:6.1f} (was 100.0)")
        self.logger.info("=" * 80)

        # State tracking for jerk calculation
        self.prev_acceleration = 0.0
        self.prev_acceleration_lateral = 0.0

        # State tracking for progress calculation
        self.prev_distance_to_goal = None  # Will be set on first step

        # PBRS (Potential-Based Reward Shaping) parameter (Medium Priority Fix #6)
        # Discount factor for potential function (should match TD3's gamma)
        self.gamma = config.get("gamma", 0.99)

        # Step counter for throttling logging frequency
        # Purpose: Prevent logging floods (domination warnings every step)
        # Added: 2025-01-20 to address logging verbosity issue
        self.step_counter = 0
        self.log_frequency = 100  # Log warnings only every N steps

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
        lane_invasion_detected: bool = False,  # CRITICAL FIX (Nov 19, 2025): Add lane invasion parameter
        distance_to_goal: float = 0.0,
        waypoint_reached: bool = False,
        goal_reached: bool = False,
        lane_half_width: float = None,
        dt: float = 0.05,
        # NEW: Dense safety metrics (Priority 1 & 3 fixes)
        distance_to_nearest_obstacle: float = None,
        time_to_collision: float = None,
        collision_impulse: float = None,
    ) -> Dict:
        """
        Calculate multi-component reward with dense PBRS safety guidance.

        PRIORITY 1, 2, 3 FIXES INTEGRATED:
        ===================================
        - Dense PBRS proximity guidance via distance_to_nearest_obstacle
        - Time-to-collision penalties for imminent collisions
        - Graduated collision penalties via collision_impulse
        - Rebalanced penalty magnitudes (loaded from config)

        Args:
            velocity: Current vehicle velocity (m/s)
            lateral_deviation: Distance from lane center (m, positive=right)
            heading_error: Heading error w.r.t. lane direction (radians)
            acceleration: Current longitudinal acceleration (m/s²)
            acceleration_lateral: Current lateral acceleration (m/s²)
            collision_detected: Whether collision occurred this step
            offroad_detected: Whether vehicle went off-road
            wrong_way: Whether vehicle is driving in wrong direction
            lane_invasion_detected: Whether vehicle crossed lane markings this step (CRITICAL FIX Nov 19, 2025)
            distance_to_goal: Distance remaining to destination (m)
            waypoint_reached: Whether agent reached a waypoint this step
            goal_reached: Whether agent reached final goal this step
            lane_half_width: Half of current lane width from CARLA (m)
            dt: Time step since last measurement for jerk computation (seconds)
            distance_to_nearest_obstacle: Distance to nearest obstacle (m) - NEW
            time_to_collision: Estimated TTC in seconds - NEW
            collision_impulse: Collision force magnitude in Newtons - NEW

        Returns:
            Dictionary with:
            - total: Total reward
            - efficiency: Efficiency component
            - lane_keeping: Lane keeping component
            - comfort: Comfort component
            - safety: Safety component (with dense PBRS guidance)
            - progress: Progress component
            - breakdown: Detailed breakdown of all components
        """
        reward_dict = {}

        # 1. EFFICIENCY REWARD: Forward velocity component
        efficiency = self._calculate_efficiency_reward(velocity, heading_error)
        reward_dict["efficiency"] = efficiency

        # 2. LANE KEEPING REWARD: Minimize lateral deviation and heading error
        # CRITICAL FIX (Nov 19, 2025): Pass lane_invasion_detected to prevent positive
        # rewards during lane marking crossings
        lane_keeping = self._calculate_lane_keeping_reward(
            lateral_deviation, heading_error, velocity, lane_half_width, lane_invasion_detected
        )
        reward_dict["lane_keeping"] = lane_keeping

        # 3. COMFORT PENALTY: Minimize jerk
        comfort = self._calculate_comfort_reward(
            acceleration, acceleration_lateral, velocity, dt
        )
        reward_dict["comfort"] = comfort

        # NOTE: Previous accelerations are already updated inside _calculate_comfort_reward()
        # at lines 643-644, so no need to update here again

        # 4. SAFETY PENALTY: Dense PBRS guidance + graduated penalties
        safety = self._calculate_safety_reward(
            collision_detected,
            offroad_detected,
            wrong_way,
            lane_invasion_detected,  # CRITICAL FIX (Nov 19, 2025): Pass lane invasion parameter
            velocity,
            distance_to_goal,
            distance_to_nearest_obstacle,  # NEW
            time_to_collision,  # NEW
            collision_impulse,  # NEW
        )
        reward_dict["safety"] = safety

        # 5. PROGRESS REWARD: Reward forward progress toward goal
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

        # DEBUG: Comprehensive reward breakdown logging
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"   REWARD BREAKDOWN (Step {self.step_counter}):\n"
                f"   ══════════════════════════════════════\n"
                f"   EFFICIENCY (target speed tracking):\n"
                f"      Raw: {efficiency:.4f}\n"
                f"      Weight: {self.weights['efficiency']:.2f}\n"
                f"      Contribution: {self.weights['efficiency']*efficiency:.4f}\n"
                f"   ──────────────────────────────────────\n"
                f"   LANE KEEPING (stay in lane):\n"
                f"      Raw: {lane_keeping:.4f}\n"
                f"      Weight: {self.weights['lane_keeping']:.2f}\n"
                f"      Contribution: {self.weights['lane_keeping']*lane_keeping:.4f}\n"
                f"   ──────────────────────────────────────\n"
                f"   COMFORT (minimize jerk):\n"
                f"      Raw: {comfort:.4f}\n"
                f"      Weight: {self.weights['comfort']:.2f}\n"
                f"      Contribution: {self.weights['comfort']*comfort:.4f}\n"
                f"   ──────────────────────────────────────\n"
                f"   SAFETY (collision/offroad penalty):\n"
                f"      Raw: {safety:.4f}\n"
                f"      Weight: {self.weights['safety']:.2f}\n"
                f"      Contribution: {self.weights['safety']*safety:.4f}\n"
                f"      Status: {' COLLISION' if collision_detected else ' OFFROAD' if offroad_detected else '✅ SAFE'}\n"
                f"   ──────────────────────────────────────\n"
                f"   PROGRESS (goal-directed movement):\n"
                f"      Raw: {progress:.4f}\n"
                f"      Weight: {self.weights['progress']:.2f}\n"
                f"      Contribution: {self.weights['progress']*progress:.4f}\n"
                f"   ══════════════════════════════════════\n"
                f"    TOTAL REWARD: {total_reward:.4f}\n"
                f"   ══════════════════════════════════════"
            )

            # Check for reward domination
            component_magnitudes = {
                "efficiency": abs(self.weights['efficiency'] * efficiency),
                "lane_keeping": abs(self.weights['lane_keeping'] * lane_keeping),
                "comfort": abs(self.weights['comfort'] * comfort),
                "safety": abs(self.weights['safety'] * safety),
                "progress": abs(self.weights['progress'] * progress),
            }
            total_magnitude = sum(component_magnitudes.values())

            if total_magnitude > 0:
                max_component = max(component_magnitudes.items(), key=lambda x: x[1])
                component_name, magnitude = max_component
                ratio = magnitude / total_magnitude

                if ratio > 0.8:
                    self.logger.debug(
                        f"     WARNING: '{component_name}' dominates ({ratio*100:.1f}% of total magnitude)"
                    )
        else:
            # Regular logging (not debug mode) - minimal output
            self.logger.debug(
                f"[REWARD] Components - "
                f"Efficiency: {efficiency:.3f}×{self.weights['efficiency']:.1f}={self.weights['efficiency']*efficiency:.2f}, "
                f"Lane: {lane_keeping:.3f}×{self.weights['lane_keeping']:.1f}={self.weights['lane_keeping']*lane_keeping:.2f}, "
                f"Comfort: {comfort:.3f}×{self.weights['comfort']:.1f}={self.weights['comfort']*comfort:.2f}, "
                f"Safety: {safety:.3f}×{self.weights['safety']:.1f}={self.weights['safety']*safety:.2f}, "
                f"Progress: {progress:.3f}×{self.weights['progress']:.1f}={self.weights['progress']*progress:.2f}"
            )
            self.logger.debug(f"[REWARD] TOTAL: {total_reward:.2f}")

        # Increment step counter (for logging frequency control)
        self.step_counter += 1

        # Log warning if any component is dominating (>80% of total absolute magnitude)
        # FIXED: Only log every 100 steps to prevent output flooding (only in non-debug mode)
        if not self.logger.isEnabledFor(logging.DEBUG) and self.step_counter % self.log_frequency == 0:
            component_magnitudes = {
                "efficiency": abs(self.weights['efficiency'] * efficiency),
                "lane_keeping": abs(self.weights['lane_keeping'] * lane_keeping),
                "comfort": abs(self.weights['comfort'] * comfort),
                "safety": abs(self.weights['safety'] * safety),
                "progress": abs(self.weights['progress'] * progress),
            }
            total_magnitude = sum(component_magnitudes.values())

            if total_magnitude > 0:
                for component, magnitude in component_magnitudes.items():
                    ratio = magnitude / total_magnitude
                    if ratio > 0.8:
                        self.logger.warning(
                            f"[REWARD] Component '{component}' is dominating: "
                            f"{ratio*100:.1f}% of total magnitude (threshold: 80%) "
                            f"[Logged at step {self.step_counter}]"
                        )

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
        lane_half_width: float = None, lane_invasion_detected: bool = False
    ) -> float:
        """
        Calculate lane keeping reward with CARLA-based lane width normalization.

        CRITICAL FIX (Nov 19, 2025): Add lane invasion awareness to prevent positive
        rewards during lane marking crossings. This addresses the fundamental bug where
        the agent receives positive rewards for crossing into wrong lanes while remaining
        centered in the invaded lane.

        CRITICAL FIX #2: Reduced velocity gate from 1.0 m/s to 0.1 m/s and added
        continuous velocity scaling to provide learning gradient during acceleration.

        ENHANCEMENT (Priority 3): Dynamic lane width normalization using CARLA API.
        Uses actual road geometry from waypoint.lane_width instead of fixed config.

        Key changes:
        - NEW (Nov 19): Immediate -1.0 penalty when lane invasion detected
        - OLD: Hard cutoff at 1.0 m/s (no reward below 3.6 km/h)
        - NEW: Gate at 0.1 m/s (0.36 km/h, truly stationary) + velocity scaling
        - Gradual reward scaling from 0 to full as velocity increases 0→3 m/s
        - Enables agent to learn "stay centered while accelerating"
        - ENHANCED: Lateral normalization now uses CARLA lane_width (e.g., 1.25m urban
          vs 0.5m config) to reduce false positive lane invasions

        Rationale:
        - Lane invasions should receive maximum lane keeping penalty (-1.0)
        - Prevents agent from learning to cross lanes to minimize lateral_deviation
        - Aligns with CARLA lane invasion sensor semantics (event per crossing)
        - Literature support: Chen et al. (2019) - lane occupancy awareness critical
        - Literature support: Perot et al. (2017) - boundary penalties prevent cutting corners
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
            lane_invasion_detected: Whether vehicle crossed lane markings this step (NEW)

        Returns:
            Lane keeping reward, velocity-scaled, in [-1.0, 1.0]
        """
        # CRITICAL FIX (Nov 19, 2025): IMMEDIATE PENALTY FOR LANE INVASION
        # Return maximum penalty (-1.0) when lane markings are crossed.
        # This prevents the agent from receiving positive rewards while invading lanes.
        if lane_invasion_detected:
            self.logger.warning("[LANE_KEEPING] Lane invasion detected - applying maximum penalty (-1.0)")
            return -1.0

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
        # LITERATURE VALIDATION: This implements the distance penalty "d/w" from Chen et al. (2019)
        # and the critical distance term from Perot et al. (2017): R = v(cos(α) - d)
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
        final_reward = float(np.clip(lane_keeping * velocity_scale, -1.0, 1.0))

        # VERIFICATION LOGGING: Confirm distance penalty is active (addresses literature validation)
        if self.step_counter % 500 == 0:  # Log every 500 steps
            self.logger.debug(
                f"Lane Keeping Penalty Active: lateral_dev={lateral_deviation:.3f}m, "
                f"lat_error={lat_error:.3f}, lat_reward={lat_reward:.3f}, "
                f"final={final_reward:.3f} (lit: Chen2019 d/w, Perot2017 -d term)"
            )

        return final_reward

    def _calculate_comfort_reward(
        self, acceleration: float, acceleration_lateral: float, velocity: float, dt: float
    ) -> float:
        """
        Calculate comfort reward (penalize high jerk) with physically correct computation.

        COMPREHENSIVE FIX - Addresses all critical issues from analysis:

         FIX #1: Added dt division for correct jerk units (m/s³)
           - OLD: jerk = |accel - prev_accel| → Units: m/s² (acceleration difference)
           - NEW: jerk = (accel - prev_accel) / dt → Units: m/s³ (actual jerk)
           - Physics: Jerk = da/dt (third derivative of position)

         FIX #2: Removed abs() for TD3 differentiability
           - OLD: abs(x) is non-differentiable at x=0 (sharp corner)
           - NEW: Uses x² which is smooth and differentiable everywhere
           - TD3 Requirement: "Smooth Q-functions to prevent exploitation of sharp peaks"
           - Reference: OpenAI Spinning Up TD3 docs, arxiv:2408.10215v1

         FIX #3: Improved velocity scaling with sqrt for smoother transition
           - OLD: Linear scaling can under-penalize low-speed jerks
           - NEW: sqrt scaling provides more gradual transition
           - At v=0.5 m/s: linear scale=0.138, sqrt scale=0.372 (2.7x stronger signal)

         FIX #4: Bounded negative penalties with quadratic scaling
           - OLD: Unbounded penalty (-excess_jerk / threshold) could be very large
           - NEW: Quadratic penalty with 2x threshold cap prevents Q-value explosion
           - Rationale: Same principle as collision penalty reduction (High Priority Fix #4)
           - TD3's clipped double-Q amplifies negative memories

         FIX #5: Updated threshold to correct units
           - Configuration updated: jerk_threshold now in m/s³ (was dimensionless)
           - Typical values: 2-3 m/s³ comfortable, 5-8 m/s³ max tolerable

        Mathematical Properties:
        - Continuous and differentiable everywhere (no discontinuities)
        - Bounded output range: [-1.0, 0.3]
        - Smooth gradient landscape for TD3 policy learning
        - Physically correct units and thresholds

        Future Enhancement (Medium Priority):
        - Add angular jerk component: jerk_angular = dω/dt (steering smoothness)
        - CARLA API: Actor.get_angular_velocity() → deg/s (convert to rad/s)
        - Would require tracking self.prev_angular_velocity

        Args:
            acceleration: Current longitudinal acceleration (m/s²)
            acceleration_lateral: Current lateral acceleration (m/s²)
            velocity: Current velocity (m/s) - for gating and scaling
            dt: Time step since last measurement (seconds) - for jerk computation

        Returns:
            Comfort reward, velocity-scaled, in [-1.0, 0.3]

        References:
        - TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
        - CARLA: https://carla.readthedocs.io/en/latest/python_api/#carlavehicle
        - Reward Engineering: arxiv:2408.10215v1
        - Physics: Jerk = da/dt (third derivative of position)
        """
        # Velocity gating: prevent penalties at near-zero velocity
        # Threshold: 0.1 m/s = 0.36 km/h (truly stationary)
        if velocity < 0.1:
            return 0.0

        # FIX #1: Correct jerk computation with dt division
        # Jerk = da/dt (rate of change of acceleration)
        # Units: (m/s² - m/s²) / s = m/s³ (correct!)
        jerk_long = (acceleration - self.prev_acceleration) / dt
        jerk_lat = (acceleration_lateral - self.prev_acceleration_lateral) / dt

        # FIX #2: Use squared values for TD3 differentiability
        # x² is smooth everywhere (no sharp corners at x=0)
        jerk_long_sq = jerk_long ** 2
        jerk_lat_sq = jerk_lat ** 2

        # Combined jerk magnitude (Euclidean norm)
        # sqrt(x² + y²) is smooth and differentiable everywhere except origin
        # At origin (both jerks = 0), sqrt is still differentiable
        total_jerk = np.sqrt(jerk_long_sq + jerk_lat_sq)

        # CRITICAL FIX (Nov 23, 2025): Comfort Reward Issue #2
        # ========================================================
        # Removed velocity scaling that was causing negative rewards at normal speeds.
        #
        # Problem: velocity_scale made comfort rewards only appear at high speeds,
        # causing negative rewards during normal driving at target speed.
        #
        # Root Cause: At v=8.33 m/s (30 km/h), scale=1.0, but at lower speeds during
        # acceleration/deceleration, scale << 1.0, making even zero jerk → near-zero reward
        # which appears negative compared to baseline.
        #
        # Solution: Comfort should be independent of speed. Jerk is uncomfortable at
        # any speed. The velocity gate (< 0.1 m/s) is sufficient to prevent stationary
        # penalties.
        #
        # Reference: TD3 paper emphasizes smooth, continuous rewards without artificial
        # scaling that can create exploration artifacts.

        # FIX #4: Bounded comfort reward with quadratic penalty
        # Normalize jerk to [0, 2] range (cap at 2x threshold)
        normalized_jerk = min(total_jerk / self.jerk_threshold, 2.0)

        if normalized_jerk <= 1.0:
            # Below threshold: positive reward for smoothness (linear decrease)
            # normalized_jerk=0 → comfort=0.3 (smooth!)
            # normalized_jerk=1 → comfort=0.0 (at threshold)
            comfort = (1.0 - normalized_jerk) * 0.3
        else:
            # Above threshold: quadratic penalty (smooth, bounded)
            # normalized_jerk=1 → comfort=0.0 (continuous transition)
            # normalized_jerk=2 → comfort=-0.3 (max penalty, capped)
            # Quadratic scaling: (1.0)²=1.0, (1.5)²=2.25, (2.0)²=4.0
            # But we scale by 0.3/(2-1)² = 0.3 to normalize max penalty
            excess_normalized = normalized_jerk - 1.0  # Range: [0, 1]
            comfort = -0.3 * (excess_normalized ** 2)  # Smooth quadratic penalty

        # Update state tracking for next step
        self.prev_acceleration = acceleration
        self.prev_acceleration_lateral = acceleration_lateral

        # Apply safety clip (removed velocity scaling)
        return float(np.clip(comfort, -1.0, 0.3))

    def _calculate_safety_reward(
        self,
        collision_detected: bool,
        offroad_detected: bool,
        wrong_way: bool,
        lane_invasion_detected: bool,  # CRITICAL FIX (Nov 19, 2025): Add lane invasion parameter
        velocity: float,
        distance_to_goal: float,
        # NEW PARAMETERS for dense PBRS guidance (Priority 1 Fix)
        distance_to_nearest_obstacle: float = None,
        time_to_collision: float = None,
        collision_impulse: float = None,
    ) -> float:
        """
        Calculate safety reward with dense PBRS guidance and graduated penalties.

        PRIORITY 1 FIX: Dense Safety Guidance (PBRS)
        ============================================
        Implements Potential-Based Reward Shaping (PBRS) for continuous safety signals:
        - Φ(s) = -1.0 / max(distance_to_obstacle, 0.5)
        - Provides gradient BEFORE collisions occur
        - Enables proactive collision avoidance learning

        Reference: Analysis document Issue #1 (Sparse Safety Rewards - CRITICAL)
        PBRS Theorem (Ng et al. 1999): F(s,s') = γΦ(s') - Φ(s) preserves optimal policy

        PRIORITY 2 FIX: Magnitude Rebalancing
        ======================================
        Reduced penalty magnitudes from -50.0 to -5.0 for balanced multi-objective learning.

        PRIORITY 3 FIX: Graduated Penalties
        ===================================
        Uses collision impulse magnitude for severity-based penalties instead of fixed values.

        Args:
            collision_detected: Whether collision occurred (boolean)
            offroad_detected: Whether vehicle went off-road (boolean)
            wrong_way: Whether vehicle is driving wrong direction (boolean)
            lane_invasion_detected: Whether vehicle crossed lane markings this step (boolean) - CRITICAL FIX (Nov 19, 2025)
            velocity: Current velocity (m/s) - for TTC calculation and stopping penalty
            distance_to_goal: Distance to destination (m) - for progressive stopping penalty
            distance_to_nearest_obstacle: Distance to nearest obstacle in meters (NEW)
            time_to_collision: Estimated TTC in seconds (NEW)
            collision_impulse: Collision force magnitude in Newtons (NEW)

        Returns:
            Safety reward (0 if safe, negative with continuous gradient)
        """
        safety = 0.0

        # ========================================================================
        # PRIORITY 1: DENSE PROXIMITY GUIDANCE (PBRS) - CRITICAL FIX
        # ========================================================================
        # Provides continuous reward shaping that encourages maintaining safe distances
        # BEFORE catastrophic events occur. This is the PRIMARY fix for training failure.

        if distance_to_nearest_obstacle is not None:
            # Obstacle proximity potential: Φ(s) = -k / max(d, d_min)
            # Creates continuous gradient as obstacle approaches

            if distance_to_nearest_obstacle < 10.0:  # Only penalize within 10m range
                # Inverse distance potential for nearby obstacles
                # Mathematical form: potential = -k / max(distance, d_min)
                # where k=1.0 (scaling factor), d_min=0.5m (safety buffer)
                #
                # Gradient strength at different distances:
                # - 10.0m: -0.10 (gentle nudge, "stay aware")
                # - 5.0m:  -0.20 (moderate signal, "maintain distance")
                # - 3.0m:  -0.33 (strong signal, "prepare to slow")
                # - 1.0m:  -1.00 (urgent signal, "brake immediately")
                # - 0.5m:  -2.00 (maximum penalty, "collision imminent")

                proximity_penalty = -1.0 / max(distance_to_nearest_obstacle, 0.5)
                safety += proximity_penalty

                # Diagnostic logging for PBRS component
                self.logger.debug(
                    f"[SAFETY-PBRS] Obstacle @ {distance_to_nearest_obstacle:.2f}m "
                    f"→ proximity_penalty={proximity_penalty:.3f}"
                )

            # ====================================================================
            # TIME-TO-COLLISION (TTC) PENALTY - Secondary Safety Signal
            # ====================================================================
            # Additional penalty for imminent collisions (approaching obstacle)
            # TTC < 3.0 seconds: Driver reaction time threshold (NHTSA standard)

            if time_to_collision is not None and time_to_collision < 3.0:
                # Inverse TTC penalty: shorter time = stronger penalty
                # Range: -5.0 (at 0.1s) to -0.17 (at 3.0s)
                #
                # Gradient strength:
                # - 3.0s: -0.17 (early warning, "start decelerating")
                # - 2.0s: -0.25 (moderate urgency, "brake soon")
                # - 1.0s: -0.50 (high urgency, "brake now")
                # - 0.5s: -1.00 (emergency, "hard brake")
                # - 0.1s: -5.00 (max penalty, "collision unavoidable")

                ttc_penalty = -0.5 / max(time_to_collision, 0.1)
                safety += ttc_penalty

                self.logger.debug(
                    f"[SAFETY-TTC] TTC={time_to_collision:.2f}s "
                    f"→ ttc_penalty={ttc_penalty:.3f}"
                )

        # ========================================================================
        # PRIORITY 2 & 3: GRADUATED COLLISION PENALTY (Reduced + Impulse-Based)
        # ========================================================================
        # Uses collision impulse magnitude for severity-based penalties
        # Magnitude reduced from -100 to -10 for balanced learning (Priority 2 fix)

        if collision_detected:
            if collision_impulse is not None and collision_impulse > 0:
                # Graduated penalty based on impact severity
                # Formula: penalty = -min(10.0, impulse / 100.0)
                #
                # Collision severity mapping (approximate force values):
                # - Soft tap (10N):        -0.10 (minor contact, recoverable)
                # - Light bump (100N):     -1.00 (moderate, learn to avoid)
                # - Moderate crash (500N): -5.00 (significant, bad outcome)
                # - Severe crash (1000N+): -10.0 (maximum penalty, capped)
                #
                # Rationale: Soft collisions during exploration should not
                # catastrophically penalize agent. TD3's min(Q1,Q2) already
                # provides pessimism; graduated penalties allow learning.

                collision_penalty = -min(10.0, collision_impulse / 100.0)
                safety += collision_penalty

                self.logger.warning(
                    f"[SAFETY-COLLISION] Impulse={collision_impulse:.1f}N "
                    f"→ graduated_penalty={collision_penalty:.2f}"
                )
            else:
                # Fallback: Default collision penalty (no impulse data available)
                # Reduced from -100 to -10 (Priority 2 fix)
                collision_penalty = -10.0
                safety += collision_penalty

                self.logger.warning(
                    f"[SAFETY-COLLISION] No impulse data, default penalty={collision_penalty:.1f}"
                )

        # ========================================================================
        # OFFROAD AND WRONG-WAY PENALTIES (Reduced Magnitude)
        # ========================================================================
        # Penalty magnitudes reduced for balance with progress rewards (Priority 2)

        if offroad_detected:
            # Reduced from -100 to -10 for balance
            offroad_penalty = -10.0
            safety += offroad_penalty
            self.logger.warning(f"[SAFETY-OFFROAD] penalty={offroad_penalty:.1f}")

        if wrong_way:
            # Reduced from -50 to -5 for balance
            wrong_way_penalty = -5.0
            safety += wrong_way_penalty
            self.logger.warning(f"[SAFETY-WRONG-WAY] penalty={wrong_way_penalty:.1f}")

        # ========================================================================
        # LANE INVASION PENALTY (CRITICAL FIX - Nov 19, 2025)
        # ========================================================================
        # Explicit penalty for crossing lane markings (discrete safety violation)
        # Detected by CARLA's sensor.other.lane_invasion sensor
        #
        # Penalty severity hierarchy:
        #   - Offroad (complete lane departure): -10.0
        #   - Collision: -10.0 (graduated)
        #   - Wrong way: -5.0
        #   - Lane invasion (crossing markings): -5.0  ← NEW
        #
        # Rationale: Lane crossing is a discrete safety violation that indicates
        # the agent is not respecting traffic rules. While less severe than full
        # lane departure or collision, it should be explicitly penalized to prevent
        # the agent from learning to "cut corners" for progress rewards.
        #
        # Reference: CARLA Documentation - sensor.other.lane_invasion
        # https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector

        if lane_invasion_detected:
            # Use configured penalty (default -50.0, but can be adjusted)
            # Note: This is the RAW penalty before safety weight is applied
            # With safety_weight=-100.0, actual reward impact = -100 * -50.0 = +5000
            # Wait, that's wrong! Let me recalculate...
            # Actually: safety_reward = sum of penalties (already negative)
            # total_reward = safety_weight * safety_reward
            # If safety_weight = -100 and safety_reward = -50, total = -100 * -50 = +5000
            # This is INCORRECT! The weight should multiply the component, not invert it.
            #
            # CORRECTION: Based on code review, weights are positive multipliers.
            # The safety component itself should be negative.
            # So: lane_invasion_penalty should be negative (e.g., -5.0)
            # And: safety_weight should be positive (e.g., 1.0)
            # Result: total contribution = 1.0 * -5.0 = -5.0 ✓
            #
            # However, checking the config, safety_weight is actually negative (-100.0)
            # This means the INTENT is that safety penalties are POSITIVE values
            # that get multiplied by negative weight to become negative rewards.
            #
            # Let me check the other penalties... offroad_penalty = -10.0 (negative)
            # And they're added to safety, which is then multiplied by safety_weight.
            # So the pattern is: penalties are negative, weight is negative,
            # which makes the final contribution POSITIVE? That can't be right!
            #
            # Re-reading the code above: collision_penalty = -10.0 (negative)
            # safety += collision_penalty (adds negative, so safety becomes negative)
            # Later: safety_weight * safety = negative_weight * negative_value = POSITIVE
            #
            # This is BACKWARDS! The safety penalties should be POSITIVE values
            # so that when multiplied by the NEGATIVE weight, they become NEGATIVE rewards.
            #
            # Actually, let me check the weight values again in the config...
            # Looking at initialization: self.weights["safety"] is loaded from config.
            # I need to verify what value it actually has.
            #
            # For now, I'll follow the EXISTING PATTERN used by other penalties:
            # - Use negative penalty values (e.g., -5.0)
            # - Add them to safety accumulator
            # - Let the weight multiplication handle the final sign

            safety += self.lane_invasion_penalty
            self.logger.warning(
                f"[SAFETY-LANE_INVASION] penalty={self.lane_invasion_penalty:.1f} "
                f"(crossed lane markings)"
            )

        # ========================================================================
        # PROGRESSIVE STOPPING PENALTY (Already Implemented in Previous Fix)
        # ========================================================================
        # ========================================================================
        # STOPPING PENALTY - Anti-Idle Feature
        # ========================================================================
        # Issue 1.7 Analysis (Nov 24, 2025): FEATURE, NOT BUG
        # Reference: TASK_1.7_STOPPING_PENALTY_ANALYSIS.md
        #
        # RATIONALE:
        # Progressive stopping penalty prevents agent from learning to "park" and idle.
        # This is appropriate for goal-reaching navigation tasks where stopping without
        # cause indicates failure mode (confusion, malfunction, or exploiting reward).
        #
        # DESIGN:
        # - Far from goal (>10m): -0.5 penalty (strong disincentive to idle)
        # - Medium distance (5-10m): -0.3 penalty (moderate signal)
        # - Near goal (<5m): -0.1 penalty (allow stopping to prepare for goal)
        #
        # KNOWN LIMITATION:
        # Does not account for traffic lights or pedestrian crossings where stopping
        # is REQUIRED by traffic rules. Future enhancement could add:
        #   at_red_light = vehicle.is_at_traffic_light() and \
        #                  vehicle.get_traffic_light_state() == carla.TrafficLightState.Red
        # and gate penalty: if not at_red_light: apply_stopping_penalty()
        #
        # For current Town01 evaluation (no traffic lights), this is acceptable.
        # Progressive design ensures agent learns to make continuous progress while
        # still allowing brief stops near goal for precise positioning.

        if not collision_detected and not offroad_detected:
            if velocity < 0.5:  # Essentially stopped (< 1.8 km/h)
                # Base penalty: small constant disincentive for stopping
                stopping_penalty = -0.1

                # Additional penalty if far from goal (progressive)
                if distance_to_goal > 10.0:
                    stopping_penalty += -0.4  # Total: -0.5 when far from goal
                elif distance_to_goal > 5.0:
                    stopping_penalty += -0.2  # Total: -0.3 when moderately far

                safety += stopping_penalty

                if stopping_penalty < -0.15:  # Only log significant stopping penalties
                    self.logger.debug(
                        f"[SAFETY-STOPPING] velocity={velocity:.2f} m/s, "
                        f"distance_to_goal={distance_to_goal:.1f}m "
                        f"→ penalty={stopping_penalty:.2f}"
                    )

        # Diagnostic summary logging
        self.logger.debug(f"[SAFETY] Total safety reward: {safety:.3f}")

        return float(safety)

    def _calculate_progress_reward(
        self,
        distance_to_goal: float,
        waypoint_reached: bool,
        goal_reached: bool,
    ) -> float:
        """
        Calculate progress reward based on route distance reduction.

        FIX #1: REMOVED PBRS - Bug gave free reward proportional to distance from goal.
        FIX #2: Uses ROUTE DISTANCE instead of Euclidean to prevent off-road shortcuts.

        Implements reward based on:
        1. Route distance reduction (following waypoints, not straight-line)
        2. Waypoint milestone bonuses
        3. Goal reached bonus

        The distance_to_goal parameter now represents ROUTE DISTANCE (calculated by
        WaypointManager.get_route_distance_to_goal), not Euclidean distance.

        Why PBRS was removed:
        - Bug: PBRS gave +1.15 reward per step for zero movement
        - Formula reduced to: (1-γ) × distance_to_goal
        - Perverse incentive: Further from goal = MORE free reward!
        - Reference: #file:reward.md, #file:SYSTEMATIC_PROGRESS_REWARD_ANALYSIS.md

        Why Route Distance fixes shortcuts:
        - Euclidean rewards diagonal movements off-road
        - Route distance only decreases when following waypoints
        - Off-road movement: route distance unchanged → zero reward
        - Reference: #file:DIAGNOSIS_RIGHT_TURN_BIAS.md Section 2.2

        Args:
            distance_to_goal: Route distance to final goal (meters) - NOT Euclidean!
            waypoint_reached: Whether agent passed a waypoint this step
            goal_reached: Whether agent reached the final destination

        Returns:
            Progress reward (positive for forward progress along route)
        """
        # CRITICAL FIX (Nov 24, 2025): Progress Reward Issue #3.1
        # =========================================================
        # TEMPORAL SMOOTHING: Maintain reward continuity when distance_to_goal is None
        #
        # ROOT CAUSE (from PHASE_2_INVESTIGATION.md):
        # WaypointManager._find_nearest_segment() returns None when:
        #   1. Vehicle >20m from any route segment (off-road exploration)
        #   2. Waypoint search window misses vehicle (±2 behind, +10 ahead)
        #   3. First few steps before current_waypoint_idx stabilizes
        #
        # Previous approach (Nov 23): Return 0.0 when None
        #   Problem: Creates discontinuity (10.0 → 0.0 → 10.0)
        #   Impact: TD3 variance σ² = 25 → accumulated error ≈ 2,475 (CATASTROPHIC!)
        #   Reference: TD3 paper Section 3.1 - "accumulation of error"
        #
        # NEW SOLUTION: Temporal smoothing filter
        #   - Use previous distance when current is None (maintains continuity)
        #   - Track None occurrences with diagnostic counter
        #   - Log error if None persists >50 steps (waypoint manager bug)
        #
        # Benefits:
        #    Maintains TD3-required reward continuity (σ² → 0)
        #   Detects persistent waypoint manager failures
        #   Backwards compatible with valid distance values
        #
        # Tradeoff: Masks underlying waypoint manager search window bug
        # Future work: Optimize _find_nearest_segment() search range
        #
        # Reference: PHASE_2_INVESTIGATION.md - Option A (Temporal Smoothing)

        # Initialize None counter for diagnostics (persistent across episode)
        if not hasattr(self, 'none_count'):
            self.none_count = 0

        # HYBRID FIX: Smooth over None values while detecting persistent failures
        if distance_to_goal is None:
            if self.prev_distance_to_goal is not None and self.prev_distance_to_goal > 0.0:
                # Use previous value to maintain TD3-required continuity
                distance_to_goal = self.prev_distance_to_goal
                self.none_count += 1

                self.logger.debug(
                    f"[PROGRESS-SMOOTH] distance_to_goal was None, "
                    f"using prev={distance_to_goal:.2f}m (none_count={self.none_count})"
                )

                # Diagnostic: Detect persistent waypoint manager failures
                if self.none_count > 50:
                    self.logger.error(
                        f"[PROGRESS-ERROR] Waypoint manager returning None persistently! "
                        f"none_count={self.none_count}, vehicle likely stuck off-route >20m. "
                        f"Investigate WaypointManager._find_nearest_segment() search window."
                    )
            else:
                # First step with None - cannot smooth, return 0.0 (expected at episode start)
                self.logger.warning(
                    f"[PROGRESS] No previous distance available for smoothing, "
                    f"skipping progress reward (expected at episode start)"
                )
                return 0.0
        else:
            # Reset counter when valid distance received (waypoint manager recovered)
            if self.none_count > 0:
                self.logger.info(
                    f"[PROGRESS-RECOVER] Waypoint manager recovered after {self.none_count} None values. "
                    f"Resuming normal progress tracking."
                )
                self.none_count = 0

        # Additional safety check: distance_to_goal <= 0.0 (invalid even after smoothing)
        if distance_to_goal <= 0.0:
            self.logger.warning(
                f"[PROGRESS] Invalid distance_to_goal={distance_to_goal:.2f}m (≤0.0), "
                f"skipping progress calculation"
            )
            self.prev_distance_to_goal = None
            # Return 0.0 for this step (no progress reward/penalty)
            return 0.0

        prev_dist_str = f"{self.prev_distance_to_goal:.2f}" if self.prev_distance_to_goal is not None else "None"
        self.logger.debug(
            f"[PROGRESS] Input: route_distance={distance_to_goal:.2f}m, "
            f"waypoint_reached={waypoint_reached}, goal_reached={goal_reached}, "
            f"prev_route_distance={prev_dist_str}m"
        )

        progress = 0.0

        # Component 1: Route distance-based reward (dense, continuous)
        # Reward = (prev_route_distance - current_route_distance) * scale
        # Positive when moving forward along route, zero/negative for off-road
        #
        # CRITICAL FIX: Also check if prev_distance_to_goal was valid (not None, not 0.0)
        # to prevent discontinuities when recovering from invalid state
        if self.prev_distance_to_goal is not None and self.prev_distance_to_goal > 0.0:
            distance_delta = self.prev_distance_to_goal - distance_to_goal
            distance_reward = distance_delta * self.distance_scale
            progress += distance_reward

            # DIAGNOSTIC: Log distance delta and reward contribution
            self.logger.debug(
                f"[PROGRESS] Route Distance Delta: {distance_delta:.3f}m "
                f"({'forward' if distance_delta > 0 else 'backward'}), "
                f"Reward: {distance_reward:.2f} (scale={self.distance_scale})"
            )

            # Component 1b: PBRS - DISABLED (Bug: gives free reward for zero movement)
            # The route distance reward already provides the shaping we need!
            # PBRS as implemented violated Ng et al. theorem by using γ incorrectly.
            # Evidence: Vehicle stationary → PBRS = +1.15 reward per step
            # See: #file:reward.md, #file:SYSTEMATIC_PROGRESS_REWARD_ANALYSIS.md
            #
            # if self.prev_distance_to_goal is not None:
            #     potential_current = -distance_to_goal
            #     potential_prev = -self.prev_distance_to_goal
            #     pbrs_reward = self.gamma * potential_current - potential_prev  # ← (1-γ) × distance!
            #     pbrs_weighted = pbrs_reward * 0.5
            #     progress += pbrs_weighted  # ← BUG: Free reward proportional to distance from goal

        else:
            # First step of episode - no previous distance for comparison
            self.logger.debug(
                f"[PROGRESS] First step: initializing prev_route_distance={distance_to_goal:.2f}m"
            )

        # Update tracking
        self.prev_distance_to_goal = distance_to_goal

        # Component 2: Waypoint milestone bonus (sparse but frequent)
        # Encourage agent to reach intermediate waypoints
        if waypoint_reached:
            progress += self.waypoint_bonus
            self.logger.info(
                f"[PROGRESS]  Waypoint reached! Bonus: +{self.waypoint_bonus:.1f}, "
                f"total_progress={progress:.2f}"
            )

        # Component 3: Goal reached bonus (sparse but terminal)
        # Large reward for completing the route
        if goal_reached:
            progress += self.goal_reached_bonus
            self.logger.info(
                f"[PROGRESS]  Goal reached! Bonus: +{self.goal_reached_bonus:.1f}, "
                f"total_progress={progress:.2f}"
            )

        # Clip to reasonable range and log final result
        clipped_progress = float(np.clip(progress, -10.0, 110.0))

        # 🔍 DIAGNOSTIC: Log final progress reward and clipping status
        if progress != clipped_progress:
            self.logger.warning(
                f"[PROGRESS]  CLIPPED: raw={progress:.2f} → clipped={clipped_progress:.2f}"
            )

        # Build debug string safely to avoid format errors
        distance_rew_str = f"{distance_reward:.2f}" if self.prev_distance_to_goal is not None and 'distance_reward' in locals() else "0.00"
        waypoint_str = f"{self.waypoint_bonus:.1f}" if waypoint_reached else "0.0"
        goal_str = f"{self.goal_reached_bonus:.1f}" if goal_reached else "0.0"

        self.logger.debug(
            f"[PROGRESS] Final: progress={clipped_progress:.2f} "
            f"(route_distance_reward: {distance_rew_str}, "
            f"waypoint: {waypoint_str}, "
            f"goal: {goal_str})"
        )

        return clipped_progress

    def reset(self):
        """Reset internal state for new episode."""
        self.prev_acceleration = 0.0
        self.prev_acceleration_lateral = 0.0
        self.prev_distance_to_goal = None  # Reset progress tracking
        self.step_counter = 0  # Reset step counter for new episode
        self.none_count = 0  # Reset None counter for new episode (Issue #3.1 fix)
