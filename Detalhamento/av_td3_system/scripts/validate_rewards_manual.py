#!/usr/bin/env python3
"""
Manual Control Reward Validation Script for CARLA Environment

This script allows manual control of a vehicle in CARLA using keyboard input (WSAD)
while monitoring and validating the reward function in real-time.

Usage:
    python scripts/validate_rewards_manual.py --config config/baseline_config.yaml

Controls:
    W/S: Throttle/Brake
    A/D: Steering Left/Right
    Space: Hand Brake
    R: Reset Episode
    Q: Quit
    P: Pause/Resume reward logging
    1-5: Trigger specific test scenarios

Author: Based on CARLA documentation and Gymnasium best practices
"""

import argparse
import os
import sys
import time
import logging  # Added for debug logging support
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

import numpy as np
import pygame
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.carla_env import CARLANavigationEnv
from src.environment.reward_functions import RewardCalculator


@dataclass
class RewardSnapshot:
    """Stores a snapshot of reward components at a timestep."""
    timestamp: float
    step: int

    # State information
    velocity: float
    lateral_deviation: float
    heading_error: float
    distance_to_goal: float

    # Reward components
    total_reward: float
    efficiency_reward: float
    lane_keeping_reward: float
    comfort_penalty: float
    safety_penalty: float
    progress_reward: float

    # Additional context
    scenario_type: str
    user_notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RewardValidator:
    """Validates reward function behavior during manual control."""

    def __init__(self, env: CARLANavigationEnv, reward_func: RewardCalculator):
        self.env = env
        self.reward_func = reward_func
        self.snapshots: List[RewardSnapshot] = []
        self.start_time = time.time()
        self.step_counter = 0
        self.logging_enabled = True

    def capture_snapshot(
        self,
        state_info: Dict,
        reward_components: Dict,
        scenario_type: str = "normal",
        user_notes: str = ""
    ) -> RewardSnapshot:
        """Capture current reward state."""
        snapshot = RewardSnapshot(
            timestamp=time.time() - self.start_time,
            step=self.step_counter,
            velocity=state_info.get("velocity", 0.0),
            lateral_deviation=state_info.get("lateral_deviation", 0.0),
            heading_error=state_info.get("heading_error", 0.0),
            distance_to_goal=state_info.get("distance_to_goal", float('inf')),
            total_reward=reward_components.get("total", 0.0),
            efficiency_reward=reward_components.get("efficiency", 0.0),
            lane_keeping_reward=reward_components.get("lane_keeping", 0.0),
            comfort_penalty=reward_components.get("comfort", 0.0),
            safety_penalty=reward_components.get("safety", 0.0),
            progress_reward=reward_components.get("progress", 0.0),
            scenario_type=scenario_type,
            user_notes=user_notes
        )

        if self.logging_enabled:
            self.snapshots.append(snapshot)

        self.step_counter += 1
        return snapshot

    def save_validation_log(self, output_path: Path):
        """Save all snapshots to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_info": {
                "start_time": datetime.now().isoformat(),
                "total_steps": self.step_counter,
                "duration_seconds": time.time() - self.start_time
            },
            "snapshots": [s.to_dict() for s in self.snapshots]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n[VALIDATION] Saved {len(self.snapshots)} snapshots to {output_path}")

    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics from snapshots."""
        if not self.snapshots:
            return {}

        rewards = [s.total_reward for s in self.snapshots]
        velocities = [s.velocity for s in self.snapshots]
        lat_devs = [abs(s.lateral_deviation) for s in self.snapshots]

        return {
            "total_steps": len(self.snapshots),
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "reward_min": np.min(rewards),
            "reward_max": np.max(rewards),
            "avg_velocity_kmh": np.mean(velocities) * 3.6,
            "avg_lateral_deviation_m": np.mean(lat_devs),
            "max_lateral_deviation_m": np.max(lat_devs),
            "negative_rewards_pct": 100.0 * sum(1 for r in rewards if r < 0) / len(rewards)
        }


class ManualControlInterface:
    """PyGame-based manual control interface for CARLA vehicle."""

    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.display = pygame.display.set_mode(
            (width, height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA Reward Validation - Manual Control")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 14)
        self.font_small = pygame.font.SysFont('monospace', 12)

        # Control state
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.hand_brake = False
        self.reverse = False

        # Camera surface for rendering
        self.camera_surface = None

    def process_input(self) -> Tuple[Dict[str, any], bool, str]:
        """
        Process keyboard input and return control dict, quit flag, and command.

        Returns:
            - control_dict: Contains throttle, steer, brake, hand_brake, reverse
            - quit_flag: True if user wants to quit
            - command: Special command string ('reset', 'pause', 'scenario_X', etc.)
        """
        quit_flag = False
        command = ""

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flag = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit_flag = True
                elif event.key == pygame.K_r:
                    command = "reset"
                elif event.key == pygame.K_p:
                    command = "pause"
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    scenario_num = event.key - pygame.K_0
                    command = f"scenario_{scenario_num}"

        # Get continuous key states
        keys = pygame.key.get_pressed()

        # Throttle/Brake (W/S)
        if keys[pygame.K_w]:
            self.throttle = min(self.throttle + 0.02, 1.0)
            self.brake = 0.0
        elif keys[pygame.K_s]:
            # Handle reverse
            # If stopped and pressing S, enable reverse
            self.brake = min(self.brake + 0.05, 1.0)
            self.throttle = 0.0
        else:
            # Natural deceleration
            self.throttle *= 0.95
            self.brake *= 0.95

        # Steering (A/D)
        if keys[pygame.K_a]:
            self.steer = max(self.steer - 0.03, -1.0)
        elif keys[pygame.K_d]:
            self.steer = min(self.steer + 0.03, 1.0)
        else:
            # Return to center
            self.steer *= 0.85
            if abs(self.steer) < 0.01:
                self.steer = 0.0

        # Hand brake (Space)
        self.hand_brake = keys[pygame.K_SPACE]

        control_dict = {
            "throttle": self.throttle,
            "steer": self.steer,
            "brake": self.brake,
            "hand_brake": self.hand_brake,
            "reverse": self.reverse
        }

        return control_dict, quit_flag, command

    def render_hud(
        self,
        snapshot: Optional[RewardSnapshot],
        summary_stats: Dict,
        logging_enabled: bool,
        goal_reached: bool = False  # ADDED for Issue #4
    ):
        """Render heads-up display with reward information."""
        y_offset = 10
        line_height = 18

        def draw_text(text: str, color=(255, 255, 255)):
            nonlocal y_offset
            surface = self.font.render(text, True, color)
            self.display.blit(surface, (10, y_offset))
            y_offset += line_height

        # CRITICAL FIX (Nov 23, 2025): Issue #4 - Route Completion Reward Visibility
        # ============================================================================
        # Display prominent banner when goal is reached to show +100 bonus is applied
        if goal_reached:
            # Large font for celebration message
            big_font = pygame.font.Font(None, 48)
            banner_surface = big_font.render("GOAL REACHED!", True, (0, 255, 0))
            banner_rect = banner_surface.get_rect(center=(self.width // 2, 100))
            # Draw semi-transparent background for banner
            bg_rect = pygame.Rect(0, 80, self.width, 60)
            bg_surface = pygame.Surface((self.width, 60))
            bg_surface.set_alpha(200)
            bg_surface.fill((0, 0, 0))
            self.display.blit(bg_surface, (0, 80))
            # Draw banner text
            self.display.blit(banner_surface, banner_rect)

            # Show reward bonus notification
            bonus_surface = self.font.render("+100.0 Route Completion Bonus!", True, (255, 255, 0))
            bonus_rect = bonus_surface.get_rect(center=(self.width // 2, 150))
            self.display.blit(bonus_surface, bonus_rect)

            # Adjust y_offset to not overlap banner
            y_offset = 180

        # Title
        draw_text("=== CARLA Reward Validation ===", (0, 255, 255))
        draw_text("")

        # Controls
        draw_text("Controls: W/S=Throttle/Brake, A/D=Steer, SPACE=Handbrake", (200, 200, 200))
        draw_text("          R=Reset, P=Pause Logging, Q=Quit, 1-5=Scenarios", (200, 200, 200))
        draw_text("")

        # Current control state
        draw_text(f"Throttle: {self.throttle:5.2f}  Steer: {self.steer:6.2f}  Brake: {self.brake:5.2f}", (255, 255, 0))
        draw_text("")

        # Logging status
        status_color = (0, 255, 0) if logging_enabled else (255, 0, 0)
        draw_text(f"Logging: {'ENABLED' if logging_enabled else 'PAUSED'}", status_color)
        draw_text("")

        # Current reward components (if available)
        if snapshot:
            draw_text("--- Current State ---", (0, 255, 255))
            draw_text(f"Step: {snapshot.step:6d}  Time: {snapshot.timestamp:7.2f}s", (255, 255, 255))
            draw_text(f"Velocity: {snapshot.velocity * 3.6:6.2f} km/h", (255, 255, 255))
            draw_text(f"Lat Dev:  {snapshot.lateral_deviation:6.3f} m", (255, 255, 255))
            draw_text(f"Head Err: {np.rad2deg(snapshot.heading_error):6.2f} deg", (255, 255, 255))
            draw_text("")

            draw_text("--- Reward Components ---", (0, 255, 255))

            # Color code rewards (green positive, red negative)
            def reward_color(value):
                return (0, 255, 0) if value >= 0 else (255, 0, 0)

            draw_text(f"Total:        {snapshot.total_reward:8.4f}", reward_color(snapshot.total_reward))
            draw_text(f"  Efficiency: {snapshot.efficiency_reward:8.4f}", reward_color(snapshot.efficiency_reward))
            draw_text(f"  Lane Keep:  {snapshot.lane_keeping_reward:8.4f}", reward_color(snapshot.lane_keeping_reward))
            draw_text(f"  Comfort:    {snapshot.comfort_penalty:8.4f}", reward_color(snapshot.comfort_penalty))
            draw_text(f"  Safety:     {snapshot.safety_penalty:8.4f}", reward_color(snapshot.safety_penalty))
            # CRITICAL FIX: Highlight progress with special color when goal reached
            progress_color = (255, 255, 0) if goal_reached else reward_color(snapshot.progress_reward)
            draw_text(f"  Progress:   {snapshot.progress_reward:8.4f}", progress_color)
            if goal_reached:
                draw_text(f"    (includes +100 goal bonus!)", (255, 255, 0))
            draw_text("")

            if snapshot.scenario_type != "normal":
                draw_text(f"Scenario: {snapshot.scenario_type}", (255, 255, 0))
                draw_text("")

        # Summary statistics
        if summary_stats:
            draw_text("--- Session Summary ---", (0, 255, 255))
            draw_text(f"Total Steps:  {summary_stats.get('total_steps', 0)}", (200, 200, 200))
            draw_text(f"Reward Mean:  {summary_stats.get('reward_mean', 0.0):7.4f}", (200, 200, 200))
            draw_text(f"Reward Std:   {summary_stats.get('reward_std', 0.0):7.4f}", (200, 200, 200))
            draw_text(f"Avg Speed:    {summary_stats.get('avg_velocity_kmh', 0.0):6.2f} km/h", (200, 200, 200))
            draw_text(f"Avg Lat Dev:  {summary_stats.get('avg_lateral_deviation_m', 0.0):6.3f} m", (200, 200, 200))

    def update_camera(self, image_data: Optional[np.ndarray]):
        """Update camera display from environment."""
        if image_data is not None and len(image_data.shape) == 3:
            # image_data is expected to be HxWxC in range [0, 255]
            # Ensure uint8 type for pygame
            if image_data.dtype != np.uint8:
                # Assume it's normalized [0, 1] and convert to [0, 255]
                image_data = (image_data * 255).astype(np.uint8)

            # pygame.surfarray expects WxHxC (width x height x channels)
            surface = pygame.surfarray.make_surface(image_data.swapaxes(0, 1))

            # Calculate scaling to fit camera in top portion while preserving aspect ratio
            # Use 70% of window height for camera, 30% for HUD
            camera_height = int(self.height * 0.7)
            img_aspect = image_data.shape[1] / image_data.shape[0]  # width/height
            camera_width = int(camera_height * img_aspect)

            # Center camera horizontally if narrower than window
            if camera_width > self.width:
                camera_width = self.width
                camera_height = int(camera_width / img_aspect)

            self.camera_surface = pygame.transform.smoothscale(surface, (camera_width, camera_height))
            self.camera_x_offset = (self.width - camera_width) // 2  # Center horizontally

    def render(self, snapshot: Optional[RewardSnapshot], summary_stats: Dict, logging_enabled: bool, goal_reached: bool = False):
        """Render complete frame."""
        # Black background
        self.display.fill((0, 0, 0))

        # Render camera (centered, top 70% of window)
        if self.camera_surface:
            x_offset = getattr(self, 'camera_x_offset', 0)
            self.display.blit(self.camera_surface, (x_offset, 0))

        # Render HUD (bottom portion overlay) with goal reached flag
        self.render_hud(snapshot, summary_stats, logging_enabled, goal_reached)

        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS

    def cleanup(self):
        """Cleanup PyGame resources."""
        pygame.quit()


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def trigger_test_scenario(scenario_num: int, env: CARLANavigationEnv) -> str:
    """
    Trigger specific test scenarios to validate edge cases.

    Scenarios:
        1: Sharp turn (high lateral deviation expected)
        2: Emergency brake (high comfort penalty expected)
        3: Off-road (high safety penalty expected)
        4: Speeding (efficiency vs safety trade-off)
        5: Lane invasion (safety penalty expected)
    """
    scenarios = {
        1: ("sharp_turn", "Sharp turn scenario - expect high lateral deviation"),
        2: ("emergency_brake", "Emergency brake - expect comfort penalty"),
        3: ("off_road", "Off-road scenario - expect safety penalty"),
        4: ("speeding", "Speeding scenario - efficiency vs safety"),
        5: ("lane_invasion", "Lane invasion - safety penalty")
    }

    if scenario_num in scenarios:
        scenario_type, description = scenarios[scenario_num]
        print(f"\n[SCENARIO {scenario_num}] {description}")
        return scenario_type

    return "normal"


def main():
    parser = argparse.ArgumentParser(description="Manual control reward validation for CARLA")
    parser.add_argument(
        "--config",
        type=str,
        default="config/baseline_config.yaml",
        help="Path to environment configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_logs",
        help="Directory to save validation logs"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps before auto-reset (None = unlimited)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for debug output (use DEBUG to see waypoint blending logs)"
    )

    args = parser.parse_args()

    # Configure logging with specified level
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        force=True  # Force reconfiguration even if logging was already initialized
    )

    # Explicitly set log level for all relevant modules (handles import-time logger creation)
    log_level = getattr(logging, args.log_level)
    logging.getLogger('src.environment.waypoint_manager').setLevel(log_level)
    logging.getLogger('src.environment.reward_functions').setLevel(log_level)
    logging.getLogger('src.environment.carla_env').setLevel(log_level)
    logging.getLogger('src.environment.sensors').setLevel(log_level)

    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging level set to: {args.log_level}")
    if args.log_level == "DEBUG":
        logger.info("DEBUG mode enabled - you will see waypoint blending diagnostic logs")
        logger.debug("Explicitly configured loggers: waypoint_manager, reward_functions, carla_env, sensors")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("CARLA Reward Validation - Manual Control Mode")
    print("="*70)
    print(f"\nConfiguration: {config_path}")
    print(f"Output Directory: {output_dir}")
    print("\nInitializing CARLA environment...")

    # Initialize environment
    env = None
    interface = None
    validator = None

    try:
        # Create environment with required config paths
        # CARLANavigationEnv needs three separate config files
        env = CARLANavigationEnv(
            carla_config_path="config/carla_config.yaml",
            td3_config_path="config/td3_config.yaml",
            training_config_path="config/training_config.yaml",
            headless=False  # Enable rendering for manual control
        )

        # Override max_episode_steps if user specified higher limit
        # This is necessary because config files may have lower default (e.g., 1000)
        if args.max_steps and args.max_steps > env.max_episode_steps:
            print(f"[INFO] Overriding environment max_episode_steps: {env.max_episode_steps} → {args.max_steps}")
            env.max_episode_steps = args.max_steps

        reward_func = env.reward_calculator

        # Create validator
        validator = RewardValidator(env, reward_func)

        # Create manual control interface
        interface = ManualControlInterface()

        # Reset environment
        obs, info = env.reset()

        print("\n[READY] Manual control active. Use WSAD keys to drive.")
        print("[INFO] Press 'P' to pause/resume logging, 'R' to reset, 'Q' to quit")
        print("[INFO] Press 1-5 to trigger test scenarios\n")

        current_scenario = "normal"
        step_in_episode = 0

        # Main control loop
        running = True
        while running:
            # Process user input
            control_dict, quit_flag, command = interface.process_input()

            if quit_flag:
                running = False
                break

            # Handle commands
            if command == "reset":
                print(f"\n[RESET] Episode reset at step {step_in_episode}")
                obs, info = env.reset()
                step_in_episode = 0
                current_scenario = "normal"
                continue
            elif command == "pause":
                validator.logging_enabled = not validator.logging_enabled
                status = "ENABLED" if validator.logging_enabled else "PAUSED"
                print(f"\n[LOGGING] {status}")
                continue
            elif command.startswith("scenario_"):
                scenario_num = int(command.split("_")[1])
                current_scenario = trigger_test_scenario(scenario_num, env)

            # Convert control dict to action (normalized [-1, 1])
            action = np.array([
                control_dict["steer"],  # steering
                control_dict["throttle"] - control_dict["brake"]  # throttle/brake combined
            ], dtype=np.float32)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_in_episode += 1

            # Get reward components from info
            reward_components = info.get("reward_components", {})

            # Get state information
            state_info = info.get("state", {})

            # CRITICAL FIX (Nov 23, 2025): Issue #4 - Extract goal_reached flag
            goal_reached = info.get("goal_reached", False)

            # Capture snapshot
            snapshot = validator.capture_snapshot(
                state_info=state_info,
                reward_components=reward_components,
                scenario_type=current_scenario
            )

            # Get summary stats
            summary_stats = validator.get_summary_stats()

            # Update camera display from observation
            if isinstance(obs, dict) and 'image' in obs:
                # Get raw RGB frame from sensor (not preprocessed grayscale)
                rgb_frame = env.sensors.get_rgb_camera_frame()
                if rgb_frame is not None:
                    interface.update_camera(rgb_frame)
                else:
                    # Fallback: convert grayscale to RGB for display
                    latest_frame = obs['image'][-1]  # Last frame from stack (84, 84)
                    # Denormalize from [-1, 1] to [0, 255]
                    frame_uint8 = ((latest_frame + 1.0) * 127.5).astype(np.uint8)
                    rgb_frame = np.stack([frame_uint8] * 3, axis=-1)
                    interface.update_camera(rgb_frame)

            # Render interface with goal_reached flag
            interface.render(snapshot, summary_stats, validator.logging_enabled, goal_reached)

            # Check for episode end
            if terminated or truncated:
                reason = info.get("termination_reason", "unknown")
                print(f"\n[EPISODE END] Reason: {reason}, Steps: {step_in_episode}")

                # Auto-reset on safety violations (collision, off-road)
                safety_violations = ["collision", "off_road", "lane_invasion"]
                if reason in safety_violations:
                    print(f"[AUTO-RESET] Safety violation detected ({reason}), resetting in 2 seconds...")
                    time.sleep(2)  # Brief pause to show frozen frame
                    obs, info = env.reset()
                    step_in_episode = 0
                    current_scenario = "normal"
                    print("[READY] Episode reset. Continue driving with WSAD.")
                else:
                    # Manual reset for other termination reasons (e.g., goal reached)
                    # CRITICAL FIX: Check if goal was reached for special celebration
                    goal_just_reached = (reason == "route_completed" and info.get("goal_reached", False))
                    if goal_just_reached:
                        print("[SUCCESS] GOAL REACHED! +100 bonus applied to progress reward!")
                    print("[INFO] Press 'R' to reset or 'Q' to quit")

                    # Wait for user command
                    waiting = True
                    while waiting and running:
                        _, quit_flag, command = interface.process_input()
                        if quit_flag:
                            running = False
                            waiting = False
                        elif command == "reset":
                            obs, info = env.reset()
                            step_in_episode = 0
                            current_scenario = "normal"
                            waiting = False

                        # Keep rendering while waiting (show goal celebration if applicable)
                        interface.render(snapshot, summary_stats, validator.logging_enabled, goal_just_reached)

            # Check max steps
            if args.max_steps and step_in_episode >= args.max_steps:
                print(f"\n[MAX STEPS] Reached {args.max_steps} steps, resetting...")
                obs, info = env.reset()
                step_in_episode = 0
                current_scenario = "normal"

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Shutting down...")

    except Exception as e:
        print(f"\n\n[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Save validation log
        if validator and validator.snapshots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = output_dir / f"reward_validation_{timestamp}.json"
            validator.save_validation_log(log_path)

            # Print summary
            summary = validator.get_summary_stats()
            print("\n" + "="*70)
            print("VALIDATION SESSION SUMMARY")
            print("="*70)
            for key, value in summary.items():
                print(f"{key:30s}: {value}")
            print("="*70)

        # Cleanup
        if interface:
            interface.cleanup()

        if env:
            env.close()

        print("\n[DONE] Validation session completed.\n")


if __name__ == "__main__":
    main()
