# CARLA Environment Validation System

This directory contains tools for validating the CARLA RL environment wrapper implementation. The validation system tests all aspects of the environment wrapper to ensure it functions correctly before using it for reinforcement learning experiments.

## Quick Start

1. Set up dependencies:

   ```bash
   python setup_validation.py
   ```

2. Prepare CARLA for validation:

   ```bash
   python prepare_carla_for_validation.py
   ```

3. Run basic validation:

   ```bash
   python run_validation.py
   ```

4. View results:

   ```bash
   python view_validation_results.py --results validation_results_*.json --verbose
   ```

5. Generate a detailed report:

   ```bash
   python generate_validation_report.py --results validation_results_*.json --template validation_report_template.md
   ```

## Components

- **environment_validator.py**: Main validator class with test methods
- **run_validation.py**: Simple script to run validation tests
- **run_advanced_validation.py**: Advanced validation with iterations and customization
- **prepare_carla_for_validation.py**: Setup CARLA for validation tests
- **view_validation_results.py**: Display validation results in readable format
- **generate_validation_report.py**: Generate detailed markdown reports
- **compare_validation_results.py**: Compare results across multiple runs
- **setup_validation.py**: Install required dependencies

## Advanced Usage

### Running Specific Tests

```bash
python run_advanced_validation.py --tests init,reset,action
```

Available test categories:

- `init`: Environment initialization tests
- `reset`: Environment reset tests
- `action`: Action space and response tests
- `step`: Step function tests
- `reward`: Reward function tests
- `termination`: Episode termination tests
- `cleanup`: Resource cleanup tests

### Multiple Iterations

Test stability with multiple iterations:

```bash
python run_advanced_validation.py --iterations 5
```

### Generating Reports

```bash
python run_advanced_validation.py --tests all --generate-report
```

### Comparing Multiple Runs

```bash
python compare_validation_results.py --results run1_results.json run2_results.json --output comparison_report.md
```

## Test Descriptions

### Initialization Tests

- **init_environment_creation**: Tests basic environment wrapper creation
- **init_carla_connection**: Tests connection to CARLA server
- **init_action_space**: Verifies action space configuration
- **init_observation_space**: Verifies observation space configuration

### Reset Tests

- **reset_initial_state**: Verifies initial state format after reset
- **reset_vehicle_position**: Checks if vehicle is positioned correctly after reset
- **reset_consistent_outputs**: Checks consistency of reset outputs across multiple calls

### Action Tests

- **action_throttle_response**: Verifies vehicle response to throttle actions
- **action_steering_response**: Verifies vehicle response to steering actions
- **action_continuous_validity**: Tests handling of continuous action values

### Step Tests

- **step_observation_format**: Verifies observation format returned by step
- **step_reward_range**: Checks if rewards are within expected ranges
- **step_done_flag**: Verifies done flag is set correctly on termination conditions
- **step_info_contents**: Verifies info dictionary contains required information

### Reward Tests

- **reward_progress_component**: Tests progress component of reward function
- **reward_collision_penalty**: Tests collision penalties
- **reward_consistency**: Checks consistency of reward calculation

### Termination Tests

- **termination_collision_detection**: Verifies episode termination on collision
- **termination_goal_detection**: Verifies episode termination on goal achievement

### Cleanup Tests

- **cleanup_resources**: Verifies proper resource cleanup on environment close

## Performance Benchmarks

The validation system tracks performance metrics including:

- Reset time (mean, std, min, max)
- Step time (mean, std, min, max)
- Memory usage

## Creating Custom Tests

To add a new test, add a method to the `EnvironmentValidator` class in `environment_validator.py`:

```python
def new_test_name(self):
    """
    Test description.

    Returns:
        Dict with "passed" boolean and "details" dictionary
    """
    # Test implementation
    success = True  # Set based on test result

    return {
        "passed": success,
        "details": {
            "message": "Test result message if passed",
            "error": "Error description if failed"
        },
        "duration": time_taken
    }
```

Then add the test to the appropriate category in `TEST_CATEGORIES` in `run_advanced_validation.py`.
