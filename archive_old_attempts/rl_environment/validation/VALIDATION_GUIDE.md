# CARLA Environment Validation Guide

This guide provides detailed examples of how to use the validation system to test the CARLA environment wrapper.

## Basic Usage

### Setting Up

Before running any validation tests, ensure you have the necessary dependencies installed:

```bash
python setup_validation.py
```

This will install required packages such as matplotlib, numpy, pandas, and tabulate.

### Starting CARLA

The validation system needs a running CARLA server. You can start it with:

```bash
python prepare_carla_for_validation.py
```

This will configure CARLA with appropriate settings for validation.

### Running Basic Validation

To run a basic validation suite:

```bash
python run_validation.py
```

This will execute all validation tests and generate a JSON results file.

### Viewing Results

To view the validation results in a human-readable format:

```bash
python view_validation_results.py --results validation_results_YYYYMMDD_HHMMSS.json --verbose
```

### Generating a Report

To generate a detailed report:

```bash
python generate_validation_report.py --results validation_results_YYYYMMDD_HHMMSS.json
```

## Advanced Usage Examples

### Running Specific Test Categories

Test only initialization and reset functionality:

```bash
python run_advanced_validation.py --tests init,reset
```

### Testing with Multiple Iterations

Test stability by running each test multiple times:

```bash
python run_advanced_validation.py --tests all --iterations 5
```

### Setting a Timeout

Limit the total validation time:

```bash
python run_advanced_validation.py --tests all --timeout 300
```

### Specifying CARLA Connection

Connect to a specific CARLA server:

```bash
python run_advanced_validation.py --host localhost --port 2000
```

### Automated Full Validation

Run the complete validation workflow:

```powershell
# On Windows
.\run_full_validation.ps1
```

## Comparing Multiple Validation Runs

To track improvement over time or compare different environment configurations:

```bash
python compare_validation_results.py --results run1_results.json run2_results.json --output comparison.md
```

This will generate a markdown report with charts showing differences between runs.

## Common Issues and Solutions

### CARLA Connection Problems

If validation fails with "CARLA server not available":

1. Check if CARLA is running
2. Verify the port matches (default: 2000)
3. Ensure no firewall is blocking the connection

### Resource Cleanup Issues

If you see warnings about resources not being released:

1. Check the `cleanup_resources` test results
2. Ensure the environment's `close()` method properly releases all resources
3. Look for orphaned CARLA client connections

### Inconsistent Results

If tests pass sometimes but fail other times:

1. Increase the number of iterations: `--iterations 10`
2. Check for race conditions in the environment implementation
3. Examine the CARLA simulator for non-deterministic behavior

## Custom Test Development

To create a custom test for a specific feature:

1. Add a new test method to `environment_validator.py`:

```python
def custom_feature_test(self):
    """
    Test custom feature functionality.
    """
    try:
        # Test implementation
        result = self.env.custom_feature()
        success = result == expected_value

        if success:
            return {
                "passed": True,
                "details": {
                    "message": "Custom feature works correctly"
                },
                "duration": 0.5
            }
        else:
            return {
                "passed": False,
                "details": {
                    "error": f"Expected {expected_value}, got {result}"
                },
                "duration": 0.5
            }
    except Exception as e:
        return {
            "passed": False,
            "details": {
                "error": f"Exception occurred: {str(e)}"
            },
            "duration": 0.5
        }
```

1. Add the test to the appropriate category in `run_advanced_validation.py`:

```python
TEST_CATEGORIES = {
    # Existing categories...
    "custom": ["custom_feature_test"]
}
```

1. Run your custom test:

```bash
python run_advanced_validation.py --tests custom
```

## Performance Benchmarking

To benchmark environment performance:

```bash
python run_advanced_validation.py --tests all --iterations 20
```

The results will include performance metrics such as:

- Average reset time
- Average step time
- Memory usage

These metrics help identify performance bottlenecks and track improvements over time.
