# CARLA DRL Evaluation Scenarios

## Baseline Evaluation Scenarios

### Scenario 1: Town01 Basic Lane Following
**Objective**: Test basic lane-keeping and speed control in simple road geometry
- **Map**: Town01 (simple figure-8 layout)
- **Weather**: Clear day (cloudiness: 10%, precipitation: 0%)
- **Traffic**: 5 vehicles, 3 pedestrians
- **Route**: Full lap around Town01 circuit (~800m)
- **Success Criteria**: 
  - Complete route without collisions
  - Average lane deviation < 1.0m
  - Average speed > 15 km/h
  - Max 1 lane invasion event

### Scenario 2: Town02 Urban Intersections
**Objective**: Test intersection navigation and traffic signal compliance
- **Map**: Town02 (urban grid with intersections)
- **Weather**: Cloudy afternoon (cloudiness: 60%, sun_altitude: 30°)
- **Traffic**: 15 vehicles, 8 pedestrians
- **Route**: Multi-intersection traverse (~1200m)
- **Success Criteria**:
  - Navigate 4+ intersections successfully
  - Stop at red traffic lights (compliance > 90%)
  - No collisions with traffic or pedestrians
  - Route completion within 300 seconds

### Scenario 3: Town03 Complex Urban Navigation
**Objective**: Test complex urban driving with varied road types
- **Map**: Town03 (mixed residential/commercial)
- **Weather**: Light rain (cloudiness: 80%, precipitation: 30%)
- **Traffic**: 20 vehicles, 12 pedestrians
- **Route**: Complex path with curves, hills, intersections (~1500m)
- **Success Criteria**:
  - Handle varying road curvature and elevation
  - Maintain safe following distance (>2s @ current speed)
  - Successful overtaking maneuvers (if applicable)
  - Complete route with <2 lane invasions

## Evaluation Metrics

### Primary Metrics
1. **Route Completion Rate**: Percentage of routes completed successfully
2. **Collision Rate**: Number of collisions per 1000 steps
3. **Lane Keeping Performance**: 
   - Average lateral deviation from lane center (m)
   - Percentage of time within lane boundaries
4. **Speed Consistency**: 
   - Adherence to speed limits (±10% tolerance)
   - Smooth acceleration/deceleration profiles

### Secondary Metrics
5. **Traffic Signal Compliance**: Percentage of correct stops at red lights
6. **Comfort Metrics**:
   - Maximum lateral acceleration (m/s²)
   - Maximum longitudinal acceleration (m/s²)
   - Steering smoothness (steering rate change)
7. **Efficiency Metrics**:
   - Time to complete route
   - Fuel consumption equivalent
   - Unnecessary stopping events

### Safety Metrics
8. **Time to Collision (TTC)**: Minimum TTC with other vehicles/pedestrians
9. **Safety Margins**: Distance to lane boundaries, other vehicles
10. **Emergency Events**: Hard braking, swerving maneuvers

## Evaluation Scorecard

| Metric | Weight | Town01 Target | Town02 Target | Town03 Target | Pass Threshold |
|--------|--------|---------------|---------------|---------------|----------------|
| Route Completion Rate | 25% | ≥95% | ≥90% | ≥85% | ≥80% |
| Collision Rate (per 1000 steps) | 20% | <0.5 | <1.0 | <1.5 | <2.0 |
| Lane Keeping (avg deviation) | 15% | <0.8m | <1.0m | <1.2m | <1.5m |
| Speed Compliance | 10% | ≥90% | ≥85% | ≥80% | ≥75% |
| Traffic Signal Compliance | 10% | ≥95% | ≥90% | ≥85% | ≥80% |
| Comfort Score (composite) | 10% | ≥8.0/10 | ≥7.0/10 | ≥6.0/10 | ≥5.0/10 |
| Minimum TTC | 5% | ≥3.0s | ≥2.5s | ≥2.0s | ≥1.5s |
| Lane Invasions | 5% | ≤1 | ≤2 | ≤3 | ≤5 |

### Scoring Formula
```
Overall Score = Σ(Metric_Score × Weight) × 100

Where Metric_Score = min(1.0, Actual_Value / Target_Value) for positive metrics
                   = min(1.0, Target_Value / Actual_Value) for negative metrics
```

### Success Thresholds
- **Excellent**: Overall Score ≥ 90%
- **Good**: Overall Score ≥ 80%
- **Acceptable**: Overall Score ≥ 70%
- **Needs Improvement**: Overall Score < 70%

## Detailed Test Configuration

### Town01 Configuration
```yaml
scenario_name: "town01_basic_lane_following"
carla_settings:
  town: "Town01"
  weather:
    cloudiness: 10.0
    precipitation: 0.0
    sun_altitude_angle: 70.0
    fog_density: 0.0
  synchronous_mode: true
  fixed_delta_seconds: 0.02

traffic_settings:
  num_vehicles: 5
  num_pedestrians: 3
  vehicle_spawn_radius: 100.0
  pedestrian_spawn_radius: 50.0

route_settings:
  start_location: [107.0, 133.0, 0.5]
  target_location: [140.0, 207.0, 0.5]
  waypoint_spacing: 2.0

evaluation_settings:
  max_episode_steps: 2000
  timeout_seconds: 300
  success_distance_threshold: 5.0
```

### Town02 Configuration
```yaml
scenario_name: "town02_urban_intersections"
carla_settings:
  town: "Town02"
  weather:
    cloudiness: 60.0
    precipitation: 0.0
    sun_altitude_angle: 30.0
    fog_density: 0.0

traffic_settings:
  num_vehicles: 15
  num_pedestrians: 8
  vehicle_spawn_radius: 150.0
  pedestrian_spawn_radius: 75.0

route_settings:
  start_location: [38.0, 4.0, 0.5]
  target_location: [140.0, 180.0, 0.5]
  include_intersections: true
  traffic_lights_enabled: true

evaluation_settings:
  max_episode_steps: 3000
  timeout_seconds: 400
  intersection_count_target: 4
```

### Town03 Configuration
```yaml
scenario_name: "town03_complex_urban"
carla_settings:
  town: "Town03"
  weather:
    cloudiness: 80.0
    precipitation: 30.0
    sun_altitude_angle: 45.0
    fog_density: 5.0

traffic_settings:
  num_vehicles: 20
  num_pedestrians: 12
  vehicle_spawn_radius: 200.0
  pedestrian_spawn_radius: 100.0

route_settings:
  start_location: [0.0, 0.0, 0.5]
  target_location: [150.0, 150.0, 0.5]
  include_hills: true
  include_curves: true
  complex_routing: true

evaluation_settings:
  max_episode_steps: 4000
  timeout_seconds: 500
  elevation_change_tolerance: 5.0
```

## Automated Evaluation Pipeline

### Evaluation Script Structure
```python
class CarlaEvaluationSuite:
    """Automated evaluation suite for CARLA DRL agents."""
    
    def __init__(self, config_path: str):
        self.scenarios = self.load_scenarios(config_path)
        self.metrics_calculator = MetricsCalculator()
        self.report_generator = ReportGenerator()
    
    def run_evaluation(self, model_path: str, num_runs: int = 5):
        """Run complete evaluation across all scenarios."""
        results = {}
        
        for scenario in self.scenarios:
            scenario_results = []
            for run in range(num_runs):
                result = self.run_single_scenario(model_path, scenario)
                scenario_results.append(result)
            
            results[scenario.name] = self.aggregate_results(scenario_results)
        
        return self.generate_report(results)
    
    def calculate_composite_score(self, results: Dict) -> float:
        """Calculate overall performance score."""
        weighted_scores = []
        
        for scenario_name, metrics in results.items():
            scenario_score = self.calculate_scenario_score(metrics)
            weighted_scores.append(scenario_score)
        
        return np.mean(weighted_scores)
```

## Continuous Integration Evaluation

### Automated Testing Pipeline
1. **Pre-commit Hooks**: Basic syntax and configuration validation
2. **Nightly Testing**: Run Town01 basic scenario (fast smoke test)
3. **Weekly Testing**: Full evaluation suite across all scenarios
4. **Release Testing**: Extended evaluation with statistical significance

### Performance Regression Detection
- Compare against baseline performance metrics
- Alert on >5% degradation in key safety metrics
- Automatic rollback triggers for collision rate increases

### Evaluation Data Logging
- ROS bag recording for each evaluation run
- TensorBoard metrics logging during evaluation
- Automatic video recording of critical scenarios
- Performance profiling data for optimization
