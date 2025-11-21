(base) danielterra@danielterra:/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system$ cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && python3 << 'EOPYTHON'
> import math
>
> # From the logs:
> distance_to_goal_current = 229.42  # meters
> distance_to_goal_prev = 229.42     # meters (NO MOVEMENT!)
> gamma = 0.99
> pbrs_weight = 0.5
>
> # PBRS calculation from code:
> potential_current = -distance_to_goal_current
> potential_prev = -distance_to_goal_prev
> pbrs_reward = gamma * potential_current - potential_prev
> pbrs_weighted = pbrs_reward * pbrs_weight
>
> print('=' * 70)
> print('PBRS FORMULA VERIFICATION (From Logs)')
> print('=' * 70)
> print(f'\nInputs:')
> print(f'  distance_to_goal (current): {distance_to_goal_current:.2f}m')
> print(f'  distance_to_goal (previous): {distance_to_goal_prev:.2f}m')
> print(f'  Movement: {distance_to_goal_prev - distance_to_goal_current:.3f}m ← ZERO!')
> print(f'  γ (gamma): {gamma}')
> print(f'  PBRS weight: {pbrs_weight}')
>
> print(f'\nPBRS Calculation:')
> print(f'  Φ(s\'): {potential_current:.3f}')
> print(f'  Φ(s):  {potential_prev:.3f}')
> print(f'  F(s,s\') = γ×Φ(s\') - Φ(s)')
> print(f'           = {gamma} × ({potential_current}) - ({potential_prev})')
> print(f'           = {gamma * potential_current:.3f} - ({potential_prev:.3f})')
> print(f'           = {pbrs_reward:.3f}')
> print(f'  Weighted: {pbrs_reward:.3f} × {pbrs_weight} = {pbrs_weighted:.3f}')
>
> print(f'\n🔥 PROBLEM: PBRS gives +{pbrs_weighted:.3f} reward for ZERO movement!')
> print(f'\nWhy? Because:')
> print(f'  γ×Φ(s\') = {gamma} × {potential_current} = {gamma * potential_current:.3f}')
> print(f'  Φ(s) = {potential_prev:.3f}')
> print(f'  Difference = {gamma * potential_current:.3f} - ({potential_prev:.3f})')
> print(f'             = {gamma * potential_current - potential_prev:.3f}')
> print(f'\nThis equals (1-γ) × distance_to_goal:')
> print(f'  (1-γ) × distance = {1-gamma} × {distance_to_goal_current} = {(1-gamma) * distance_to_goal_current:.3f}')
> print(f'\n📊 PBRS is giving FREE REWARD proportional to distance from goal!')
> print(f'   Further from goal = MORE free reward per step!')
>
> print('\n' + '=' * 70)
> print('LANE INVASION PENALTY ANALYSIS')
> print('=' * 70)
>
> # Scenario: Vehicle turns right off-road
> distance_scale = 50.0
> efficiency_weight = 3.0
> lane_invasion_penalty = -10.0  # From config
> safety_weight = 1.0  # Fixed value
>
> print('\nScenario: Vehicle turns right 0.3m (Euclidean reduction)')
> euclidean_reduction = 0.3  # meters
> distance_reward = euclidean_reduction * distance_scale
> weighted_distance = distance_reward * efficiency_weight
> pbrs_at_new_position = 1.147  # Roughly same (still far from goal)
>
> progress_total = weighted_distance + pbrs_at_new_position
>
> print(f'\n  Progress Reward:')
> print(f'    Distance reward: {euclidean_reduction}m × {distance_scale} × {efficiency_weight} = +{weighted_distance:.2f}')
> print(f'    PBRS reward: ~{pbrs_at_new_position:.2f}')
> print(f'    TOTAL PROGRESS: +{progress_total:.2f}')
>
> print(f'\n  Safety Penalty (if lane invasion detected):')
> print(f'    Lane invasion: {lane_invasion_penalty:.2f}')
> print(f'    Weighted: {lane_invasion_penalty} × {safety_weight} = {lane_invasion_penalty * safety_weight:.2f}')
>
> net_reward = progress_total + (lane_invasion_penalty * safety_weight)
> print(f'\n  NET REWARD: {progress_total:.2f} + ({lane_invasion_penalty * safety_weight:.2f}) = {net_reward:.2f}')
>
> if net_reward > 0:
>     print(f'\n  ✅ STILL PROFITABLE! Off-road shortcut gives +{net_reward:.2f} net reward!')
> else:
>     print(f'\n  ❌ UNPROFITABLE: Net penalty of {net_reward:.2f}')
>
> EOPYTHON
======================================================================
PBRS FORMULA VERIFICATION (From Logs)
======================================================================

Inputs:
  distance_to_goal (current): 229.42m
  distance_to_goal (previous): 229.42m
  Movement: 0.000m ← ZERO!
  γ (gamma): 0.99
  PBRS weight: 0.5

PBRS Calculation:
  Φ(s'): -229.420
  Φ(s):  -229.420
  F(s,s') = γ×Φ(s') - Φ(s)
           = 0.99 × (-229.42) - (-229.42)
           = -227.126 - (-229.420)
           = 2.294
  Weighted: 2.294 × 0.5 = 1.147

🔥 PROBLEM: PBRS gives +1.147 reward for ZERO movement!

Why? Because:
  γ×Φ(s') = 0.99 × -229.42 = -227.126
  Φ(s) = -229.420
  Difference = -227.126 - (-229.420)
             = 2.294

This equals (1-γ) × distance_to_goal:
  (1-γ) × distance = 0.010000000000000009 × 229.42 = 2.294

📊 PBRS is giving FREE REWARD proportional to distance from goal!
   Further from goal = MORE free reward per step!

======================================================================
LANE INVASION PENALTY ANALYSIS
======================================================================

Scenario: Vehicle turns right 0.3m (Euclidean reduction)

  Progress Reward:
    Distance reward: 0.3m × 50.0 × 3.0 = +45.00
    PBRS reward: ~1.15
    TOTAL PROGRESS: +46.15

  Safety Penalty (if lane invasion detected):
    Lane invasion: -10.00
    Weighted: -10.0 × 1.0 = -10.00

  NET REWARD: 46.15 + (-10.00) = 36.15

  ✅ STILL PROFITABLE! Off-road shortcut gives +36.15 net reward!
(base) danielterra@danielterra:/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system$ 
