(base) danielterra@danielterra:/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system$ cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && python3 -c "
> import tensorflow as tf
> from tensorflow.python.summary.summary_iterator import summary_iterator
>
> log_file = 'data/logs/TD3_scenario_0_npcs_20_20251121-174251/events.out.tfevents.1763746971.danielterra.1.0'
>
> # Extract Q-values around the transition (steps 900-1200)
> q1_values = []
> q2_values = []
> target_q_values = []
> rewards = []
>
> for event in summary_iterator(log_file):
>     step = event.step
>     if step >= 900 and step <= 1200:
>         for value in event.summary.value:
>             if value.tag == 'train/q1_value':
>                 q1_values.append((step, value.simple_value))
>             elif value.tag == 'train/q2_value':
>                 q2_values.append((step, value.simple_value))
>             elif value.tag == 'debug/target_q_mean':
>                 target_q_values.append((step, value.simple_value))
>             elif value.tag == 'debug/reward_mean':
>                 rewards.append((step, value.simple_value))
>
> print('=' * 70)
> print('Q-VALUE ANALYSIS AROUND EXPLORATION → LEARNING TRANSITION')
> print('=' * 70)
> print('\\nPhase Transition: Step 1000 → 1001')
> print('  Before 1000: EXPLORATION (random actions)')
> print('  After 1001: LEARNING (actor policy + noise)')
> print('')
>
> if q1_values:
>     print('Q1-VALUES (Critic 1 Estimates):')
>     for step, val in q1_values[-20:]:
>         phase = 'EXPLORE' if step <= 1000 else 'LEARN  '
>         print(f'  Step {step:4d} [{phase}]: Q1 = {val:+8.2f}')
>
>     print('\\nQ2-VALUES (Critic 2 Estimates):')
>     for step, val in q2_values[-20:]:
>         phase = 'EXPLORE' if step <= 1000 else 'LEARN  '
>         print(f'  Step {step:4d} [{phase}]: Q2 = {val:+8.2f}')
>
>     print('\\nTARGET Q-VALUES (Used for Bellman updates):')
>     for step, val in target_q_values[-20:]:
>         phase = 'EXPLORE' if step <= 1000 else 'LEARN  '
>         print(f'  Step {step:4d} [{phase}]: Target Q = {val:+8.2f}')
>
>     print('\\nREWARDS (Actual environment feedback):')
>     for step, val in rewards[-20:]:
>         phase = 'EXPLORE' if step <= 1000 else 'LEARN  '
>         print(f'  Step {step:4d} [{phase}]: Reward = {val:+8.2f}')
> else:
>     print('❌ NO Q-VALUE DATA FOUND')
>     print('This suggests Q-networks may not be training correctly!')
> "
2025-11-21 15:36:38.254483: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:From /home/danielterra/miniconda3/lib/python3.13/site-packages/tensorflow/python/summary/summary_iterator.py:27: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
Instructions for updating:
Use eager execution and:
`tf.data.TFRecordDataset(path)`
======================================================================
Q-VALUE ANALYSIS AROUND EXPLORATION → LEARNING TRANSITION
======================================================================

Phase Transition: Step 1000 → 1001
  Before 1000: EXPLORATION (random actions)
  After 1001: LEARNING (actor policy + noise)

Q1-VALUES (Critic 1 Estimates):
  Step 1100 [LEARN  ]: Q1 =   +15.79
  Step 1200 [LEARN  ]: Q1 =   +14.33

Q2-VALUES (Critic 2 Estimates):
  Step 1100 [LEARN  ]: Q2 =   +15.75
  Step 1200 [LEARN  ]: Q2 =   +13.97

TARGET Q-VALUES (Used for Bellman updates):
  Step 1100 [LEARN  ]: Target Q =   +15.44
  Step 1200 [LEARN  ]: Target Q =   +15.12

REWARDS (Actual environment feedback):
  Step 1100 [LEARN  ]: Reward =   +14.31
  Step 1200 [LEARN  ]: Reward =   +12.52
(base) danielterra@danielterra:/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system$ 
