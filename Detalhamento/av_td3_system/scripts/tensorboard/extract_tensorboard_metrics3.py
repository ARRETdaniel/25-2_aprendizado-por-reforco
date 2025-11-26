#!/usr/bin/env python3
"""
Extract TensorBoard metrics from event file for systematic analysis.

Reads TensorBoard event file and exports all scalar metrics to CSV format
for detailed analysis of TD3 training behavior.

Author: Daniel Terra
Date: 2025-11-26
"""
import sys
import os
sys.path.insert(0, '/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system')

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
from pathlib import Path


def extract_metrics(event_file_path: str, output_dir: str = None):
    """
    Extract all scalar metrics from TensorBoard event file.

    Args:
        event_file_path: Path to TensorBoard event file
        output_dir: Directory to save extracted CSVs (default: same as event file)
    """
    # Validate event file exists
    if not os.path.exists(event_file_path):
        print(f'ERROR: Event file not found: {event_file_path}')
        sys.exit(1)

    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(event_file_path), 'extracted_metrics')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading event file: {event_file_path}')
    print(f'Output directory: {output_dir}')

    # Load TensorBoard event file
    ea = event_accumulator.EventAccumulator(
        event_file_path,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        }
    )
    ea.Reload()

    # Get all available scalar tags
    tags = ea.Tags()
    scalar_tags = tags.get('scalars', [])

    if not scalar_tags:
        print('WARNING: No scalar metrics found in event file!')
        return

    print(f'\n{"="*80}')
    print(f'Found {len(scalar_tags)} scalar metrics:')
    for tag in scalar_tags:
        print(f'  - {tag}')
    print(f'{"="*80}\n')

    # Extract all metrics
    metrics_data = {}
    all_metrics_combined = []

    for tag in scalar_tags:
        events = ea.Scalars(tag)

        if not events:
            print(f'WARNING: No data for tag: {tag}')
            continue

        # Convert to DataFrame
        df = pd.DataFrame([{
            'step': e.step,
            'wall_time': e.wall_time,
            'value': e.value
        } for e in events])

        metrics_data[tag] = df

        # Add to combined dataset
        df_combined = df.copy()
        df_combined['metric'] = tag
        all_metrics_combined.append(df_combined)

        # Save individual metric CSV
        filename = tag.replace('/', '_').replace('\\', '_') + '.csv'
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)

        print(f'Extracted {tag}:')
        print(f'  - Data points: {len(df)}')
        print(f'  - Step range: {df["step"].min()} to {df["step"].max()}')
        print(f'  - Saved to: {filename}')

    # Save combined metrics file
    if all_metrics_combined:
        combined_df = pd.concat(all_metrics_combined, ignore_index=True)
        combined_path = os.path.join(output_dir, 'all_metrics_combined.csv')
        combined_df.to_csv(combined_path, index=False)
        print(f'\nCombined metrics saved to: all_metrics_combined.csv')

    # Generate summary statistics
    print(f'\n{"="*80}')
    print('SUMMARY STATISTICS')
    print(f'{"="*80}')

    summary_rows = []
    for tag, df in metrics_data.items():
        if len(df) == 0:
            continue

        summary = {
            'metric': tag,
            'count': len(df),
            'step_min': df['step'].min(),
            'step_max': df['step'].max(),
            'value_mean': df['value'].mean(),
            'value_std': df['value'].std(),
            'value_min': df['value'].min(),
            'value_max': df['value'].max(),
        }
        summary_rows.append(summary)

        print(f'\n{tag}:')
        print(f'  Count: {summary["count"]} data points')
        print(f'  Steps: {summary["step_min"]} to {summary["step_max"]}')
        print(f'  Value: {summary["value_mean"]:.4f} ± {summary["value_std"]:.4f}')
        print(f'  Range: [{summary["value_min"]:.4f}, {summary["value_max"]:.4f}]')

    # Save summary statistics
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, 'summary_statistics.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f'\nSummary statistics saved to: summary_statistics.csv')

    print(f'\n{"="*80}')
    print(f'Extraction complete!')
    print(f'All files saved to: {output_dir}')
    print(f'{"="*80}')

    return metrics_data


if __name__ == '__main__':
    # Path to the latest training run
    event_file = '/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/data/logs/TD3_scenario_0_npcs_20_20251126-171053/events.out.tfevents.1764177053.danielterra.1.0'

    # Extract metrics
    metrics = extract_metrics(event_file)
