#!/usr/bin/env python3
"""
TensorBoard Event File Inspector
=================================

Purpose: Directly inspect TensorBoard event file to see which metrics are
         actually written to disk and at what steps.

Reference: https://www.tensorflow.org/tensorboard/get_started
           https://www.tensorflow.org/tensorboard/scalars_and_keras

Date: November 13, 2025
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
    print("✅ TensorFlow imported successfully")
except ImportError:
    print("❌ TensorFlow not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    from tensorflow.python.summary.summary_iterator import summary_iterator
    print("✅ TensorFlow installed and imported")


def inspect_event_file(event_file_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Read TensorBoard event file and extract all scalar data.

    Args:
        event_file_path: Path to events.out.tfevents.* file

    Returns:
        Dictionary mapping scalar tags to list of (step, value) tuples
    """
    print(f"\n{'='*80}")
    print(f"INSPECTING EVENT FILE")
    print(f"{'='*80}")
    print(f"File: {event_file_path}\n")

    if not Path(event_file_path).exists():
        print(f"❌ ERROR: File not found: {event_file_path}")
        return {}

    print(f"File size: {Path(event_file_path).stat().st_size / 1024:.2f} KB\n")

    # Collect all scalar data
    scalars = defaultdict(list)
    total_events = 0

    try:
        for event in summary_iterator(event_file_path):
            total_events += 1

            # Process scalar values
            for value in event.summary.value:
                if hasattr(value, 'simple_value'):
                    # This is a scalar metric
                    scalars[value.tag].append((event.step, value.simple_value))

        print(f"✅ Successfully read {total_events} events\n")

    except Exception as e:
        print(f"❌ ERROR reading event file: {e}")
        return {}

    return dict(scalars)


def analyze_scalars(scalars: Dict[str, List[Tuple[int, float]]]):
    """
    Analyze and display scalar data from TensorBoard event file.

    Args:
        scalars: Dictionary mapping scalar tags to list of (step, value) tuples
    """
    if not scalars:
        print("❌ No scalar data found in event file!")
        return

    print(f"{'='*80}")
    print(f"SCALAR METRICS ANALYSIS")
    print(f"{'='*80}\n")

    # Sort by category
    categories = defaultdict(list)
    for tag in sorted(scalars.keys()):
        category = tag.split('/')[0] if '/' in tag else 'other'
        categories[category].append(tag)

    print(f"Total unique scalar tags: {len(scalars)}\n")

    # Display by category
    for category, tags in sorted(categories.items()):
        print(f"\n{'─'*80}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'─'*80}\n")

        for tag in tags:
            data_points = scalars[tag]
            num_points = len(data_points)

            if num_points == 0:
                print(f"  ⚠️  {tag}: NO DATA POINTS")
                continue

            # Get first, last, min, max values
            steps = [step for step, _ in data_points]
            values = [value for _, value in data_points]

            first_step, first_value = data_points[0]
            last_step, last_value = data_points[-1]
            min_value = min(values)
            max_value = max(values)

            # Check if values change (not frozen)
            is_constant = len(set(values)) == 1
            value_range = max_value - min_value

            # Determine status
            if num_points < 5:
                status = "⚠️  FEW POINTS"
            elif is_constant:
                status = "⚠️  CONSTANT"
            elif value_range < 1e-6:
                status = "⚠️  FROZEN"
            else:
                status = "✅ UPDATING"

            print(f"  {status} {tag}")
            print(f"      Data points: {num_points}")
            print(f"      Steps: {first_step} → {last_step}")
            print(f"      Values: {first_value:.6f} → {last_value:.6f}")

            if is_constant:
                print(f"      ⚠️  All values identical: {first_value:.6f}")
            else:
                print(f"      Range: [{min_value:.6f}, {max_value:.6f}]")

            print()


def main():
    """Main function to inspect TensorBoard event file."""

    # Event file path from user request - UPDATED to new run (11:00:06)
    event_file = Path("/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/data/logs/TD3_scenario_0_npcs_20_20251113-110006/events.out.tfevents.1763031606.danielterra.1.0")

    if not event_file.exists():
        print(f"❌ ERROR: Event file not found!")
        print(f"Expected path: {event_file}")

        # Try to find any event files in logs directory
        logs_dir = event_file.parent.parent
        print(f"\nSearching for event files in: {logs_dir}")

        event_files = list(logs_dir.rglob("events.out.tfevents.*"))
        if event_files:
            print(f"\nFound {len(event_files)} event file(s):")
            for f in event_files:
                print(f"  - {f}")

            # Use the first found file
            event_file = event_files[0]
            print(f"\n✅ Using: {event_file}")
        else:
            print("❌ No event files found!")
            return

    # Inspect the event file
    scalars = inspect_event_file(str(event_file))

    # Analyze the data
    analyze_scalars(scalars)

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")

    if scalars:
        total_tags = len(scalars)
        empty_tags = sum(1 for data in scalars.values() if len(data) == 0)
        constant_tags = sum(1 for data in scalars.values() if len(set(v for _, v in data)) == 1)
        updating_tags = total_tags - empty_tags - constant_tags

        print(f"Total scalar tags: {total_tags}")
        print(f"  ✅ Updating (values change): {updating_tags}")
        print(f"  ⚠️  Constant (same value): {constant_tags}")
        print(f"  ❌ Empty (no data): {empty_tags}")

        if constant_tags > 0:
            print(f"\n⚠️  WARNING: {constant_tags} metrics appear frozen (constant values)")
            print(f"   This may indicate:")
            print(f"   1. Short training run (values don't change yet)")
            print(f"   2. No learning rate scheduling (LRs stay constant)")
            print(f"   3. Phase detection bug (stats not logged during learning)")
    else:
        print("❌ No scalar data found!")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
