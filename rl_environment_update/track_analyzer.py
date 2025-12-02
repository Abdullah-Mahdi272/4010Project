#!/usr/bin/env python3
"""
Track Analyzer - Understand Mario Kart gradient/position system

This script analyzes the gradient.txt and position.txt files to understand:
- How progress is measured (position values, not gradients!)
- Lap completion detection
- Track structure

CRITICAL INSIGHTS:
- Position values DECREASE as you progress (e.g., 419 → 1)
- Gradient values are WEIGHTED distances (include wall penalties, terrain)
- Lap completion: position wraps from LOW (~1) to HIGH (~419)
- GRADIENT_LAP_CHECK is the position threshold for lap detection
"""
import numpy as np
import os


def analyze_track(track_path):
    """
    Analyze a track's gradient and position files
    
    Args:
        track_path: Path to track directory (e.g., assets/circuit/donut_plains_1)
    """
    gradient_file = os.path.join(track_path, "gradient.txt")
    position_file = os.path.join(track_path, "position.txt")
    
    if not os.path.exists(gradient_file) or not os.path.exists(position_file):
        print(f"Error: Track files not found in {track_path}")
        return None
    
    print("="*70)
    print(f"ANALYZING TRACK: {os.path.basename(track_path)}")
    print("="*70)
    
    # Load gradient matrix
    gradient = []
    with open(gradient_file, 'r') as f:
        for line in f:
            values = [int(x) for x in line.strip().split()]
            gradient.append(values)
    gradient = np.array(gradient)
    
    # Load position matrix and lap check value
    position = []
    with open(position_file, 'r') as f:
        lines = f.readlines()
        for line in lines[:-1]:  # All except last line
            values = [int(x) for x in line.strip().split()]
            position.append(values)
        # Last line is GRADIENT_LAP_CHECK
        gradient_lap_check = int(lines[-1].strip())
    position = np.array(position)
    
    print(f"\nTrack Dimensions: {gradient.shape[0]} x {gradient.shape[1]}")
    
    # Analyze gradient values
    valid_gradient = gradient[gradient >= 0]
    print(f"\nGRADIENT VALUES (weighted distance from finish):")
    print(f"  Min: {valid_gradient.min()}")
    print(f"  Max: {valid_gradient.max()}")
    print(f"  Mean: {valid_gradient.mean():.1f}")
    print(f"  Note: -1 = wall/boundary, -2 = uninitialized")
    
    # Analyze position values
    valid_position = position[position >= 0]
    print(f"\nPOSITION VALUES (actual progress metric!):")
    print(f"  Min: {valid_position.min()}")
    print(f"  Max: {valid_position.max()}")
    print(f"  Mean: {valid_position.mean():.1f}")
    print(f"  Range: {valid_position.max() - valid_position.min()}")
    
    # Lap detection
    max_position = valid_position.max()
    print(f"\nLAP COMPLETION DETECTION:")
    print(f"  GRADIENT_LAP_CHECK: {gradient_lap_check}")
    print(f"  MAX_POSITION_MATRIX: {gradient_lap_check + 10}")
    print(f"  Actual max position: {max_position}")
    print(f"\n  How it works:")
    print(f"    - Position DECREASES as you progress ({max_position} → 1)")
    print(f"    - Lap complete when: position < {gradient_lap_check}")
    print(f"    - Then position WRAPS to ~{max_position}")
    
    # Find finish line
    finish_positions = np.where(position == 0)
    if len(finish_positions[0]) > 0:
        print(f"\n  Finish line location (position = 0):")
        print(f"    Rows: {finish_positions[0][:5]}")
        print(f"    Cols: {finish_positions[1][:5]}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS FOR RL TRAINING:")
    print("="*70)
    print(f"1. Track progress using POSITION values, not gradients!")
    print(f"2. Normal progress: position DECREASES ({max_position} → {gradient_lap_check})")
    print(f"3. Lap completion: position < {gradient_lap_check}, then wraps to ~{max_position}")
    print(f"4. Going backwards: position INCREASES")
    print(f"5. State vector index 1 is 'gradient' but actually uses position internally")
    print("="*70)
    
    return {
        'gradient_shape': gradient.shape,
        'position_shape': position.shape,
        'min_position': int(valid_position.min()),
        'max_position': int(valid_position.max()),
        'gradient_lap_check': gradient_lap_check,
        'max_position_matrix': gradient_lap_check + 10,
    }


def compare_tracks():
    """Compare multiple tracks if available"""
    tracks = [
        "assets/circuit/donut_plains_1",
        "assets/circuit/mario_circuit_2",
        "assets/circuit/rainbow_road",
    ]
    
    results = []
    for track in tracks:
        if os.path.exists(track):
            result = analyze_track(track)
            if result:
                results.append((track, result))
            print()
    
    if len(results) > 1:
        print("\n" + "="*70)
        print("TRACK COMPARISON:")
        print("="*70)
        for track, data in results:
            print(f"\n{os.path.basename(track)}:")
            print(f"  Max position: {data['max_position']}")
            print(f"  Lap check threshold: {data['gradient_lap_check']}")


if __name__ == '__main__':
    import sys
    
    # Analyze Donut Plains 1 (the uploaded track)
    print("\nAnalyzing Donut Plains 1 from uploaded files...")
    
    # Create temp directory with uploaded files
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    track_dir = os.path.join(temp_dir, "donut_plains_1")
    os.makedirs(track_dir, exist_ok=True)
    
    # Copy uploaded files
    shutil.copy("/mnt/user-data/uploads/gradient.txt", track_dir)
    shutil.copy("/mnt/user-data/uploads/position.txt", track_dir)
    
    # Analyze
    analyze_track(track_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print("\n" + "="*70)
    print("USAGE FOR YOUR TRACKS:")
    print("="*70)
    print("python track_analyzer.py <path_to_track_directory>")
    print("\nExample:")
    print("  python track_analyzer.py ../cloned_4010Project/assets/circuit/donut_plains_1")
