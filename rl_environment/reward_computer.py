#!/usr/bin/env python3
"""
Mario Kart RL Reward Computer

Designed to guide the agent to complete laps on the correct path while avoiding stuck situations.

Core Philosophy:
1. LAP COMPLETION is the #1 priority (5000 points)
2. FORWARD PROGRESS is the primary goal (20 points per position)
3. Getting stuck is heavily penalized with escalating severity (-50 to -1750/step)
4. Going backwards is penalized (-100 to -300)
5. Speed is secondary and only matters when making progress

The game uses POSITION values to track progress:
- Position DECREASES as you progress (e.g., 419 → 1)
- Lap completion: position < LAP_CHECK, then wraps to MAX
- Going backwards: position INCREASES

State format: [time, position, lap, split, posX, posY, speed, turnSpeed, angle, rank, coins]
"""
import numpy as np


class ImprovedRacingRewardComputer:
    """
    Reward computer that prioritizes lap completion and forward progress.

    Priority hierarchy:
    1. Complete laps (exploration encouraged)
    2. Forward progress down the track
    3. Avoid extended stuck situations (brief stuck OK for exploration)

    Reward Structure:
    - Lap completion: +5000 (highest priority, enables exploration)
    - Forward progress: +20 per position moved
    - Forward momentum: +50 for good progress (>3 positions/step)
    - Consistency: +20 when moving forward with speed
    - Speed: +1 per unit (secondary to progress)

    Penalty Structure (escalating severity):
    - Going backwards: -100 to -300 (triggers at 2+ positions backward)
    - Getting stuck (0.5s): -50/step (allows exploration)
    - Getting stuck (30s): -200 additional/step (-250 total)
    - Getting stuck (1 min): -500 additional/step (-750 total)
    - Getting stuck (2+ min): -1000 additional/step (-1750 total), then episode terminates
    - Off-track/water: -150 (disabled when already stuck to avoid double penalty)
    - Wall collision: -60
    """

    def __init__(self, max_position=419, lap_check_threshold=409):
        """
        Args:
            max_position: Maximum position value on track (from position.txt)
            lap_check_threshold: Threshold for lap completion detection (GRADIENT_LAP_CHECK)
        """
        self.max_position = max_position
        self.lap_check_threshold = lap_check_threshold

        # Previous state for computing deltas
        self.prev_position = None
        self.prev_lap = 1
        self.prev_rank = 8
        self.prev_coins = 0
        self.prev_speed = 0.0
        self.prev_pos_x = 0.0
        self.prev_pos_y = 0.0

        # Episode statistics
        self.total_laps_completed = 0
        self.best_position = max_position
        self.stuck_counter = 0
        self.backward_counter = 0
        self.offtrack_counter = 0
        self.low_speed_counter = 0

        # Step counter and grace period
        self.step_count = 0
        self.grace_period_steps = 10  # Short grace period at episode start

        # Reward weights
        self.PROGRESS_WEIGHT = 20.0               # Forward progress primary reward
        self.SPEED_WEIGHT = 1.0                   # Speed reward (secondary to progress)
        self.LAP_COMPLETE_BONUS = 5000.0          # Lap completion bonus (highest priority)
        self.LAP_PROGRESS_BONUS = 200.0           # Bonus for getting close to lap completion
        self.RANK_IMPROVEMENT = 100.0             # Passing other racers
        self.RANK_LOSS = -50.0                    # Getting passed
        self.TIME_PENALTY = -0.05                 # Time penalty to encourage speed
        self.FORWARD_MOMENTUM_BONUS = 50.0        # Bonus for sustained forward progress

        # Escalating penalties (brief stuck allows exploration, extended stuck heavily penalized)
        self.BACKWARD_PENALTY = -100.0            # Any backward movement
        self.STUCK_PENALTY = -50.0                # Brief stuck (0.5s)
        self.OFFTRACK_PENALTY = -150.0            # Hitting water/off track
        self.PERSISTENT_BACKWARD_PENALTY = -300.0  # Continuing to go backwards
        self.PERSISTENT_STUCK_PENALTY = -200.0     # Stuck for 30s (additional penalty)
        self.EXTREMELY_STUCK_PENALTY = -500.0      # Stuck for 1 minute (additional penalty)
        self.MEGA_STUCK_PENALTY = -1000.0          # Stuck for 2+ minutes (additional penalty)
        self.WALL_COLLISION_PENALTY = -60.0        # Wall hits

        self.COIN_BONUS = 5.0                      # Collecting coins
        self.CONSISTENCY_BONUS = 20.0              # Steady progress bonus

        # Episode termination threshold
        self.TERMINATE_IF_STUCK = 7200             # Terminate episode if stuck > 2 minutes (7200 frames @ 60fps)

    def reset(self):
        """Reset for new episode"""
        self.prev_position = None
        self.prev_lap = 1
        self.prev_rank = 8
        self.prev_coins = 0
        self.prev_speed = 0.0
        self.prev_pos_x = 0.0
        self.prev_pos_y = 0.0
        self.total_laps_completed = 0
        self.best_position = self.max_position
        self.stuck_counter = 0
        self.backward_counter = 0
        self.offtrack_counter = 0
        self.low_speed_counter = 0
        self.step_count = 0

    def compute_reward(self, state, info):
        """
        Compute reward from current state

        Args:
            state: np.array of shape (11,) with [time, position, lap, split, pos_x, pos_y,
                                                  speed, turn_speed, angle, rank, coins]
            info: dict with same info

        Returns:
            reward: float - total reward for this timestep
            components: dict - reward breakdown for debugging
        """
        reward = 0.0
        components = {}

        # Increment step counter
        self.step_count += 1
        in_grace_period = self.step_count <= self.grace_period_steps

        # Extract state
        speed = float(state[6])
        position = int(state[1])  # This is actually position value, not gradient!
        lap = int(state[2])
        rank = int(state[9])
        coins = int(state[10])
        pos_x = float(state[4])
        pos_y = float(state[5])

        # Initialize on first call
        if self.prev_position is None:
            self.prev_position = position
            self.best_position = position
            self.prev_pos_x = pos_x
            self.prev_pos_y = pos_y
            return 0.0, {}

        # Calculate movement delta
        spatial_movement = np.sqrt((pos_x - self.prev_pos_x)**2 + (pos_y - self.prev_pos_y)**2)

        # === 1. POSITION-BASED PROGRESS ===
        position_delta = self.prev_position - position  # Positive = moving forward

        # Detect lap completion (crossing finish line going forward)
        # Must be near finish line AND position wraps to near start
        if (self.prev_position < self.lap_check_threshold and
            position > self.max_position * 0.8 and
            position_delta < -self.max_position * 0.5):
            # Position jumped from low (near finish) to high (near start)
            lap_bonus = self.LAP_COMPLETE_BONUS
            reward += lap_bonus
            components['lap_complete'] = lap_bonus
            self.total_laps_completed += 1

            print(f"    🏁 LAP {self.total_laps_completed} COMPLETE! "
                  f"(position: {self.prev_position}→{position}) +{lap_bonus:.1f}")

            # Also reward the full lap progress
            full_progress = self.prev_position + (self.max_position - position)
            progress_reward = self.PROGRESS_WEIGHT * full_progress
            reward += progress_reward
            components['lap_progress'] = progress_reward

            # Reset counters
            self.stuck_counter = 0
            self.backward_counter = 0
            self.offtrack_counter = 0
            self.low_speed_counter = 0
            self.best_position = position

        elif position_delta > 0:
            # Normal forward progress (position decreasing)
            progress_reward = self.PROGRESS_WEIGHT * position_delta
            reward += progress_reward
            components['progress'] = progress_reward

            # Update best position
            if position < self.best_position:
                self.best_position = position
                # Extra bonus for new best
                reward += 2.0
                components['new_best'] = 2.0

            # Near lap completion bonus
            if position < self.lap_check_threshold:
                proximity_bonus = self.LAP_PROGRESS_BONUS * \
                    (1.0 - position / self.lap_check_threshold)
                reward += proximity_bonus
                components['near_lap_end'] = proximity_bonus

            self.stuck_counter = 0
            self.backward_counter = 0
            self.offtrack_counter = 0

        elif position_delta < -2 and position_delta > -self.max_position * 0.5:
            # Going backwards (wrong direction)
            if not in_grace_period:
                backward_magnitude = abs(position_delta)
                backward_reward = self.BACKWARD_PENALTY * (backward_magnitude / 2.0)
                reward += backward_reward
                components['backward'] = backward_reward
                self.backward_counter += 1

                # Additional penalty for persistent backward movement
                if self.backward_counter > 15:  # ~0.25 seconds
                    persistent_penalty = self.PERSISTENT_BACKWARD_PENALTY
                    reward += persistent_penalty
                    components['persistent_backward'] = persistent_penalty
                    print(f"    ⚠️  WRONG DIRECTION! {persistent_penalty:.1f}")
                    self.backward_counter = 0

        else:
            # Stuck (no position change)
            self.stuck_counter += 1

            # Apply stuck penalty after 0.5 seconds
            if not in_grace_period and self.stuck_counter > 30:  # 0.5 seconds stuck
                stuck_penalty = self.STUCK_PENALTY
                reward += stuck_penalty
                components['stuck'] = stuck_penalty

                # Escalating penalties for extended stuck duration
                if self.stuck_counter > 1800:  # 30 seconds
                    persistent_stuck = self.PERSISTENT_STUCK_PENALTY
                    reward += persistent_stuck
                    components['persistent_stuck'] = persistent_stuck
                    if self.stuck_counter % 600 == 0:  # Print every 10 seconds
                        print(f"    ⚠️  STUCK FOR 30s! {persistent_stuck:.1f}")

                if self.stuck_counter > 3600:  # 60 seconds
                    extreme_stuck = self.EXTREMELY_STUCK_PENALTY
                    reward += extreme_stuck
                    components['extremely_stuck'] = extreme_stuck
                    if self.stuck_counter % 600 == 0:
                        print(f"    🚨 STUCK FOR 1 MINUTE! {extreme_stuck:.1f}")

                if self.stuck_counter > 7200:  # 120 seconds
                    mega_stuck = self.MEGA_STUCK_PENALTY
                    reward += mega_stuck
                    components['mega_stuck'] = mega_stuck
                    if self.stuck_counter % 1200 == 0:  # Print every 20 seconds
                        print(f"    💀 STUCK FOR 2+ MINUTES! {mega_stuck:.1f}")

        self.prev_position = position

        # === 2. OFF-TRACK / WATER DETECTION ===
        # Detect by very low speed + no spatial movement
        # Only trigger if NOT already being penalized for being stuck (avoid double penalty)
        if speed < 0.05 and spatial_movement < 0.001 and self.stuck_counter < 30:
            self.low_speed_counter += 1
            # Trigger after 1 second of low speed (60 frames @ 60fps)
            if not in_grace_period and self.low_speed_counter > 60:
                self.offtrack_counter += 1

                # Apply off-track penalty
                if self.offtrack_counter > 5:
                    offtrack_penalty = self.OFFTRACK_PENALTY
                    reward += offtrack_penalty
                    components['offtrack'] = offtrack_penalty
                    print(f"    ⚠️  OFF TRACK / WATER! {offtrack_penalty:.1f}")
                    self.offtrack_counter = 0
        else:
            self.low_speed_counter = 0
            self.offtrack_counter = 0

        # === 3. SPEED REWARD ===
        if speed > 0:
            speed_reward = self.SPEED_WEIGHT * speed
            reward += speed_reward
            components['speed'] = speed_reward

        # === 3.5. WALL COLLISION DETECTION ===
        # Detect sudden speed drops (wall hits) - walls stop progress!
        if not in_grace_period and self.prev_speed > 0.4 and speed < 0.1:
            wall_penalty = self.WALL_COLLISION_PENALTY
            reward += wall_penalty
            components['wall_collision'] = wall_penalty
            print(f"    ⚠️  WALL COLLISION! (speed {self.prev_speed:.2f}→{speed:.2f}) {wall_penalty:.1f}")

        # === 4. LAP CHANGES (backup detection) ===
        if lap > self.prev_lap:
            lap_change = self.LAP_COMPLETE_BONUS * (lap - self.prev_lap)
            reward += lap_change
            components['lap_change_backup'] = lap_change
            print(f"    🏁 Lap {lap} (game state backup) +{lap_change:.1f}")
        self.prev_lap = lap

        # === 5. RANK CHANGES ===
        if rank > 0:  # Valid rank
            rank_delta = self.prev_rank - rank  # Positive = improved rank
            if rank_delta > 0:
                rank_reward = self.RANK_IMPROVEMENT * rank_delta
                reward += rank_reward
                components['rank_up'] = rank_reward
                print(f"    🏆 Rank → {rank}! +{rank_reward:.1f}")
            elif rank_delta < 0:
                rank_reward = self.RANK_LOSS * abs(rank_delta)
                reward += rank_reward
                components['rank_down'] = rank_reward
            self.prev_rank = rank

        # === 6. TIME PENALTY (encourages fast completion) ===
        reward += self.TIME_PENALTY
        components['time'] = self.TIME_PENALTY

        # === 7. COINS ===
        if coins > self.prev_coins:
            coin_reward = self.COIN_BONUS * (coins - self.prev_coins)
            reward += coin_reward
            components['coins'] = coin_reward
        self.prev_coins = coins

        # === 8. CONSISTENCY BONUS ===
        # Reward for making steady progress while maintaining speed
        if speed > 0.1 and position_delta > 0:
            consistency = self.CONSISTENCY_BONUS
            reward += consistency
            components['consistency'] = consistency

        # === 8.5. FORWARD MOMENTUM BONUS ===
        # Bonus for sustained good forward progress
        if position_delta > 3:  # Good forward movement
            momentum_bonus = self.FORWARD_MOMENTUM_BONUS
            reward += momentum_bonus
            components['momentum'] = momentum_bonus

        # === 9. SPEED ACCELERATION BONUS ===
        # Reward for increasing speed
        if speed > self.prev_speed:
            accel_bonus = (speed - self.prev_speed) * 2.0
            reward += accel_bonus
            components['acceleration'] = accel_bonus
        self.prev_speed = speed

        # Update previous position
        self.prev_pos_x = pos_x
        self.prev_pos_y = pos_y

        # === 10. TERMINATION CHECK ===
        # If stuck for too long (2 minutes), terminate episode
        should_terminate = self.stuck_counter > self.TERMINATE_IF_STUCK
        if should_terminate:
            components['terminated_stuck'] = True
            print(f"    💀 TERMINATING: Stuck for {self.stuck_counter/60:.1f}s")

        return reward, components

    def get_episode_stats(self):
        """Get statistics about current episode"""
        return {
            'laps_completed': self.total_laps_completed,
            'best_position': self.best_position,
            'progress_pct': (self.max_position - self.best_position) / self.max_position * 100
        }


if __name__ == '__main__':
    print("Testing reward computer...")
    print("Primary goal: Lap completion (+5000) and forward progress (+20/position)")
    print("Getting stuck is heavily penalized with escalating severity")
    print()

    # Create reward computer
    rc = ImprovedRacingRewardComputer(max_position=419, lap_check_threshold=409)

    # Test scenario 1: Normal forward progress
    print("\nTest 1: Normal forward progress")
    state = np.array([0.0, 419, 1, 0, 0.5, 0.5, 0.15, 0.0, 0.0, 8, 0], dtype=np.float32)
    rc.step_count = 100  # Skip grace period

    for i in range(5):
        state[1] -= 4  # Decrease position by 4 (good progress)
        state[4] += 0.01  # Move spatially
        state[5] += 0.01
        state[6] = 0.15  # Good speed
        reward, components = rc.compute_reward(state, {})
        print(f"  Step {i+1}: position={int(state[1])}, reward={reward:.1f}")
        if i == 4:
            print(f"    Components: {components}")

    # Test scenario 2: Going backwards
    print("\nTest 2: Going backwards")
    rc.reset()
    rc.step_count = 100  # Skip grace period
    state = np.array([0.0, 300, 1, 0, 0.5, 0.5, 0.5, 0.0, 0.0, 8, 0], dtype=np.float32)
    rc.compute_reward(state, {})  # Initialize

    for i in range(20):
        state[1] += 3  # Increase position (going backwards)
        reward, components = rc.compute_reward(state, {})
        if i == 0 or i == 19:
            print(f"  Step {i+1}: position={int(state[1])}, reward={reward:.1f}")
            if 'persistent_backward' in components:
                print(f"    Persistent backward: {components['persistent_backward']:.1f}")

    # Test scenario 3: Getting stuck
    print("\nTest 3: Getting stuck (escalating penalties)")
    rc.reset()
    rc.step_count = 100  # Skip grace period
    state = np.array([0.0, 300, 1, 0, 0.5, 0.5, 0.5, 0.0, 0.0, 8, 0], dtype=np.float32)
    rc.compute_reward(state, {})  # Initialize

    test_times = [35, 1805, 3605, 7205]  # 0.5s, 30s, 1min, 2min
    for target in test_times:
        while rc.stuck_counter < target:
            state[6] = 0.1  # Low speed, no movement
            reward, components = rc.compute_reward(state, {})

        time_label = {35: "0.5s", 1805: "30s", 3605: "1min", 7205: "2min"}[target]
        print(f"  Stuck for {time_label}: reward={reward:.1f}, components={components}")

    # Test scenario 4: Off track / water
    print("\nTest 4: Off track / water")
    rc.reset()
    rc.step_count = 100  # Skip grace period
    state = np.array([0.0, 300, 1, 0, 0.5, 0.5, 0.5, 0.0, 0.0, 8, 0], dtype=np.float32)
    rc.compute_reward(state, {})  # Initialize

    for i in range(70):
        state[6] = 0.01  # Very low speed, no movement
        reward, components = rc.compute_reward(state, {})
        if i == 69 and 'offtrack' in components:
            print(f"  Step {i+1}: OFF TRACK! reward={reward:.1f}")
            print(f"    Off-track penalty: {components['offtrack']:.1f}")

    print("\n" + "="*60)
    print("✓ Reward Computer Test Complete")
    print("="*60)
    print("REWARDS:")
    print("  ✅ Lap completion: +5000")
    print("  ✅ Forward progress: +20 per position")
    print("  ✅ Momentum bonus: +50 for sustained progress")
    print()
    print("PENALTIES (escalating):")
    print("  ❌ Going backwards: -100 to -300")
    print("  ❌ Stuck (0.5s): -50/step")
    print("  ❌ Stuck (30s): -200/step additional (-250 total)")
    print("  ❌ Stuck (1min): -500/step additional (-750 total)")
    print("  ❌ Stuck (2min): -1000/step additional (-1750 total)")
    print("  ❌ Stuck (2min): Episode terminates")
    print("  ❌ Off-track/water: -150 (only if not stuck)")
    print("  ❌ Wall collision: -60")
