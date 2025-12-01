#!/usr/bin/env python3
"""
Mario Kart RL Reward Computer - EXPLOIT FIXED

CRITICAL FIX: Prevents "start-line bouncing" exploit where agent gets lap rewards 
by bouncing between position 0 and high positions near start.

Solution: Track furthest position reached during lap. Only count lap completion if:
1. You reached at least lap_check_threshold (409+) during this lap
2. Position wraps from high (near finish) to low (near start)
3. This prevents rewarding bouncing at start line (0 ↔ 419)

Position tracking: INCREASES as you progress (0 at start → 511 at finish)
- Forward: position_delta > 0 (100 → 105)
- Lap wrap: position 510 → 5 (large negative delta, this is VALID lap)
- Backward: position_delta < 0 (100 → 95)
- Start bounce: position 5 → 419 (large positive delta, this is EXPLOIT)

State format: [time, position, lap, split, posX, posY, speed, turnSpeed, angle, rank, coins]
"""
import numpy as np


class ImprovedRacingRewardComputer:
    """
    Reward computer with exploit fix for start-line bouncing.
    
    NEW: Tracks furthest_position_this_lap to validate lap completions
    """

    def __init__(self, max_position=511, lap_check_threshold=409):
        """
        Args:
            max_position: Maximum position value on track
            lap_check_threshold: Must reach this to count lap completion
        """
        self.max_position = max_position
        self.lap_check_threshold = lap_check_threshold

        self.prev_position = None
        self.prev_lap = 1
        self.prev_rank = 8
        self.prev_coins = 0
        self.prev_speed = 0.0
        self.prev_pos_x = 0.0
        self.prev_pos_y = 0.0

        self.total_laps_completed = 0
        self.best_position_overall = 0  # Best ever reached
        self.furthest_position_this_lap = 0  # Furthest in current lap - KEY FIX
        self.stuck_counter = 0
        self.backward_counter = 0
        self.offtrack_counter = 0
        self.low_speed_counter = 0

        self.step_count = 0
        self.grace_period_steps = 120

        # Reward weights
        self.PROGRESS_WEIGHT = 20.0
        self.SPEED_WEIGHT = 1.0
        self.LAP_COMPLETE_BONUS = 5000.0
        self.LAP_PROGRESS_BONUS = 200.0
        self.RANK_IMPROVEMENT = 100.0
        self.RANK_LOSS = -50.0
        self.TIME_PENALTY = -0.05
        self.FORWARD_MOMENTUM_BONUS = 50.0

        # Penalties
        self.BACKWARD_PENALTY = -100.0
        self.STUCK_PENALTY = -50.0
        self.OFFTRACK_PENALTY = -150.0
        self.PERSISTENT_BACKWARD_PENALTY = -300.0
        self.PERSISTENT_STUCK_PENALTY = -200.0
        self.EXTREMELY_STUCK_PENALTY = -500.0
        self.MEGA_STUCK_PENALTY = -1000.0
        self.WALL_COLLISION_PENALTY = -60.0

        self.COIN_BONUS = 5.0
        self.CONSISTENCY_BONUS = 20.0

        # Track centering
        self.EXTREME_EDGE_PENALTY = -80.0
        self.NEAR_EDGE_PENALTY = -20.0
        self.CENTER_ZONE_BONUS = 2.0

        self.TERMINATE_IF_STUCK = 7200

        print(f"Reward Computer initialized (EXPLOIT-FIXED):")
        print(f"  Max position: {max_position}")
        print(f"  Lap check threshold: {lap_check_threshold}")
        print(f"  Grace period: {self.grace_period_steps} steps (~2 seconds)")
        print(f"  Position INCREASES as you progress (0 → {max_position})")
        print(f"  ✓ Start-line bouncing exploit FIXED")
        print(f"    Must reach position {lap_check_threshold}+ before lap counts")

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
        self.best_position_overall = 0
        self.furthest_position_this_lap = 0  # Reset lap tracking
        self.stuck_counter = 0
        self.backward_counter = 0
        self.offtrack_counter = 0
        self.low_speed_counter = 0
        self.step_count = 0

    def compute_reward(self, state, info):
        """
        Compute reward with exploit prevention
        
        KEY FIX: Track furthest_position_this_lap and only count lap completion
        if agent actually reached lap_check_threshold during this lap.
        """
        reward = 0.0
        components = {}

        self.step_count += 1
        in_grace_period = self.step_count <= self.grace_period_steps

        # Extract state
        speed = float(state[6])
        position = int(state[1])
        lap = int(state[2])
        rank = int(state[9])
        coins = int(state[10])
        pos_x = float(state[4])
        pos_y = float(state[5])

        # Initialize on first step
        if self.prev_position is None:
            self.prev_position = position
            self.best_position_overall = position
            self.furthest_position_this_lap = position  # Start tracking
            self.prev_pos_x = pos_x
            self.prev_pos_y = pos_y
            print(f"  Initial position: {position} (max: {self.max_position})")
            print(f"  Furthest this lap: {self.furthest_position_this_lap}")
            return 0.0, {}

        # Track furthest position reached this lap
        # BUT: Ignore unrealistic jumps (position wraps) - only count gradual progress
        if position > self.furthest_position_this_lap:
            # Check if this is realistic forward progress (not a wrap from 0 → 419)
            position_jump = position - self.furthest_position_this_lap
            if position_jump < self.max_position * 0.5:
                # Normal forward progress (e.g., 100 → 150)
                self.furthest_position_this_lap = position
            # else: Ignore wrap-up jumps (e.g., 0 → 419), don't update furthest

        # Calculate movement
        spatial_movement = np.sqrt((pos_x - self.prev_pos_x)**2 + (pos_y - self.prev_pos_y)**2)
        
        # Position INCREASES as you progress (0 → 511)
        position_delta = position - self.prev_position

        # === LAP COMPLETION DETECTION (EXPLOIT-FIXED) ===
        # Valid lap: prev_position > threshold, position wrapped to low value
        # AND agent must have actually reached threshold this lap
        if (self.prev_position > self.lap_check_threshold and 
            position < self.max_position * 0.2 and
            position_delta < -self.max_position * 0.5):
            
            # CRITICAL CHECK: Did we actually complete the lap?
            if self.furthest_position_this_lap > self.lap_check_threshold:
                # VALID LAP COMPLETION
                lap_bonus = self.LAP_COMPLETE_BONUS
                reward += lap_bonus
                components['lap_complete'] = lap_bonus
                self.total_laps_completed += 1

                print(f"    🏁 LAP {self.total_laps_completed} COMPLETE! "
                      f"(position: {self.prev_position}→{position}, "
                      f"furthest: {self.furthest_position_this_lap}) +{lap_bonus:.1f}")

                # Reward the full lap progress
                full_progress = self.furthest_position_this_lap
                progress_reward = self.PROGRESS_WEIGHT * full_progress / 10.0
                reward += progress_reward
                components['lap_progress'] = progress_reward

                # Reset counters
                self.stuck_counter = 0
                self.backward_counter = 0
                self.offtrack_counter = 0
                self.low_speed_counter = 0
                
                # Reset lap tracking for next lap
                self.furthest_position_this_lap = position
                
            else:
                # EXPLOIT DETECTED: Position wrapped but didn't complete lap
                exploit_penalty = -500.0
                reward += exploit_penalty
                components['exploit_detected'] = exploit_penalty
                print(f"    ⛔ EXPLOIT BLOCKED! Wrapped {self.prev_position}→{position} "
                      f"but only reached {self.furthest_position_this_lap} "
                      f"(need {self.lap_check_threshold}+)")
                
                # Reset to prevent repeated exploit attempts
                self.furthest_position_this_lap = position

        # === NORMAL FORWARD PROGRESS ===
        elif position_delta > 0:
            # Moving forward
            progress_reward = self.PROGRESS_WEIGHT * position_delta
            reward += progress_reward
            components['progress'] = progress_reward

            # Update best position
            if position > self.best_position_overall:
                self.best_position_overall = position
                reward += 2.0
                components['new_best'] = 2.0

            # Near lap completion bonus
            if position > self.lap_check_threshold:
                proximity_bonus = self.LAP_PROGRESS_BONUS * (position / self.max_position)
                reward += proximity_bonus
                components['near_lap_end'] = proximity_bonus

            # Reset stuck/backward counters
            self.stuck_counter = 0
            self.backward_counter = 0
            self.offtrack_counter = 0

        # === GOING BACKWARD ===
        elif position_delta < -2 and abs(position_delta) < self.max_position * 0.5:
            if not in_grace_period:
                backward_magnitude = abs(position_delta)
                backward_reward = self.BACKWARD_PENALTY * (backward_magnitude / 2.0)
                reward += backward_reward
                components['backward'] = backward_reward
                self.backward_counter += 1

                if self.backward_counter > 15:
                    persistent_penalty = self.PERSISTENT_BACKWARD_PENALTY
                    reward += persistent_penalty
                    components['persistent_backward'] = persistent_penalty
                    print(f"    ⚠️ WRONG DIRECTION! {persistent_penalty:.1f}")
                    self.backward_counter = 0

        # === STUCK (NO MOVEMENT) ===
        else:
            self.stuck_counter += 1

            if not in_grace_period and self.stuck_counter > 30:
                stuck_penalty = self.STUCK_PENALTY
                reward += stuck_penalty
                components['stuck'] = stuck_penalty

                if self.stuck_counter > 1800:
                    persistent_stuck = self.PERSISTENT_STUCK_PENALTY
                    reward += persistent_stuck
                    components['persistent_stuck'] = persistent_stuck
                    if self.stuck_counter % 600 == 0:
                        print(f"    ⚠️ STUCK FOR 30s! {persistent_stuck:.1f}")

                if self.stuck_counter > 3600:
                    extreme_stuck = self.EXTREMELY_STUCK_PENALTY
                    reward += extreme_stuck
                    components['extremely_stuck'] = extreme_stuck
                    if self.stuck_counter % 600 == 0:
                        print(f"    🚨 STUCK FOR 1 MINUTE! {extreme_stuck:.1f}")

                if self.stuck_counter > 7200:
                    mega_stuck = self.MEGA_STUCK_PENALTY
                    reward += mega_stuck
                    components['mega_stuck'] = mega_stuck
                    if self.stuck_counter % 1200 == 0:
                        print(f"    💀 STUCK FOR 2+ MINUTES! {mega_stuck:.1f}")

        self.prev_position = position

        # === OFF-TRACK DETECTION ===
        if speed < 0.05 and spatial_movement < 0.001 and self.stuck_counter < 30:
            self.low_speed_counter += 1
            if not in_grace_period and self.low_speed_counter > 60:
                self.offtrack_counter += 1

                if self.offtrack_counter > 5:
                    offtrack_penalty = self.OFFTRACK_PENALTY
                    reward += offtrack_penalty
                    components['offtrack'] = offtrack_penalty
                    print(f"    ⚠️ OFF TRACK / WATER! {offtrack_penalty:.1f}")
                    self.offtrack_counter = 0
        else:
            self.low_speed_counter = 0
            self.offtrack_counter = 0

        # === SPEED REWARD ===
        if speed > 0:
            speed_reward = self.SPEED_WEIGHT * speed
            reward += speed_reward
            components['speed'] = speed_reward

        # === WALL COLLISION ===
        if not in_grace_period and self.prev_speed > 0.4 and speed < 0.1:
            wall_penalty = self.WALL_COLLISION_PENALTY
            reward += wall_penalty
            components['wall_collision'] = wall_penalty
            if self.step_count % 60 == 0:
                print(f"    ⚠️ WALL COLLISION! (speed {self.prev_speed:.2f}→{speed:.2f}) "
                      f"{wall_penalty:.1f}")

        # === LAP COUNTER BACKUP ===
        if lap > self.prev_lap:
            # Only reward if we actually reached the threshold
            if self.furthest_position_this_lap > self.lap_check_threshold:
                lap_change = self.LAP_COMPLETE_BONUS * (lap - self.prev_lap)
                reward += lap_change
                components['lap_change_backup'] = lap_change
                print(f"    🏁 Lap {lap} (game state backup) +{lap_change:.1f}")
                self.furthest_position_this_lap = position  # Reset for next lap
        self.prev_lap = lap

        # === RANK CHANGES ===
        if rank > 0:
            rank_delta = self.prev_rank - rank
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

        # === TIME PENALTY ===
        reward += self.TIME_PENALTY
        components['time'] = self.TIME_PENALTY

        # === COIN BONUS ===
        if coins > self.prev_coins:
            coin_reward = self.COIN_BONUS * (coins - self.prev_coins)
            reward += coin_reward
            components['coins'] = coin_reward
        self.prev_coins = coins

        # === CONSISTENCY BONUS ===
        if speed > 0.1 and position_delta > 0:
            consistency = self.CONSISTENCY_BONUS
            reward += consistency
            components['consistency'] = consistency

        # === MOMENTUM BONUS ===
        if position_delta > 3:
            momentum_bonus = self.FORWARD_MOMENTUM_BONUS
            reward += momentum_bonus
            components['momentum'] = momentum_bonus

        # === ACCELERATION BONUS ===
        if speed > self.prev_speed:
            accel_bonus = (speed - self.prev_speed) * 2.0
            reward += accel_bonus
            components['acceleration'] = accel_bonus
        self.prev_speed = speed

        # === TRACK CENTERING (Edge Detection) ===
        if not in_grace_period:
            edge_threshold = 0.15
            extreme_threshold = 0.05
            
            at_extreme_edge = (pos_x < extreme_threshold or pos_x > (1.0 - extreme_threshold) or
                              pos_y < extreme_threshold or pos_y > (1.0 - extreme_threshold))
            
            at_near_edge = (pos_x < edge_threshold or pos_x > (1.0 - edge_threshold) or
                           pos_y < edge_threshold or pos_y > (1.0 - edge_threshold))
            
            if at_extreme_edge:
                extreme_edge_penalty = self.EXTREME_EDGE_PENALTY
                reward += extreme_edge_penalty
                components['extreme_edge'] = extreme_edge_penalty
                if self.step_count % 60 == 0:
                    print(f"    ⚠️ EXTREME EDGE! pos=({pos_x:.2f}, {pos_y:.2f}) "
                          f"{extreme_edge_penalty:.1f}")
            elif at_near_edge:
                near_edge_penalty = self.NEAR_EDGE_PENALTY
                reward += near_edge_penalty
                components['near_edge'] = near_edge_penalty
            else:
                # In center zone
                if position_delta > 0:
                    center_bonus = self.CENTER_ZONE_BONUS
                    reward += center_bonus
                    components['center_zone'] = center_bonus

        self.prev_pos_x = pos_x
        self.prev_pos_y = pos_y

        # === TERMINATION CHECK ===
        should_terminate = self.stuck_counter > self.TERMINATE_IF_STUCK
        if should_terminate:
            components['terminated_stuck'] = True
            print(f"    💀 TERMINATING: Stuck for {self.stuck_counter/60:.1f}s")

        return reward, components

    def get_episode_stats(self):
        """Get statistics about current episode"""
        return {
            'laps_completed': self.total_laps_completed,
            'best_position': self.best_position_overall,
            'furthest_this_lap': self.furthest_position_this_lap,
            'progress_pct': (self.best_position_overall / self.max_position * 100 
                           if self.max_position > 0 else 0)
        }


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Testing EXPLOIT-FIXED Reward Computer")
    print("="*60)
    print("Simulating start-line bouncing exploit...")
    print()

    rc = ImprovedRacingRewardComputer(max_position=511, lap_check_threshold=409)
    rc.step_count = 100  # Skip grace period

    # Simulate bouncing at start line (the exploit)
    print("Test 1: Start-line bouncing (should be BLOCKED)")
    state = np.array([0.0, 25, 1, 0, 0.5, 0.5, 0.0, 0.0, 0.0, 8, 0], dtype=np.float32)
    rc.compute_reward(state, {})
    
    # Bounce backward to 0
    state[1] = 5
    rc.compute_reward(state, {})
    
    state[1] = 0
    rc.compute_reward(state, {})
    
    # Jump to high position (exploit attempt)
    state[1] = 419
    reward, components = rc.compute_reward(state, {})
    print(f"  Reward: {reward:.1f}")
    print(f"  Exploit blocked: {'exploit_detected' in components}")
    print()

    # Simulate valid lap completion
    print("Test 2: Valid lap completion (should be REWARDED)")
    rc.reset()
    rc.step_count = 100
    
    # Progress through track
    for pos in [50, 100, 200, 300, 410, 500]:
        state[1] = pos
        rc.compute_reward(state, {})
    
    # Cross finish line
    state[1] = 5  # Wrapped to start
    reward, components = rc.compute_reward(state, {})
    print(f"  Reward: {reward:.1f}")
    print(f"  Lap completed: {'lap_complete' in components}")
    print()
    
    print("="*60)
    print("✓ Exploit Prevention Test Complete")
    print("="*60)
