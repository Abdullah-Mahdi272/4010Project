The RL agent is trained using a custom reward scheme designed to prioritize:
1. Lap completion (highest priority)
2. Forward progress on the correct racing path
3. Avoiding stuck situations (escalating penalties)

The game state is tracked through position values that DECREASE as the player
progresses (e.g., position 419 → 0 indicates forward movement toward the
finish line).

REWARDS:
- Lap completion: +5000 (enables exploration while maintaining high priority)
- Forward progress: +20 per position moved
- Forward momentum: +50 for sustained progress (>3 positions/step)
- Consistency: +20 when moving forward with speed
- Speed: +1 per unit (secondary to progress)
- Rank improvement: +100 per position gained
- Coin collection: +5 per coin

PENALTIES (escalating severity):
- Going backwards: -100 to -300
- Getting stuck:
  - 0.5 seconds: -50/step (allows exploration)
  - 30 seconds: -250/step total (-200 additional)
  - 1 minute: -750/step total (-500 additional)
  - 2+ minutes: -1750/step total (-1000 additional), then episode terminates
- Off-track/water: -150 (disabled when already stuck to avoid double penalty)
- Wall collision: -60
- Time penalty: -0.05/step (encourages fast completion)


The reward scheme balances three critical objectives:

1. Lap completion dominates brief stuck situations
   - Lap completion (+5000) is worth 3.3x more than being stuck for 0.5
     seconds (-1500)
   - This allows the agent to explore and discover lap completion without
     being overly discouraged by brief mistakes

2. Forward progress is the primary path
   - Moving forward is heavily rewarded (+20/position)
   - Typical good timestep yields ~100-150 points
   - Agent learns the correct racing line through positive reinforcement

3. Extended stuck situations are catastrophic
   - Brief stuck (-50/step) allows exploration
   - Extended stuck escalates rapidly to prevent the agent from getting
     permanently stuck
   - Episode termination at 2 minutes prevents hopeless situations


Manual Testing (In-Game)

To manually test that the environment and reward system work correctly in
the game:

1. Start the game:
   cd ~/F25_4010/1.2/cloned_4010Project/rl_environment
   python3 mario_kart_env_improved.py

2. Navigate menus:
   - Wait for automated menu navigation (5-10 seconds)
   - The script automatically selects: 1P → Time Trial → OK → Character →
     Track → OK

3. Test race functionality:
   - Once in-race, observe the state printouts showing:
     - Position values (should decrease as you progress)
     - Speed, lap number, rank
     - Reward values per timestep
   - Let the race run or control manually with random actions

4. Verify reward behavior:
   - Forward progress: Reward should be positive when position decreases
   - Stuck situations: Reward should become increasingly negative if position
     doesn't change
   - Lap completion: Should see +5000 bonus when lap completes

5. Interrupt: Press Ctrl+C to stop


Running Training
----------------

Command:
cd ~/F25_4010/1.2/cloned_4010Project/rl_environment
python3 train_robust_ac.py

Training Parameters
-------------------

- Episodes: 200 (default, configurable in script)
- Algorithm: Actor-Critic (A2C)
- Estimated time: 8-10 hours for 200 episodes (varies by system)

Training Process
----------------

Early training (episodes 1-50):
- Agent explores randomly
- Frequently gets stuck (high negative rewards)
- Gradually learns to avoid walls and obstacles

Mid training (episodes 50-150):
- Agent learns basic forward movement
- Starts discovering the correct path
- Begins completing laps occasionally

Late training (episodes 150-200):
- Agent reliably completes laps
- Optimizes racing line
- Stuck situations become rare

Monitoring Training
-------------------

The training script outputs:
- Episode number and total reward
- Episode statistics (laps completed, best position reached)
- Actor and critic losses
- Checkpoints saved every 10 episodes

Models are saved to:
- actor_model_checkpoint_ep{N}.pth - Actor network weights
- critic_model_checkpoint_ep{N}.pth - Critic network weights

KEY FILES


reward_computer.py
Contains the ImprovedRacingRewardComputer class that implements the reward
scheme. This is the core logic that guides agent learning.

Key methods:
- compute_reward(state, info) - Computes reward for current timestep
- reset() - Resets episode statistics
- get_episode_stats() - Returns episode progress metrics

mario_kart_env_improved.py
OpenAI Gym environment wrapper for Mario Kart 64. Handles:
- Game state extraction (position, speed, lap, etc.)
- Action execution (button presses)
- Episode termination conditions
- Reward computation via reward_computer.py

train_robust_ac.py
Training script using Actor-Critic algorithm. Features:
- Neural network actor and critic models
- Episode rollout and gradient computation
- Model checkpointing
- Training statistics logging


The game state is an 11-element array:
[time, position, lap, split, posX, posY, speed, turnSpeed, angle, rank, coins]

Critical notes:
- Position DECREASES as you progress (419 → 0)
- Position wraps from low to high at lap completion
- Speed and spatial coordinates (posX, posY) detect movement
- All values are normalized to reasonable ranges

Game mechanics:
- The game runs at 60 FPS (60 frames per second)
- Position values range from 0-419 on most tracks
- Lap completion detected when position < 409 then wraps to ~419
- Stuck is measured in frames (30 frames = 0.5 seconds)

Reward tuning:
- Current values are optimized for exploration and lap completion
- Lap bonus (5000) is intentionally 3.3x greater than brief stuck penalty
  (1500 for 0.5s)
- This ratio enables exploration while maintaining strong gradients toward
  the goal

