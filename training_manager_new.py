#!/usr/bin/env python3
"""
Training Manager for Mario Kart RL
Handles training loop, saving/loading, and algorithm selection
"""
import argparse
import time
import signal
import numpy as np
from typing import Optional


class TrainingManager:
    """
    Manages training for any RL algorithm.
    Provides unified interface for training, saving, and resuming.
    """

    def __init__(self, env, agent, algorithm_name=None):
        """
        Initialize training manager.

        Args:
            env: Mario Kart environment
            agent: RL agent (must inherit from RLAgentBase)
            algorithm_name: Name of algorithm (auto-detected if None)
        """
        self.env = env
        self.agent = agent
        self.algorithm_name = algorithm_name or agent.get_algorithm_name()

        self.interrupted = False
        self.best_reward = -float('inf')
        self.start_time = None

        # Setup signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        self.interrupted = True
        print("\n\n⚠️  Training interrupted! Saving progress...")

    def train(self, num_episodes=200, save_freq=25, log_freq=5,
              warm_start_steps=0, start_episode=1):
        """
        Train the agent.

        Args:
            num_episodes: Total number of episodes to train
            save_freq: Save checkpoint every N episodes
            log_freq: Log statistics every N episodes
            warm_start_steps: Force accelerate for first N steps (0=disabled)
            start_episode: Episode to start from (for resuming)
        """
        self._print_header(num_episodes, start_episode, warm_start_steps)

        input("Press Enter to start training...")

        action_names = ["None", "Accel", "Brake", "Left", "Right",
                       "Accel+Left", "Accel+Right", "Brake+Left", "Brake+Right"]

        self.start_time = time.time()

        for episode in range(start_episode, num_episodes + 1):
            if self.interrupted:
                print(f"\nStopping at episode {episode}...")
                break

            # Run episode
            state, info = self.env.reset()
            episode_steps = 0
            done = False
            last_log = 0

            while not done and not self.interrupted:
                # Select action (with optional warm start)
                if episode_steps < warm_start_steps:
                    action, value, log_prob = self.agent.select_action(
                        state, training=True, force_action=1
                    )
                else:
                    action, value, log_prob = self.agent.select_action(
                        state, training=True
                    )

                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store transition
                #self.agent.store_transition(state, action, reward, value, log_prob, done)
                self.agent.store_transition(state, action, reward, value, log_prob, done, next_state)

                episode_steps += 1
                state = next_state

                # Log progress during episode
                if episode_steps - last_log >= 600:
                    warm = " [WARM]" if episode_steps < warm_start_steps else ""
                    print(f"  Step {episode_steps}{warm}: {action_names[action]}, "
                          f"position={info.get('position', '?')}, "
                          f"rank={info.get('rank', '?')}, "
                          f"lap={info.get('lap', '?')}")
                    last_log = episode_steps

            # Update agent
            metrics = self.agent.update(next_state)
            loss1, loss2, metric3, ep_reward, ep_length = metrics

            # Learning rate scheduling (if agent has schedulers)
            if hasattr(self.agent, 'actor_scheduler') and episode % 50 == 0:
                self.agent.actor_scheduler.step()
                self.agent.critic_scheduler.step()

            # Logging
            if episode % log_freq == 0:
                self._log_episode(episode, num_episodes, metrics, info)

            # Save checkpoint
            if episode % save_freq == 0:
                self._save_checkpoint(episode, ep_reward)

        # Final save on exit
        self._save_on_exit(episode if self.interrupted else num_episodes)
        self._print_summary()

    def _print_header(self, num_episodes, start_episode, warm_start_steps):
        """Print training header"""
        print("\n" + "="*70)
        if start_episode > 1:
            print(f"{self.algorithm_name.upper()} TRAINING (Resuming from Episode {start_episode})")
        else:
            print(f"{self.algorithm_name.upper()} TRAINING (Fresh Training)")
        print("="*70)
        print("Features:")
        print(f"  ✓ Algorithm: {self.algorithm_name}")
        print(f"  ✓ Episodes: {start_episode} → {num_episodes}")
        print(f"  ✓ Warm start: {warm_start_steps} steps ({'ENABLED' if warm_start_steps > 0 else 'DISABLED'})")
        print("  ✓ Grace period: 120 steps (~2s no penalties at race start)")
        print("  ✓ Press Ctrl+C to stop and save progress")
        print("="*70 + "\n")

    def _log_episode(self, episode, num_episodes, metrics, info):
        """Log episode statistics"""
        loss1, loss2, metric3, ep_reward, ep_length = metrics
        stats = self.agent.get_stats()
        elapsed = time.time() - self.start_time

        avg_reward = np.mean(stats.get('episode_rewards', [0])) if stats.get('episode_rewards') else 0
        avg_length = np.mean(stats.get('episode_lengths', [0])) if stats.get('episode_lengths') else 0

        print(f"\n{'='*70}")
        print(f"Episode {episode}/{num_episodes}")
        print(f"  Loss 1: {loss1:.4f}")
        print(f"  Loss 2: {loss2:.4f}")
        print(f"  Metric 3: {metric3:.4f}")
        print(f"  Episode reward: {ep_reward:.2f}")
        print(f"  Episode length: {ep_length} steps")
        print(f"  Avg reward (100): {avg_reward:.2f}")
        print(f"  Avg length (100): {avg_length:.1f}")
        print(f"  Final rank: {info.get('rank', '?')}")
        print(f"  Final position: {info.get('position', '?')}")
        print(f"  Laps completed: {info.get('laps_completed', 0)}")
        print(f"  Progress: {info.get('progress_pct', 0):.1f}%")
        print(f"  Elapsed: {elapsed/60:.1f} min")
        print(f"{'='*70}\n")

    def _save_checkpoint(self, episode, ep_reward):
        """Save checkpoint and track best model"""
        algo_short = self.algorithm_name.lower().replace('-', '').replace(' ', '_')
        filepath = f"./checkpoint_{algo_short}_ep{episode}.pt"
        self.agent.save(filepath, episode=episode)

        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
            self.agent.save(f"./best_{algo_short}_model.pt", episode=episode)
            print(f"🏆 New best! Reward: {self.best_reward:.2f}\n")

    def _save_on_exit(self, episode):
        """Save on exit (whether interrupted or completed)"""
        algo_short = self.algorithm_name.lower().replace('-', '').replace(' ', '_')

        if self.interrupted:
            interrupt_path = f"./interrupted_{algo_short}_ep{episode}.pt"
            self.agent.save(interrupt_path, episode=episode)
            print(f"\n✓ Progress saved to {interrupt_path}")
            print(f"Resume with: --resume {interrupt_path}")
        else:
            self.agent.save(f"./final_{algo_short}_model.pt", episode=episode)

    def _print_summary(self):
        """Print training summary"""
        stats = self.agent.get_stats()
        elapsed = (time.time() - self.start_time) / 60 if self.start_time else 0

        print("\n" + "="*70)
        if self.interrupted:
            print("TRAINING INTERRUPTED")
        else:
            print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Algorithm: {self.algorithm_name}")
        print(f"Time: {elapsed:.1f} minutes")
        print(f"Best reward: {self.best_reward:.2f}")
        if stats.get('episode_rewards'):
            print(f"Final avg reward: {np.mean(stats['episode_rewards']):.2f}")
        print("="*70)

    @staticmethod
    def load_agent(agent_class, checkpoint_path, **agent_kwargs):
        """
        Load an agent from checkpoint.

        Args:
            agent_class: Class of the agent (e.g., ActorCriticAgent)
            checkpoint_path: Path to checkpoint file
            **agent_kwargs: Arguments to pass to agent constructor

        Returns:
            agent: Loaded agent
            start_episode: Episode to resume from
        """
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        agent = agent_class(**agent_kwargs)
        start_episode = agent.load(checkpoint_path) + 1
        print(f"Resuming from episode {start_episode}")
        return agent, start_episode
