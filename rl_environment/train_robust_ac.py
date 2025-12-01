#!/usr/bin/env python3
"""
Actor-Critic (A2C) Training for Mario Kart RL

This script implements an Actor-Critic reinforcement learning agent for the Mario Kart
environment. The agent learns to play Mario Kart by training separate neural networks
for action selection (actor) and state value estimation (critic).

Architecture:
- Actor Network: Maps states to action probabilities (policy)
- Critic Network: Estimates state values for advantage calculation
- Both networks use 2 hidden layers (256 units) with LayerNorm and Dropout

Training Features:
- Generalized Advantage Estimation (GAE) for more stable learning
- Entropy regularization to encourage exploration
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling for improved convergence
- Optional warm start (force accelerate for N steps to learn basic movement)
- Checkpoint saving/loading for resuming training
- Graceful interruption handling (Ctrl+C saves progress)

Usage:
    python3 train_robust_ac.py [options]

Key Arguments:
    --project-path PATH    Path to directory containing super_mario_kart executable
                          (default: .. - assumes script is in rl_environment/ subfolder)
    --episodes N           Number of training episodes (default: 200)
    --warm-start N         Steps to force accelerate at start (default: 0)
    --resume PATH          Resume from checkpoint file
    --lr-actor FLOAT       Actor learning rate (default: 3e-4)
    --lr-critic FLOAT      Critic learning rate (default: 1e-3)

Example:
    # From rl_environment/ folder (uses parent directory by default)
    python3 train_robust_ac.py --episodes 300
    
    # With custom project path
    python3 train_robust_ac.py --project-path /path/to/game --episodes 300
    
    # Resume from checkpoint
    python3 train_robust_ac.py --resume checkpoint_ac_ep100.pt
    
    # Train with warm start (helps agent learn to accelerate)
    python3 train_robust_ac.py --warm-start 600
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import time
import os
from collections import deque
import sys
import signal

sys.path.insert(0, os.path.dirname(__file__))

from rl_agent_base import RLAgentBase


class ActorNetwork(nn.Module):
    """Policy network (actor)"""
    def __init__(self, state_dim=11, action_dim=9, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        logits = self.network(x)
        return logits


class CriticNetwork(nn.Module):
    """Value network (critic)"""
    def __init__(self, state_dim=11, hidden_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        value = self.network(x)
        return value.squeeze(-1)


class ActorCriticAgent(RLAgentBase):
    """
    Robust A2C agent with GAE, entropy regularization, and other improvements
    """
    def __init__(self,
                 state_dim=11,
                 action_dim=9,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 gamma=0.99,
                 gae_lambda=0.95,
                 entropy_coef=0.02,
                 value_coef=0.5,
                 max_grad_norm=0.5):

        super().__init__(state_dim, action_dim)
        print(f"Using device: {self.device}")

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.actor_losses = deque(maxlen=100)
        self.critic_losses = deque(maxlen=100)
        
        self.actor_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=50, gamma=0.95
        )
        self.critic_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=50, gamma=0.95
        )
    
    def select_action(self, state, training=True, force_action=None):
        """
        Select action using current policy

        Args:
            state: Current state
            training: If True, sample from distribution. If False, take argmax.
            force_action: If provided, force this action (for warm start)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits = self.actor(state_tensor)
            value = self.critic(state_tensor)

        if force_action is not None:
            log_prob = torch.log_softmax(action_logits, dim=-1)[0, force_action]
            return force_action, value.item(), log_prob

        action_probs = torch.softmax(action_logits, dim=-1)

        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), value.item(), log_prob
        else:
            action = action_probs.argmax()
            return action.item(), None, None
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in episode memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value=0.0):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        More stable than simple advantage = return - value
        """
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, next_state=None):
        """
        Update policy and value function using A2C with GAE
        """
        if len(self.rewards) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0
        
        if next_state is not None:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                next_value = self.critic(next_state_tensor).item()
        else:
            next_value = 0.0
        
        advantages = self.compute_gae(next_value)
        
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = advantages + values
        
        predicted_values = self.critic(states)
        critic_loss = nn.MSELoss()(predicted_values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        action_logits = self.actor(states)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages.detach()).mean()
        actor_loss = actor_loss - self.entropy_coef * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        episode_reward = sum(self.rewards)
        episode_length = len(self.rewards)
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []
        
        return (actor_loss.item(), critic_loss.item(), entropy.item(), 
                episode_reward, episode_length)
    
    def save(self, filepath, episode=None):
        """Save model and training state"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
        }
        if episode is not None:
            checkpoint['episode'] = episode
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        if 'actor_scheduler_state_dict' in checkpoint:
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
        if 'critic_scheduler_state_dict' in checkpoint:
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])

        print(f"Model loaded from {filepath}")

        return checkpoint.get('episode', 0)

    def get_algorithm_name(self):
        """Return algorithm name"""
        return "Actor-Critic (A2C)"

    def get_stats(self):
        """Return training statistics"""
        return {
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'actor_losses': list(self.actor_losses),
            'critic_losses': list(self.critic_losses)
        }


def train(env, agent, num_episodes=200, save_freq=25, log_freq=5, warm_start_steps=600, start_episode=1):
    """
    Train the agent

    Args:
        warm_start_steps: Number of steps to force accelerate-only (helps agent learn basic movement)
        start_episode: Episode number to start from (for resuming training)
    """
    print("\n" + "="*70)
    if start_episode > 1:
        print(f"ROBUST ACTOR-CRITIC TRAINING (Resuming from Episode {start_episode})")
    else:
        print("ROBUST ACTOR-CRITIC TRAINING (Fresh Training)")
    print("="*70)
    print("Features:")
    print("  ✓ Separate actor/critic networks")
    print("  ✓ GAE for advantage estimation")
    print("  ✓ Entropy regularization")
    print("  ✓ Gradient clipping")
    print("  ✓ Learning rate scheduling")
    print(f"  ✓ Warm start: {warm_start_steps} steps ({'ENABLED' if warm_start_steps > 0 else 'DISABLED'})")
    print("  ✓ Wall collision penalties active")
    print("  ✓ Grace period: 120 steps (~2s no penalties at race start)")
    print("  ✓ Press Ctrl+C to stop and save progress")
    print("="*70 + "\n")

    input("Press Enter to start training...")

    action_names = ["None", "Accel", "Brake", "Left", "Right",
                   "Accel+Left", "Accel+Right", "Brake+Left", "Brake+Right"]

    best_reward = -float('inf')
    start_time = time.time()
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n⚠️  Training interrupted! Saving progress...")

    signal.signal(signal.SIGINT, signal_handler)

    for episode in range(start_episode, num_episodes + 1):
        if interrupted:
            print(f"\nStopping at episode {episode}...")
            break

        state, info = env.reset()
        episode_steps = 0
        done = False
        last_log = 0

        while not done and not interrupted:
            if episode_steps < warm_start_steps:
                action, value, log_prob = agent.select_action(state, training=True, force_action=1)
            else:
                action, value, log_prob = agent.select_action(state, training=True)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_steps += 1
            state = next_state
            
            if episode_steps - last_log >= 600:
                warm = " [WARM]" if episode_steps < warm_start_steps else ""
                print(f"  Step {episode_steps}{warm}: {action_names[action]}, "
                      f"position={info.get('position', '?')}, "
                      f"rank={info.get('rank', '?')}, "
                      f"lap={info.get('lap', '?')}")
                last_log = episode_steps
        
        actor_loss, critic_loss, entropy, ep_reward, ep_length = agent.update(next_state)
        
        if episode % 50 == 0:
            agent.actor_scheduler.step()
            agent.critic_scheduler.step()
        
        if episode % log_freq == 0:
            avg_reward = np.mean(agent.episode_rewards) if len(agent.episode_rewards) > 0 else 0
            avg_length = np.mean(agent.episode_lengths) if len(agent.episode_lengths) > 0 else 0
            avg_actor_loss = np.mean(agent.actor_losses) if len(agent.actor_losses) > 0 else 0
            avg_critic_loss = np.mean(agent.critic_losses) if len(agent.critic_losses) > 0 else 0
            elapsed = time.time() - start_time
            
            print(f"\n{'='*70}")
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Actor loss: {actor_loss:.4f} (avg: {avg_actor_loss:.4f})")
            print(f"  Critic loss: {critic_loss:.4f} (avg: {avg_critic_loss:.4f})")
            print(f"  Entropy: {entropy:.4f}")
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
        
        if episode % save_freq == 0:
            filepath = f"./checkpoint_ac_ep{episode}.pt"
            agent.save(filepath, episode=episode)
            
            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save("./best_ac_model.pt")
                print(f"🏆 New best! Reward: {best_reward:.2f}\n")
    
    if interrupted:
        interrupt_path = f"./interrupted_ac_ep{episode-1}.pt"
        agent.save(interrupt_path, episode=episode-1)
        print(f"\n✓ Progress saved to {interrupt_path}")
        print(f"Resume with: --resume {interrupt_path}")

    print("\n" + "="*70)
    if interrupted:
        print("TRAINING INTERRUPTED")
    else:
        print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best reward: {best_reward:.2f}")
    if len(agent.episode_rewards) > 0:
        print(f"Final avg reward: {np.mean(agent.episode_rewards):.2f}")

    if not interrupted:
        agent.save("./final_ac_model.pt")


def main():
    parser = argparse.ArgumentParser(description='Robust Actor-Critic Training')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--entropy-coef', type=float, default=0.02)
    parser.add_argument('--warm-start', type=int, default=0,
                       help='Steps to force accelerate at start (default: 0 = disabled)')
    parser.add_argument('--save-freq', type=int, default=25)
    parser.add_argument('--log-freq', type=int, default=5)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max-position', type=int, default=419)
    parser.add_argument('--lap-check', type=int, default=409)
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--project-path', type=str, default="..",
                       help='Path to project directory containing super_mario_kart executable (default: parent directory)')
    args = parser.parse_args()
    
    try:
        from mario_kart_env_improved import MarioKartEnv
    except ImportError:
        print("Error: Could not import MarioKartEnv")
        print("Make sure mario_kart_env_improved.py is in the same directory")
        return
    
    env = MarioKartEnv(
        project_path=args.project_path,
        debug=args.debug,
        max_steps=14400,
        normalize_state=True,
        max_position=args.max_position,
        lap_check_threshold=args.lap_check
    )

    agent = ActorCriticAgent(
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef
    )

    start_episode = 1
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}...")
        start_episode = agent.load(args.resume) + 1
        print(f"Resuming from episode {start_episode}")
    else:
        print("\nCreating agent (fresh training, no pretraining)...")

    print("\nWarming up neural networks (prevents first-episode delay)...")
    import numpy as np
    dummy_state = np.zeros(11, dtype=np.float32)
    _ = agent.select_action(dummy_state, training=False)
    print("✓ Neural networks initialized\n")

    train(env, agent,
          num_episodes=args.episodes,
          save_freq=args.save_freq,
          log_freq=args.log_freq,
          warm_start_steps=args.warm_start,
          start_episode=start_episode)
    
    env.close()


if __name__ == '__main__':
    main()
