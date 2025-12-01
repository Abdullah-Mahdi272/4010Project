#!/usr/bin/env python3
"""
Deep Q-Network (DQN) Agent for Mario Kart
Implements DQN with experience replay and target network
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from rl_agent_base import RLAgentBase


class QNetwork(nn.Module):
    """Deep Q-Network for action-value function approximation"""

    def __init__(self, state_dim=11, action_dim=9, hidden_dim=128):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        """Forward pass returns Q-values for all actions"""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent(RLAgentBase):
    """
    Deep Q-Network agent with experience replay and target network.
    """

    def __init__(self, state_dim=11, action_dim=9, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=10):
        super().__init__(state_dim, action_dim)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.losses = deque(maxlen=100)

        # Current episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.update_counter = 0

    def select_action(self, state, training=True, force_action=None):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            training: Whether in training mode (exploration)
            force_action: Force a specific action (for warm-start)

        Returns:
            action: Selected action
            value: Q-value of selected action (for logging)
            log_prob: None (not used in DQN)
        """
        if force_action is not None:
            return force_action, 0.0, None

        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            value = 0.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(1).item()
                value = q_values[0, action].item()

        return action, value, None

    def store_transition(self, state, action, reward, value, log_prob, done, next_state):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Not used in DQN
            log_prob: Not used in DQN
            done: Whether episode ended
        """
        # Store for current episode tracking
        self.current_episode_reward += reward
        self.current_episode_length += 1

        # Store transition (next_state will be set in update)
        #self.last_transition = (state, action, reward, done)
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self, next_state=None):
        """
        Update Q-network using experience replay.

        Args:
            next_state: Next state (for completing last transition)

        Returns:
            Tuple of metrics: (loss, epsilon, 0, episode_reward, episode_length)
        """
        # Complete last transition
        #if hasattr(self, 'last_transition'):
            #state, action, reward, done = self.last_transition
        #self.replay_buffer.push(state, action, reward, next_state, done)

        # Track episode completion
        ep_reward = self.current_episode_reward
        ep_length = self.current_episode_length
        self.episode_rewards.append(ep_reward)
        self.episode_lengths.append(ep_length)

        # Reset episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        # Only train if buffer has enough samples
        loss = 0.0
        if len(self.replay_buffer) >= self.batch_size:
            loss = self._train_step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss, self.epsilon, 0.0, ep_reward, ep_length

    def _train_step(self):
        """Perform one training step using experience replay"""
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        loss_value = loss.item()
        self.losses.append(loss_value)
        return loss_value

    def save(self, filepath, episode=None):
        """Save agent checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode or 0,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
        }, filepath)

    def load(self, filepath):
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.episode_rewards = deque(checkpoint.get('episode_rewards', []), maxlen=100)
        self.episode_lengths = deque(checkpoint.get('episode_lengths', []), maxlen=100)
        return checkpoint.get('episode', 0)

    def get_algorithm_name(self):
        """Return algorithm name"""
        return "Deep Q-Network (DQN)"

    def get_stats(self):
        """Get training statistics"""
        return {
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'losses': list(self.losses),
            'epsilon': self.epsilon
        }
