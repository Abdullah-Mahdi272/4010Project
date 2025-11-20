#!/usr/bin/env python3
"""
TD Q-Learning Agent for Mario Kart
Implements online Q-learning with neural network function approximation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from rl_agent_base import RLAgentBase


class QNetwork(nn.Module):
    """Q-Network for action-value function approximation"""

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


class QLearningAgent(RLAgentBase):
    """
    TD Q-Learning agent with neural network function approximation.
    Uses online updates (no experience replay).
    """

    def __init__(self, state_dim=11, action_dim=9, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        super().__init__(state_dim, action_dim)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-network
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim=128).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.losses = deque(maxlen=100)

        # Current episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        # Transition buffer (for online updates)
        self.transitions = []

    def select_action(self, state, training=True, force_action=None):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            training: Whether in training mode (exploration)
            force_action: Force a specific action (for warm-start)

        Returns:
            action: Selected action
            value: Q-value of selected action
            log_prob: None (not used in Q-learning)
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

    def store_transition(self, state, action, reward, value, log_prob, done):
        """
        Store transition for online Q-learning update.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Not used in Q-learning
            log_prob: Not used in Q-learning
            done: Whether episode ended
        """
        self.transitions.append((state, action, reward, done))
        self.current_episode_reward += reward
        self.current_episode_length += 1

    def update(self, next_state=None):
        """
        Update Q-network using online TD Q-learning.

        Args:
            next_state: Next state (for bootstrapping)

        Returns:
            Tuple of metrics: (loss, epsilon, 0, episode_reward, episode_length)
        """
        total_loss = 0.0
        num_updates = 0

        # Process all transitions from this episode
        for i, (state, action, reward, done) in enumerate(self.transitions):
            # Determine next state for this transition
            if i < len(self.transitions) - 1:
                transition_next_state = self.transitions[i + 1][0]
            else:
                transition_next_state = next_state

            # Perform Q-learning update
            loss = self._q_learning_update(state, action, reward,
                                          transition_next_state, done)
            total_loss += loss
            num_updates += 1

        # Average loss
        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        self.losses.append(avg_loss)

        # Track episode completion
        ep_reward = self.current_episode_reward
        ep_length = self.current_episode_length
        self.episode_rewards.append(ep_reward)
        self.episode_lengths.append(ep_length)

        # Reset episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.transitions = []

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return avg_loss, self.epsilon, 0.0, ep_reward, ep_length

    def _q_learning_update(self, state, action, reward, next_state, done):
        """
        Perform single Q-learning update step.

        Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)

        # Current Q-value
        current_q = self.q_network(state_tensor).gather(1, action_tensor.unsqueeze(1))

        # Target Q-value
        if done or next_state is None:
            target_q = reward_tensor
        else:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_q = self.q_network(next_state_tensor).max(1)[0]
                target_q = reward_tensor + self.gamma * next_q

        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def save(self, filepath, episode=None):
        """Save agent checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
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
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.episode_rewards = deque(checkpoint.get('episode_rewards', []), maxlen=100)
        self.episode_lengths = deque(checkpoint.get('episode_lengths', []), maxlen=100)
        return checkpoint.get('episode', 0)

    def get_algorithm_name(self):
        """Return algorithm name"""
        return "TD Q-Learning"

    def get_stats(self):
        """Get training statistics"""
        return {
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'losses': list(self.losses),
            'epsilon': self.epsilon
        }
