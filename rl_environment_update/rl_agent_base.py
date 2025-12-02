#!/usr/bin/env python3
"""
Base class for all RL agents
Provides common interface for training, saving, and loading
"""
from abc import ABC, abstractmethod
import torch
import os


class RLAgentBase(ABC):
    """
    Abstract base class that all RL agents must inherit from.
    Ensures consistent interface across different algorithms.
    """

    def __init__(self, state_dim=11, action_dim=9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def select_action(self, state, training=True, **kwargs):
        """
        Select an action given a state.

        Args:
            state: Current state observation
            training: Whether in training mode (exploration) or eval mode
            **kwargs: Algorithm-specific parameters

        Returns:
            action: Selected action
            value: State value estimate (if applicable, else None)
            log_prob: Log probability of action (if applicable, else None)
        """
        pass

    @abstractmethod
    def store_transition(self, state, action, reward, value, log_prob, done, next_state=None):
        """
        Store a transition in the agent's memory.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate (can be None for some algorithms)
            log_prob: Log probability (can be None for some algorithms)
            done: Whether episode ended
        """
        pass

    @abstractmethod
    def update(self, next_state=None):
        """
        Update the agent's policy/value function.

        Args:
            next_state: Next state (for bootstrapping, can be None)

        Returns:
            Tuple of metrics: (loss1, loss2, metric3, episode_reward, episode_length)
        """
        pass

    @abstractmethod
    def save(self, filepath, episode=None):
        """
        Save the agent's state to disk.

        Args:
            filepath: Path to save checkpoint
            episode: Current episode number (optional)
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        Load the agent's state from disk.

        Args:
            filepath: Path to checkpoint file

        Returns:
            episode: Episode number to resume from (0 if not available)
        """
        pass

    @abstractmethod
    def get_algorithm_name(self):
        """Return the name of the algorithm (e.g., 'Actor-Critic', 'DQN')"""
        pass

    @abstractmethod
    def get_stats(self):
        """
        Get training statistics.

        Returns:
            Dictionary with keys like 'episode_rewards', 'losses', etc.
        """
        pass

    def get_device(self):
        """Return the device (CPU/GPU) being used"""
        return self.device
