#!/usr/bin/env python3
"""
Main entry point for Mario Kart RL Training
Supports multiple algorithms with unified interface
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mario_kart_env_improved import MarioKartEnv
from training_manager_new import TrainingManager


# Algorithm registry
ALGORITHMS = {
    'ac': {
        'name': 'Actor-Critic (A2C)',
        'module': 'train_robust_ac',
        'class': 'ActorCriticAgent',
        'default_params': {
            'lr_actor': 3e-4,
            'lr_critic': 1e-3,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'entropy_coef': 0.02
        }
    },
    'dqn': {
        'name': 'Deep Q-Network (DQN)',
        'module': 'train_dqn_new',  # To be implemented
        'class': 'DQNAgent',
        'default_params': {
            'lr': 1e-3,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995
        }
    },
    'qlearning': {
        'name': 'TD Q-Learning',
        'module': 'train_qlearning',  # To be implemented
        'class': 'QLearningAgent',
        'default_params': {
            'lr': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995
        }
    }
}


def main():
    parser = argparse.ArgumentParser(
        description='Mario Kart RL Training - Unified Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Actor-Critic for 200 episodes
  python3 train.py --algo ac --episodes 200

  # Resume Actor-Critic training
  python3 train.py --algo ac --episodes 200 --resume checkpoint_actorcritic_ep50.pt

  # Train DQN (when implemented)
  python3 train.py --algo dqn --episodes 200

Available algorithms: ac (Actor-Critic), dqn (DQN), qlearning (TD Q-Learning)
        """
    )

    # Algorithm selection
    parser.add_argument('--algo', type=str, required=True, choices=ALGORITHMS.keys(),
                       help='RL algorithm to use')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of episodes to train')
    parser.add_argument('--save-freq', type=int, default=25,
                       help='Save checkpoint every N episodes')
    parser.add_argument('--log-freq', type=int, default=5,
                       help='Log statistics every N episodes')
    parser.add_argument('--warm-start', type=int, default=0,
                       help='Force accelerate for first N steps (0=disabled)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Environment parameters
    parser.add_argument('--max-position', type=int, default=419,
                       help='Max track position value')
    parser.add_argument('--lap-check', type=int, default=409,
                       help='Lap completion threshold')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')

    args = parser.parse_args()

    # Get algorithm info
    algo_info = ALGORITHMS[args.algo]

    print("\n" + "="*70)
    print(f"Mario Kart RL Training - {algo_info['name']}")
    print("="*70)

    # Import the algorithm module dynamically
    try:
        module = __import__(algo_info['module'])
        agent_class = getattr(module, algo_info['class'])
    except (ImportError, AttributeError) as e:
        print(f"\n❌ Error: {algo_info['name']} not yet implemented!")
        print(f"   Module: {algo_info['module']}.py")
        print(f"   Class: {algo_info['class']}")
        print(f"\n   {str(e)}")
        return

    # Create environment
    print("\nInitializing environment...")
    env = MarioKartEnv(
        debug=args.debug,
        max_steps=14400,
        normalize_state=True,
        max_position=args.max_position,
        lap_check_threshold=args.lap_check
    )

    # Create or load agent
    agent_params = algo_info['default_params'].copy()

    if args.resume:
        agent, start_episode = TrainingManager.load_agent(
            agent_class, args.resume, **agent_params
        )
    else:
        print(f"\nCreating {algo_info['name']} agent...")
        agent = agent_class(**agent_params)
        start_episode = 1

        # PyTorch warm-up pass (for neural network based agents)
        if hasattr(agent, 'actor'):
            print("\nWarming up neural networks (prevents first-episode delay)...")
            import numpy as np
            dummy_state = np.zeros(11, dtype=np.float32)
            _ = agent.select_action(dummy_state, training=False)
            print("✓ Neural networks initialized\n")

    # Create training manager
    manager = TrainingManager(env, agent, algo_info['name'])

    # Train
    manager.train(
        num_episodes=args.episodes,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        warm_start_steps=args.warm_start,
        start_episode=start_episode
    )

    env.close()


if __name__ == '__main__':
    main()
