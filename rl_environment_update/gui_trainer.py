#!/usr/bin/env python3
"""
GUI Trainer for Mario Kart RL
Simple interface to train, save, and resume RL agents
"""
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mario_kart_env_improved import MarioKartEnv
from training_manager import TrainingManager
import numpy as np


# Algorithm registry
ALGORITHMS = {
    'Actor-Critic (A2C)': {
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
    'Deep Q-Network (DQN)': {
        'module': 'train_dqn',
        'class': 'DQNAgent',
        'default_params': {
            'lr': 1e-3,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995
        }
    },
    'TD Q-Learning': {
        'module': 'train_qlearning',
        'class': 'QLearningAgent',
        'default_params': {
            'lr': 1e-3,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995
        }
    }
}


class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mario Kart RL Trainer")
        self.root.geometry("800x700")

        self.training_thread = None
        self.training_active = False
        self.checkpoint_path = None

        self.create_widgets()

    def create_widgets(self):
        """Create GUI widgets"""
        # Title
        title = tk.Label(self.root, text="Mario Kart RL Trainer",
                        font=("Arial", 20, "bold"))
        title.pack(pady=10)

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Algorithm selection
        algo_frame = ttk.LabelFrame(main_frame, text="Algorithm", padding="10")
        algo_frame.pack(fill=tk.X, pady=5)

        ttk.Label(algo_frame, text="Select Algorithm:").grid(row=0, column=0, sticky=tk.W)
        self.algo_var = tk.StringVar(value="Actor-Critic (A2C)")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algo_var,
                                  values=list(ALGORITHMS.keys()),
                                  state="readonly", width=30)
        algo_combo.grid(row=0, column=1, padx=10)

        # Training parameters
        params_frame = ttk.LabelFrame(main_frame, text="Training Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=5)

        ttk.Label(params_frame, text="Episodes:").grid(row=0, column=0, sticky=tk.W)
        self.episodes_var = tk.StringVar(value="200")
        ttk.Entry(params_frame, textvariable=self.episodes_var, width=10).grid(row=0, column=1, padx=10)

        ttk.Label(params_frame, text="Save Frequency:").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.save_freq_var = tk.StringVar(value="25")
        ttk.Entry(params_frame, textvariable=self.save_freq_var, width=10).grid(row=0, column=3, padx=10)

        ttk.Label(params_frame, text="Log Frequency:").grid(row=1, column=0, sticky=tk.W)
        self.log_freq_var = tk.StringVar(value="5")
        ttk.Entry(params_frame, textvariable=self.log_freq_var, width=10).grid(row=1, column=1, padx=10)

        ttk.Label(params_frame, text="Warm Start Steps:").grid(row=1, column=2, sticky=tk.W, padx=(20,0))
        self.warm_start_var = tk.StringVar(value="0")
        ttk.Entry(params_frame, textvariable=self.warm_start_var, width=10).grid(row=1, column=3, padx=10)

        # Resume checkpoint
        resume_frame = ttk.LabelFrame(main_frame, text="Resume Training (Optional)", padding="10")
        resume_frame.pack(fill=tk.X, pady=5)

        self.checkpoint_label = tk.Label(resume_frame, text="No checkpoint selected",
                                        fg="gray")
        self.checkpoint_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(resume_frame, text="Load Checkpoint",
                  command=self.load_checkpoint).pack(side=tk.LEFT, padx=5)
        ttk.Button(resume_frame, text="Clear",
                  command=self.clear_checkpoint).pack(side=tk.LEFT, padx=5)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        self.start_button = ttk.Button(control_frame, text="Start Training",
                                       command=self.start_training,
                                       style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Training",
                                      command=self.stop_training,
                                      state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Progress
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_label = tk.Label(progress_frame, text="Ready to start",
                                      font=("Arial", 10))
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Output log
        log_frame = ttk.LabelFrame(main_frame, text="Training Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15,
                                                  state=tk.DISABLED,
                                                  font=("Courier", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def load_checkpoint(self):
        """Load checkpoint file"""
        filepath = filedialog.askopenfilename(
            title="Select Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")]
        )
        if filepath:
            self.checkpoint_path = filepath
            filename = os.path.basename(filepath)
            self.checkpoint_label.config(text=f"Checkpoint: {filename}", fg="green")

    def clear_checkpoint(self):
        """Clear checkpoint selection"""
        self.checkpoint_path = None
        self.checkpoint_label.config(text="No checkpoint selected", fg="gray")

    def log(self, message):
        """Add message to log"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_progress(self, current, total):
        """Update progress bar"""
        progress = (current / total) * 100
        self.progress_bar['value'] = progress
        self.progress_label.config(text=f"Episode {current}/{total}")

    def start_training(self):
        """Start training in separate thread"""
        if self.training_active:
            return

        # Get parameters
        try:
            episodes = int(self.episodes_var.get())
            save_freq = int(self.save_freq_var.get())
            log_freq = int(self.log_freq_var.get())
            warm_start = int(self.warm_start_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric parameters")
            return

        # Disable controls
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.training_active = True

        # Clear log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        # Start training thread
        self.training_thread = threading.Thread(
            target=self.run_training,
            args=(episodes, save_freq, log_freq, warm_start),
            daemon=True
        )
        self.training_thread.start()

    def stop_training(self):
        """Stop training (will save on next episode)"""
        self.training_active = False
        self.log("⚠️  Stop requested - will save at next checkpoint...\n")
        self.stop_button.config(state=tk.DISABLED)

    def run_training(self, episodes, save_freq, log_freq, warm_start):
        """Run training (in separate thread)"""
        try:
            # Get algorithm info
            algo_name = self.algo_var.get()
            algo_info = ALGORITHMS[algo_name]

            self.log(f"{'='*70}\n")
            self.log(f"Mario Kart RL Training - {algo_name}\n")
            self.log(f"{'='*70}\n\n")

            # Import algorithm
            module = __import__(algo_info['module'])
            agent_class = getattr(module, algo_info['class'])

            # Create environment
            self.log("Initializing environment...\n")
            env = MarioKartEnv(
                debug=False,
                max_steps=14400,
                normalize_state=True,
                max_position=419,
                lap_check_threshold=409
            )

            # Create or load agent
            agent_params = algo_info['default_params'].copy()

            if self.checkpoint_path:
                self.log(f"Loading checkpoint from {self.checkpoint_path}...\n")
                agent, start_episode = TrainingManager.load_agent(
                    agent_class, self.checkpoint_path, **agent_params
                )
            else:
                self.log(f"Creating {algo_name} agent...\n")
                agent = agent_class(**agent_params)
                start_episode = 1

                # PyTorch warm-up pass
                if hasattr(agent, 'actor') or hasattr(agent, 'q_network'):
                    self.log("Warming up neural networks...\n")
                    dummy_state = np.zeros(11, dtype=np.float32)
                    _ = agent.select_action(dummy_state, training=False)
                    self.log("✓ Neural networks initialized\n\n")

            # Create custom training manager with GUI callbacks
            manager = GUITrainingManager(
                env, agent, algo_name,
                log_callback=self.log,
                progress_callback=self.update_progress,
                should_stop_callback=lambda: not self.training_active
            )

            # Train
            manager.train(
                num_episodes=episodes,
                save_freq=save_freq,
                log_freq=log_freq,
                warm_start_steps=warm_start,
                start_episode=start_episode
            )

            env.close()

            self.log("\n✓ Training complete!\n")

        except Exception as e:
            self.log(f"\n❌ Error: {str(e)}\n")
            import traceback
            self.log(traceback.format_exc())

        finally:
            # Re-enable controls
            self.root.after(0, self._training_finished)

    def _training_finished(self):
        """Called when training finishes"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.training_active = False
        self.progress_bar['value'] = 0
        self.progress_label.config(text="Ready to start")


class GUITrainingManager(TrainingManager):
    """Training manager with GUI callbacks"""

    def __init__(self, env, agent, algorithm_name, log_callback, progress_callback, should_stop_callback):
        super().__init__(env, agent, algorithm_name)
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.should_stop_callback = should_stop_callback

    def train(self, num_episodes=200, save_freq=25, log_freq=5,
              warm_start_steps=0, start_episode=1):
        """Override train to use GUI callbacks"""
        import time

        self.log_callback(f"\n{'='*70}\n")
        self.log_callback(f"Starting training...\n")
        self.log_callback(f"Episodes: {start_episode} → {num_episodes}\n")
        self.log_callback(f"{'='*70}\n\n")

        action_names = ["None", "Accel", "Brake", "Left", "Right",
                       "Accel+Left", "Accel+Right", "Brake+Left", "Brake+Right"]

        self.start_time = time.time()

        for episode in range(start_episode, num_episodes + 1):
            if self.should_stop_callback():
                self.log_callback(f"\n⚠️  Training stopped by user at episode {episode}\n")
                break

            # Update progress
            self.progress_callback(episode, num_episodes)

            # Run episode
            state, info = self.env.reset()
            episode_steps = 0
            done = False

            while not done and not self.should_stop_callback():
                # Select action
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

                # 
                #  transition
                self.agent.store_transition(state, action, reward, value, log_prob, done, next_state)

                episode_steps += 1
                state = next_state

            # Update agent
            metrics = self.agent.update(next_state)
            loss1, loss2, metric3, ep_reward, ep_length = metrics

            # Learning rate scheduling
            if hasattr(self.agent, 'actor_scheduler') and episode % 50 == 0:
                self.agent.actor_scheduler.step()
                self.agent.critic_scheduler.step()

            # Logging
            if episode % log_freq == 0:
                self._log_episode(episode, num_episodes, metrics, info)

            # Save checkpoint
            if episode % save_freq == 0:
                self._save_checkpoint(episode, ep_reward)

        # Final save
        self._save_on_exit(episode if self.should_stop_callback() else num_episodes)
        self._print_summary()

    def _log_episode(self, episode, num_episodes, metrics, info):
        """Log episode (override to use callback)"""
        loss1, loss2, metric3, ep_reward, ep_length = metrics
        stats = self.agent.get_stats()
        import time
        elapsed = time.time() - self.start_time

        avg_reward = np.mean(stats.get('episode_rewards', [0])) if stats.get('episode_rewards') else 0
        avg_length = np.mean(stats.get('episode_lengths', [0])) if stats.get('episode_lengths') else 0

        self.log_callback(f"\n{'='*70}\n")
        self.log_callback(f"Episode {episode}/{num_episodes}\n")
        self.log_callback(f"  Episode reward: {ep_reward:.2f}\n")
        self.log_callback(f"  Episode length: {ep_length} steps\n")
        self.log_callback(f"  Avg reward (100): {avg_reward:.2f}\n")
        self.log_callback(f"  Final rank: {info.get('rank', '?')}\n")
        self.log_callback(f"  Final position: {info.get('position', '?')}\n")
        self.log_callback(f"  Laps completed: {info.get('laps_completed', 0)}\n")
        self.log_callback(f"  Elapsed: {elapsed/60:.1f} min\n")
        self.log_callback(f"{'='*70}\n\n")


def main():
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
