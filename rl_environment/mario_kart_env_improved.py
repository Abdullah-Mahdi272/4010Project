#!/usr/bin/env python3
"""
Improved Mario Kart Environment
- Proper position-based progress tracking  
- Better reward computation
- State normalization
- Configurable track parameters
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import time
import os
import sys
import select
import threading
from typing import Optional, Tuple, Dict
from collections import deque

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

# Import reward computer
sys.path.insert(0, os.path.dirname(__file__))
try:
    from reward_computer import ImprovedRacingRewardComputer
except ImportError:
    print("Warning: reward_computer.py not found, using built-in rewards")
    ImprovedRacingRewardComputer = None


class MarioKartEnv(gym.Env):
    """Improved Mario Kart RL Environment"""
    
    def __init__(self, 
                 project_path=None, 
                 max_steps=14400,
                 debug=False,
                 normalize_state=True,
                 max_position=419,
                 lap_check_threshold=409):
        super().__init__()
        
        if project_path is None:
            project_path = "/home/student/F25_4010/1.2/cloned_4010Project"
        
        self.game_executable = os.path.join(project_path, "super_mario_kart")
        self.working_directory = project_path
        self.debug = debug
        self.normalize_state = normalize_state
        
        if not os.path.exists(self.game_executable):
            raise FileNotFoundError(f"Game not found: {self.game_executable}")
        
        self.max_steps = max_steps
        self.process = None
        self.game_started = False
        
        # Track parameters
        self.max_position = max_position
        self.lap_check_threshold = lap_check_threshold
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        
        # Reward computer
        if ImprovedRacingRewardComputer:
            self.reward_computer = ImprovedRacingRewardComputer(
                max_position=max_position,
                lap_check_threshold=lap_check_threshold
            )
        else:
            self.reward_computer = None
        
        # State tracking
        self.last_state = None
        self.state_buffer = deque(maxlen=20)
        self.reader_thread = None
        self.reader_running = False
        
        # Define spaces
        self.action_space = spaces.Discrete(9)
        
        if normalize_state:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(11,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
            )
        
        print(f"Mario Kart RL Environment initialized")
        print(f"  Max steps: {max_steps} (~{max_steps/60:.0f}s)")
        print(f"  Track: max_position={max_position}")
        print(f"  State normalization: {normalize_state}")
    
    def _log(self, msg):
        if self.debug:
            print(f"[ENV] {msg}")
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to improve learning"""
        if not self.normalize_state:
            return state
        
        normalized = state.copy()
        normalized[0] = min(state[0] / (self.max_steps / 60.0), 1.0)  # time
        normalized[1] = state[1] / self.max_position  # position
        normalized[2] = min(state[2] / 5.0, 1.0)  # lap
        normalized[3] = min(state[3] / 10.0, 1.0)  # split
        normalized[4] = state[4]  # pos_x (already 0-1)
        normalized[5] = state[5]  # pos_y (already 0-1)
        normalized[6] = np.clip(state[6] / 2.0, -1.0, 1.0)  # speed
        normalized[7] = np.clip(state[7] / 0.5, -1.0, 1.0)  # turn_speed
        normalized[8] = np.clip((state[8] - 180.0) / 180.0, -1.0, 1.0)  # angle
        normalized[9] = (state[9] - 1) / 7.0  # rank (1-8)
        normalized[10] = min(state[10] / 10.0, 1.0)  # coins
        
        return normalized
    
    def _start_reader_thread(self):
        if self.reader_thread is not None and self.reader_thread.is_alive():
            return
        
        self.reader_running = True
        self.reader_thread = threading.Thread(target=self._read_game_output, daemon=True)
        self.reader_thread.start()
    
    def _read_game_output(self):
        while self.reader_running and self.process and self.process.poll() is None:
            try:
                line = self._read_line_blocking(timeout=0.05)
                if line:
                    state = self._parse_state(line)
                    if state is not None:
                        self.state_buffer.append(state)
                        self.last_state = state

                    if "EPISODE_END" in line or "RACE_END" in line:
                        self.state_buffer.append(("END", line))
            except:
                pass
    
    def _start_game_once(self):
        if self.process is not None:
            return
        
        print("\n" + "="*60)
        print("STARTING GAME")
        print("="*60)
        
        env = os.environ.copy()
        
        self.process = subprocess.Popen(
            [self.game_executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
            text=True,
            cwd=self.working_directory
        )
        
        import fcntl
        flags = fcntl.fcntl(self.process.stdout, fcntl.F_GETFL)
        fcntl.fcntl(self.process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        
        print(f"Game process started (PID: {self.process.pid})")
        
        self._start_reader_thread()
        
        if HAS_PYAUTOGUI:
            print("\nNavigating menus...")
            time.sleep(3.0)
            pyautogui.press('return')
            time.sleep(2.0)
            pyautogui.press('down')
            pyautogui.press('return')
            time.sleep(1.5)
            pyautogui.press('return')
            time.sleep(1.5)
            pyautogui.press('return')
            time.sleep(1.5)
            pyautogui.press('return')
            time.sleep(1.0)
            pyautogui.press('return')
            time.sleep(2.0)
            pyautogui.press('return')
            time.sleep(3.0)
        else:
            print("\nManual navigation required - press Enter when ready...")
            input()
        
        self.game_started = True
        print("✓ Ready!\n")
    
    def _read_line_blocking(self, timeout=0.1) -> Optional[str]:
        if self.process is None:
            return None
        
        try:
            ready, _, _ = select.select([self.process.stdout], [], [], timeout)
            if ready:
                line = self.process.stdout.readline()
                if line:
                    return line.strip()
        except:
            pass
        
        return None
    
    def _parse_state(self, line: str) -> Optional[np.ndarray]:
        if not line.startswith("RL_STATE|"):
            return None
        
        try:
            parts = line.split("|")[1:]
            if len(parts) != 11:
                return None
            
            state = np.array([float(x) for x in parts], dtype=np.float32)
            return state
        except:
            return None
    
    def _send_action(self, action: int):
        if self.process is None or self.process.stdin is None:
            return
        
        accel = 1 if action in [1, 5, 6] else 0
        brake = 1 if action in [2, 7, 8] else 0
        left = 1 if action in [3, 5, 7] else 0
        right = 1 if action in [4, 6, 8] else 0
        
        try:
            cmd = f"ACTION|{accel}|{brake}|{left}|{right}\n"
            self.process.stdin.write(cmd)
            self.process.stdin.flush()
        except:
            pass
    
    def _get_latest_state(self) -> Optional[np.ndarray]:
        if len(self.state_buffer) > 0:
            item = self.state_buffer[-1]
            if isinstance(item, np.ndarray):
                return item
        return self.last_state
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if not self.game_started:
            self._start_game_once()
            time.sleep(2.0)

        # Send reset signal to game
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write("RESET\n")
                # Added more reset signals confirm the agent resets after being stuck
                self.process.stdin.write("RESET\n")
                self.process.stdin.write("RESET\n")
                self.process.stdin.flush()
            except:
                pass

        # Clear buffers
        self.state_buffer.clear()
        self.current_step = 0
        self.episode_count += 1

        if self.reward_computer:
            self.reward_computer.reset()

        # Wait for first valid state while sending accelerate commands
        # This ensures input buffer has accelerate ready when race starts
        max_wait = 300  # 3 seconds max
        state = None
        for i in range(max_wait):
            state = self._get_latest_state()
            if state is not None and not np.all(state == 0):
                break
            # While waiting, send accelerate so it's queued
            if i % 2 == 0:
                self._send_action(1)
            time.sleep(0.01)

        if state is None:
            state = self.last_state if self.last_state is not None else np.zeros(11, dtype=np.float32)

        normalized_state = self._normalize_state(state)
        info = self._make_info(state)
        self.last_state = state

        # Send burst of accelerate commands to ensure immediate movement
        for _ in range(20):
            self._send_action(1)
            time.sleep(0.001)

        return normalized_state, info
    
    def _make_info(self, state: np.ndarray) -> Dict:
        info = {
            'time': float(state[0]),
            'position': int(state[1]),
            'lap': int(state[2]),
            'split': int(state[3]),
            'pos_x': float(state[4]),
            'pos_y': float(state[5]),
            'speed': float(state[6]),
            'turn_speed': float(state[7]),
            'angle': float(state[8]),
            'rank': int(state[9]),
            'coins': int(state[10])
        }
        
        if self.reward_computer:
            info.update(self.reward_computer.get_episode_stats())
        
        return info
    
    def step(self, action: int):
        # Send action
        for _ in range(2):
            self._send_action(action)
            time.sleep(0.003)
        
        self.current_step += 1
        
        # Check for end
        end_detected = False
        for item in list(self.state_buffer):
            if isinstance(item, tuple) and item[0] == "END":
                end_detected = True
                break
        
        time.sleep(0.016)
        
        state = self._get_latest_state()
        if state is None:
            state = self.last_state if self.last_state is not None else np.zeros(11, dtype=np.float32)
        
        normalized_state = self._normalize_state(state)
        
        # Compute reward
        if self.reward_computer:
            reward, reward_components = self.reward_computer.compute_reward(state, {})
        else:
            # Fallback reward
            reward = float(state[6]) - 0.01  # speed - time penalty
            reward_components = {}
        
        info = self._make_info(state)
        info['reward_components'] = reward_components
        
        terminated = end_detected
        truncated = self.current_step >= self.max_steps
        
        # Early termination if stuck
        if self.current_step > 1800 and int(state[1]) > self.max_position * 0.9:
            truncated = True
            reward -= 100
        
        return normalized_state, reward, terminated, truncated, info
    
    def close(self):
        self.reader_running = False
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
    
    def render(self):
        pass


if __name__ == '__main__':
    print("Testing Environment")
    env = MarioKartEnv(debug=True)
    obs, info = env.reset()
    print(f"Initial state: {obs}")
    
    for i in range(120):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
        if i % 30 == 0:
            print(f"Step {i}: reward={reward:.2f}, pos={info.get('position')}")
        
        if term or trunc:
            break
    
    env.close()
