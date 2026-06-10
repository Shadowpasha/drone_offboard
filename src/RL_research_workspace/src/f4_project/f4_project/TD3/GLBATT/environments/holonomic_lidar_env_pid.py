import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from environments.holonomic_lidar_env import HolonomicLidarEnv

class HolonomicLidarEnvPID(HolonomicLidarEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        
        # PID Reward Shaping Constants
        self.kp = 0.21
        self.ki = 0.001
        self.kd = 0.00001
        self.error_gain = 0.1
        self.output_gain = 1.0
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.i_limit = 0.75      # Anti-windup limit
        self.output_limit = 2.0 # Output limit for reward scaling

    def step(self, action):
        # We reuse the base step but intercept the reward
        # Actually, base step handles movement and collision.
        # We can call super().step(action) and then overwrite the reward.
        
        obs, base_reward, terminated, truncated, info = super().step(action)
        
        # If terminal (goal or collision), keep base reward (200 or -100)
        if terminated:
            return obs, base_reward, terminated, truncated, info
            
        # Otherwise, calculate PID reward
        dist = np.linalg.norm(self.goal - self.pose[:2])
        error = dist * self.error_gain
        self.integral += error
        self.integral = np.clip(self.integral, -self.i_limit, self.i_limit)
        derivative = error - self.prev_error
        
        pid_signal = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        pid_signal *= self.output_gain
        pid_signal = np.clip(pid_signal, -self.output_limit, self.output_limit)
        
        # PID reward
        reward = -pid_signal - 0.1 # signal + time penalty
        
        # Safety penalty (already in base reward for non-terminal, but let's re-calculate or just add it)
        # In base class: reward = 20.0 * progress - 0.1 + obstacle_penalty
        # We want to replace the 'progress' part with PID.
        
        closest_laser = np.min(obs[:64])
        if closest_laser < 0.3:
             reward += -0.5 * (0.3 - closest_laser)**2
             
        self.prev_error = error
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Reset PID state
        self.integral = 0.0
        dist = np.linalg.norm(self.goal - self.pose[:2])
        self.prev_error = dist * self.error_gain
        
        return obs, info
