import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from environments.holonomic_lidar_env import HolonomicLidarEnv

class HolonomicLidarEnvMoving(HolonomicLidarEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.obs_speed_range = 0.04 
        self.num_obstacles = 6 # Standardized Peer Density
        
    def reset(self, seed=None, options=None):
        # We want to ensure 3 are static and 3 are moving
        # We'll override the spawning logic or just post-process them
        obs, info = super().reset(seed=seed, options=options)
        
        # Post-process: half of obstacles get velocity
        # self.obstacles is a list of dicts from base class
        for i, obs_dict in enumerate(self.obstacles):
            if i >= self.num_obstacles // 2: # Second half are moving
                speed = np.random.uniform(0.02, self.obs_speed_range)
                direction = np.random.choice([-1, 1])
                # Perpendicular Patch: Move only on Y-axis (Cross-Traffic)
                obs_dict["vel"] = np.array([0.0, speed * direction])
            else:
                obs_dict["vel"] = np.zeros(2)
        
        return obs, info

    def _update_obstacles(self):
        for obs in self.obstacles:
            if "vel" in obs and np.any(obs["vel"] != 0):
                # Move
                obs["pos"] += obs["vel"]
                
                # Bounce off boundaries (-8 to 8)
                # Adjusted for object size
                limit = 7.5
                if obs["pos"][0] > limit or obs["pos"][0] < -limit:
                    obs["vel"][0] *= -1
                    obs["pos"][0] = np.clip(obs["pos"][0], -limit, limit)
                if obs["pos"][1] > limit or obs["pos"][1] < -limit:
                    obs["vel"][1] *= -1
                    obs["pos"][1] = np.clip(obs["pos"][1], -limit, limit)

    def step(self, action):
        # 1. Update obstacles
        self._update_obstacles()
        
        # 2. Call base step
        return super().step(action)
