
import numpy as np
import math
from environments.differential_lidar_env import DifferentialLidarEnv
from environments.holonomic_lidar_env import HolonomicLidarEnv
from environments.holonomic_lidar_env_moving import HolonomicLidarEnvMoving

class DifferentialTrapEnv(DifferentialLidarEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.trap_type = "u_shape"
        self.difficulty = 1.0 # 0.0 (Open) to 1.0 (Closed)

    def set_difficulty(self, level):
        self.difficulty = np.clip(level, 0.0, 1.0)

    def _spawn_robot(self):
        # Spawn robot with some variance
        self.pose = np.zeros(3)
        self.pose[0] = np.random.uniform(-7.0, -6.0) # Start further left
        self.pose[1] = np.random.uniform(-1.0, 1.0)
        self.pose[2] = np.random.uniform(-0.1, 0.1) # Facing roughly right
        self.target_pos = self.pose[:2].copy()

    def _spawn_goal(self):
        # Spawn goal with variance, further right
        self.goal = np.array([np.random.uniform(5.0, 6.0), np.random.uniform(-1.0, 1.0)])

    def _spawn_obstacle(self, shape="rect"):
        # This method is called in a loop by the parent class, but we want to define a specific layout.
        # So we override the reset process instead.
        pass

    def reset(self, seed=None, options=None):
        # Override reset to manually place traps
        if seed is not None:
            np.random.seed(seed)
            
        super(DifferentialLidarEnv, self).reset(seed=seed)
        
        # Clear random obstacles
        self.obstacles = []
        
        # Trap Design: A U-shape facing the robot
        # Robot is at (-4, 0), Goal is at (4, 0)
        # Trap center is at (0, 0)
        
        # Randomize dimensions to prevent overfitting
        trap_x = np.random.uniform(-1.0, 1.0)
        # Reduced max dimensions as per request
        back_wall_height = np.random.uniform(3.0, 4.0) 
        arm_length = np.random.uniform(2.0, 3.5)
        
        # Back wall of the U (blocking direct path)
        # Difficulty 0.0 -> Size 0.0 (Open)
        # Difficulty 1.0 -> Size back_wall_height (Closed)
        
        current_wall_height = back_wall_height * self.difficulty
        
        if current_wall_height > 0.1:
            self.obstacles.append({
                "type": "rect",
                "pos": np.array([trap_x, 0.0]),
                "size": np.array([1.0, current_wall_height]) 
            })
        
        # Top arm of the U
        self.obstacles.append({
            "type": "rect",
            "pos": np.array([trap_x - (arm_length/2) - 0.5, back_wall_height/2 + 0.5]),
            "size": np.array([arm_length, 1.0]) 
        })
        
        # Bottom arm of the U
        self.obstacles.append({
            "type": "rect",
            "pos": np.array([trap_x - (arm_length/2) - 0.5, -(back_wall_height/2 + 0.5)]),
            "size": np.array([arm_length, 1.0]) 
        })
        
        # Goal is reachable if you go around the arms
        
        self.prev_distance = np.linalg.norm(self.goal - self.pose[:2])
        self.last_action = np.zeros(2)
        self.prev_action = np.zeros(2)
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}

class DifferentialMovingEnv(DifferentialLidarEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.obs_speed_range = 0.04 
        self.num_obstacles = 6 # Standardized Peer Density
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Post-process: half of obstacles get velocity
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
                limit = 7.5
                if obs["pos"][0] > limit or obs["pos"][0] < -limit:
                    obs["vel"][0] *= -1
                    obs["pos"][0] = np.clip(obs["pos"][0], -limit, limit)
                if obs["pos"][1] > limit or obs["pos"][1] < -limit:
                    obs["vel"][1] *= -1
                    obs["pos"][1] = np.clip(obs["pos"][1], -limit, limit)

    def step(self, action):
        self._update_obstacles()
        return super().step(action)

class HolonomicBlindCorridorEnv(HolonomicLidarEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.max_steps = 800 # Increased for two-phase maneuver
        self.phase = 1
        
    def _spawn_robot(self):
        # Start at the far left
        self.pose = np.zeros(3)
        self.pose[0] = -6.5
        self.pose[1] = 0.0
        self.pose[2] = 0.0 # Facing forward (Right)
        self.target_pos = self.pose[:2].copy()

    def _spawn_goal(self):
        # Initial goal (Phase 1) - will be randomized in reset
        self.goal = np.array([2.0, 0.0])

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset phase
        self.phase = 1
        
        # Call grand-parent reset to skip random obstacle spawning
        obs, info = super().reset(seed=seed)
        
        # Clear any random obstacles
        self.obstacles = []
        
        # Randomized Corridor Parameters
        self.corridor_y = np.random.uniform(-2.0, 2.0)
        self.corridor_width = np.random.uniform(0.8, 1.3)
        self.corridor_len = np.random.uniform(3.5, 5.0)
        
        corridor_x = 0.0
        wall_thick = 0.5
        
        # Phase 1 Goal: At the dead end of this specific corridor
        self.goal = np.array([corridor_x + (self.corridor_len/2) - 0.5, self.corridor_y])
        
        # Back wall (Blocking the goal in Phase 2)
        self.obstacles.append({
            "type": "rect",
            "pos": np.array([corridor_x + (self.corridor_len/2), self.corridor_y]),
            "size": np.array([wall_thick, self.corridor_width + 2*wall_thick])
        })
        
        # Top Side Wall
        self.obstacles.append({
            "type": "rect",
            "pos": np.array([corridor_x, self.corridor_y + self.corridor_width/2 + wall_thick/2]),
            "size": np.array([self.corridor_len, wall_thick])
        })
        
        # Bottom Side Wall
        self.obstacles.append({
            "type": "rect",
            "pos": np.array([corridor_x, self.corridor_y - (self.corridor_width/2 + wall_thick/2)]),
            "size": np.array([self.corridor_len, wall_thick])
        })
        
        self.prev_distance = np.linalg.norm(self.goal - self.pose[:2])
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if self.phase == 1 and info.get("reached", False):
            # Phase 1 Success: Switch to Phase 2
            self.phase = 2
            self.goal = np.array([-6.5, 0.0]) # Exit goal
            self.prev_distance = np.linalg.norm(self.goal - self.pose[:2])
            
            # Special Phase Transition Reward
            reward += 10.0 # Standardized (Scale 0.1)
            
            # Don't terminate yet
            terminated = False
            # Update obs with new goal info
            obs = self._get_obs()
            
        return obs, reward, terminated, truncated, info

class LidarBlackoutMovingEnv(HolonomicLidarEnvMoving):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.blackout_timer = 0
        self.blackout_chance = 0.05 # Increased chance once warmup is over
        self.blackout_duration = 15 # 15 steps of total blindness
        self.warmup_steps = 50 

    def reset(self, seed=None, options=None):
        self.blackout_timer = 0
        self.episode_steps = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.episode_steps += 1
        
        # 1. Update Blackout Logic (Only after warmup)
        if self.blackout_timer > 0:
            self.blackout_timer -= 1
        elif self.episode_steps > self.warmup_steps:
            if np.random.random() < self.blackout_chance:
                self.blackout_timer = self.blackout_duration
            
        # 2. Step the base environment (Calculates Oracle Reward)
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 3. Apply Sensory Blindness to the AGENT'S observation
        if self.blackout_timer > 0:
            obs[:64] = 0.0 
            info["blackout"] = True
        else:
            info["blackout"] = False
            
        return obs, reward, terminated, truncated, info
