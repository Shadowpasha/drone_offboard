import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame
from holonomic_lidar_env import HolonomicLidarEnv

class Holonomic360LidarEnv(HolonomicLidarEnv):
    """
    A holonomic robot environment with a 360-degree lidar matching 
    the YDLIDAR T-mini Plus specifications:
    - FOV: 360 degrees
    - Max Range: 12.0 meters
    - Angular Resolution: ~0.54 degrees (667 rays)
    - Obstacles: Many small circular obstacles (radius 0.05m)
    """
    def __init__(self, render_mode=None):
        # We call super().__init__ but we'll override the critical constants
        super().__init__(render_mode=render_mode)
        
        # Override Lidar specs to match YDLIDAR T-mini Plus
        self.lidar_fov = 2 * np.pi 
        self.num_lidar_rays = 64 
        self.lidar_max_range = 12.0
        
        # Override Obstacle count for "lots of them"
        self.num_obstacles = 20
        self.obstacle_radius = 0.05 # Increased from 0.05 for better visibility
        self.obstacle_range = 7.0 # Spread them wider
        
        # Re-initialize observation space for the new number of rays
        # 64 Lidar + 6 State info = 70
        self.observation_space = spaces.Box(-12.0, 12.0, shape=(self.num_lidar_rays + 6,), dtype=np.float64)
        
        # Re-initialize state variables that might depend on lidar rays
        self.last_action = np.zeros(2)
        self.prev_action = np.zeros(2)

    def _get_obs(self):
        lidar_rays = self._raycast()
        dist = np.linalg.norm(self.goal - self.pose[:2])
        heading = math.atan2(self.goal[1] - self.pose[1], self.goal[0] - self.pose[0])
        
        dev_x = self.target_pos[0] - self.pose[0]
        dev_y = self.target_pos[1] - self.pose[1]
        
        heading_diff = heading - self.pose[2]
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
        
        state_info = np.array([
            self.last_action[0],
            self.last_action[1],
            dist / 15.0, # Adjusted normalization for larger world
            heading_diff / np.pi,
            dev_x / self.world_size,
            dev_y / self.world_size
        ], dtype=np.float64)
        
        return np.concatenate([lidar_rays, state_info])

    def _spawn_obstacle(self, shape="circle"):
        # Force only circular obstacles with small radius
        attempts = 0
        while attempts < 100:
            x = np.random.uniform(-self.obstacle_range, self.obstacle_range)
            y = np.random.uniform(-self.obstacle_range, self.obstacle_range)
            
            r = self.obstacle_radius
            # Check vs Robot (Safe zone around spawn)
            collision = np.linalg.norm([x - self.pose[0], y - self.pose[1]]) < (r + self.robot_radius + 0.8)
            # Check vs Goal
            collision |= np.linalg.norm([x - self.goal[0], y - self.goal[1]]) < (r + 0.8)
            # Check vs Existing obstacles
            for existing in self.obstacles:
                if np.linalg.norm([x - existing["pos"][0], y - existing["pos"][1]]) < 0.3:
                    collision = True; break
            
            if not collision:
                self.obstacles.append({"type": "circle", "pos": np.array([x, y]), "r": r})
                break
            attempts += 1

    def reset(self, seed=None, options=None):
        # We need to ensure we call our overridden _spawn_obstacle
        super().reset(seed=seed, options=options)
        # Note: super().reset calls self._spawn_obstacle in a loop.
        # Since self is a Holonomic360LidarEnv instance, it will call OUR version.
        return self._get_obs(), {}

    def _raycast(self):
        """
        Vectorized raycasting for 667 rays against 80 circular obstacles.
        Significant performance boost over the nested Python loop.
        """
        # 1. Directions and robot pose
        start_angle = self.pose[2] - self.lidar_fov / 2
        angles = np.linspace(start_angle, start_angle + self.lidar_fov, self.num_lidar_rays, endpoint=False)
        ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1) # (N_rays, 2)
        
        # 2. Obstacle data (Assume all are circles for speed)
        if not self.obstacles:
            return np.ones(self.num_lidar_rays)
            
        obs_pos = np.stack([obs["pos"] for obs in self.obstacles]) # (N_obs, 2)
        obs_r = np.array([obs["r"] for obs in self.obstacles]) # (N_obs,)
        
        # 3. Vectorized Intersection
        # Vector from robot to each obstacle center: (N_obs, 2)
        OC = self.pose[:2] - obs_pos 
        
        # Quadratic eq: at^2 + bt + c = 0
        # a = 1 (unit ray directions)
        # b = 2 * (D . OC)
        # (N_rays, N_obs)
        b = 2 * np.matmul(ray_dirs, OC.T)
        
        # c = |OC|^2 - r^2: (N_obs,)
        c = np.sum(OC**2, axis=1) - obs_r**2
        
        # Discriminant: b^2 - 4ac (a=1)
        # (N_rays, N_obs)
        delta = b**2 - 4 * c
        
        # Find intersections where delta >= 0
        mask = delta >= 0
        t = np.full_like(delta, np.inf)
        
        # Only compute sqrt for valid hits
        if np.any(mask):
            t[mask] = (-b[mask] - np.sqrt(delta[mask])) / 2
        
        # Filter: t > 0 and t < lidar_max_range
        t[t <= 0] = np.inf
        t[t > self.lidar_max_range] = np.inf
        
        # Closest hit for each ray
        min_dists = np.min(t, axis=1)
        
        # Return normalized results [0, 1]
        return np.clip(min_dists / self.lidar_max_range, 0.0, 1.0)

    def step(self, action):
        """
        Overridden step method to provide custom reward shaping for 360 environment:
        - Goal reward: 100.0 (from 20.0)
        - Progress weight: 50.0 (from 20.0)
        - Collision penalty: -20.0 (from -10.0)
        """
        self.steps += 1
        self.last_action = action
        
        dx = action[0] * 0.1
        dy = action[1] * 0.1
        
        new_x = self.pose[0] + dx
        new_y = self.pose[1] + dy
        new_x = np.clip(new_x, -8.0, 8.0)
        new_y = np.clip(new_y, -8.0, 8.0)
        
        collision = self._check_collision(new_x, new_y)
        
        if not collision:
             self.pose[0] = new_x
             self.pose[1] = new_y
             self.target_pos = np.array([new_x, new_y])
            
        dist = np.linalg.norm(self.goal - self.pose[:2])
        reward = 0.0
        terminated = False
        truncated = False
        
        if self.steps >= self.max_steps:
            truncated = True
            
        goal_reached = dist < 1.0 # Increased threshold from 0.5 to 1.0 for easier initial learning
        
        if goal_reached:
            reward = 100.0 # Increased from 20.0
            terminated = True
        elif collision:
            reward = -20.0 # Increased from -10.0
            terminated = True
        else:
            # Reward: Progress + Time + Safety + Smoothness
            # Increased progress coefficient from 20.0 to 50.0
            reward = 50.0 * (self.prev_distance - dist) - 0.1
            
            # Action Smoothness Penalty
            smoothness_penalty = -0.1 * np.linalg.norm(action - self.prev_action)
            reward += smoothness_penalty
            
            obs = self._get_obs()
            closest_laser = np.min(obs[:self.num_lidar_rays])
            if closest_laser < 0.3:
                 reward += -1.0 * (0.3 - closest_laser)**2 # Increased penalty weight
            
            # REWARD SCALING: Scale by 0.1 to keep values stable for TD3
            reward *= 0.1
            
        self.prev_distance = dist
        self.prev_action = action.copy()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, {"reached": goal_reached}

    def _render_frame(self):
        # We reuse the base render frame but it might be dense with 667 rays.
        # Let's ensure the rays are drawn correctly for 360.
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        def to_screen(x, y):
            sx = (self.window_size / 2) - (y * self.scale)
            sy = (self.window_size / 2) - (x * self.scale)
            return int(sx), int(sy)
        
        def scale_len(l): return int(l * self.scale)
        
        # Draw Goal
        gx, gy = to_screen(*self.goal)
        pygame.draw.circle(canvas, (0, 255, 0), (gx, gy), scale_len(0.5))
        
        # Draw Obstacles
        for obs in self.obstacles:
            osx, osy = to_screen(*obs["pos"])
            pygame.draw.circle(canvas, (100, 100, 100), (osx, osy), scale_len(obs["r"]))
        
        # Draw Lidar Rays
        rx, ry = to_screen(*self.pose[:2])
        obs_vals = self._get_obs()
        lasers = obs_vals[:self.num_lidar_rays]
        
        start_angle = self.pose[2] - self.lidar_fov / 2
        angle_step = self.lidar_fov / self.num_lidar_rays
        
        # Adjust stride based on number of rays to keep rendering fast but accurate
        stride = max(1, self.num_lidar_rays // 128) if self.render_mode == "human" else 1
        
        for i in range(0, self.num_lidar_rays, stride):
            r_norm = lasers[i]
            r_dist = r_norm * self.lidar_max_range
            angle = start_angle + i * angle_step
            end_x = self.pose[0] + r_dist * math.cos(angle)
            end_y = self.pose[1] + r_dist * math.sin(angle)
            ex, ey = to_screen(end_x, end_y)
            pygame.draw.line(canvas, (255, 100, 100), (rx, ry), (ex, ey), 1)
        
        # Draw Robot
        offsets = [(math.cos(a), math.sin(a)) for a in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]]
        for ox, oy in offsets:
            rotor_x = self.pose[0] + ox * self.robot_radius
            rotor_y = self.pose[1] + oy * self.robot_radius
            rex, rey = to_screen(rotor_x, rotor_y)
            pygame.draw.line(canvas, (0,0,0), (rx, ry), (rex, rey), 3)
            pygame.draw.circle(canvas, (0,0,255), (rex, rey), scale_len(0.08))
        pygame.draw.circle(canvas, (50, 50, 50), (rx, ry), scale_len(0.1))
        
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
