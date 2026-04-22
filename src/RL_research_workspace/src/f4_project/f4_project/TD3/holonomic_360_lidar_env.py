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
        self.num_lidar_rays = 128  # 256 rays -> 1.4° resolution, reliably detects 6cm poles
        self.lidar_max_range = 12.0
        
        # Override Obstacle count for "lots of them"
        self.num_obstacles = 3
        self.obstacle_radius = 0.1
        self.obstacle_range = 3.0 # Fits within 8x8 world
        
        # Override world size to 8x8 meters (-4 to 4)
        self.world_size = 8.0
        self.scale = self.window_size / self.world_size
        
        # Re-initialize observation space for the new number of rays
        # 256 Lidar + 6 State info = 262
        self.observation_space = spaces.Box(-12.0, 12.0, shape=(self.num_lidar_rays + 6,), dtype=np.float64)
        
        # Re-initialize state variables that might depend on lidar rays
        self.last_action = np.zeros(2)
        self.prev_action = np.zeros(2)
        
        # New Simulation Settings: Input Delay and Increased Obstacle Safety
        self.action_delay = 0 # Disabled input delay for baseline testing
        self.action_queue = []
        self.rings = []

    def _get_obs(self):
        lidar_rays = self._raycast()
        dist = np.linalg.norm(self.goal - self.pose[:2])
        heading = math.atan2(self.goal[1] - self.pose[1], self.goal[0] - self.pose[0])
        
        # Fix: Provide Cartesian offset to the actual goal, not the immediate target
        dev_x = self.goal[0] - self.pose[0]
        dev_y = self.goal[1] - self.pose[1]
        
        heading_diff = heading - self.pose[2]
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
        
        state_info = np.array([
            self.last_action[0],
            self.last_action[1],
            dist / 11.0, # Normalized for 8x8 world (max ~11m diagonal)
            heading_diff / np.pi,
            dev_x / self.world_size,
            dev_y / self.world_size
        ], dtype=np.float64)
        
        return np.concatenate([lidar_rays, state_info])

    def _spawn_obstacle(self, shape="circle"):
        # Force only drone rings (70cm width gate)
        attempts = 0
        while attempts < 100:
            x = np.random.uniform(-self.obstacle_range, self.obstacle_range)
            y = np.random.uniform(-self.obstacle_range, self.obstacle_range)
            yaw = np.random.uniform(0, 2 * np.pi)
            
            w = 0.70  # 70cm width
            r = 0.03  # 6cm diameter pole -> 0.03m radius
            
            pole1 = np.array([x + (w/2) * math.cos(yaw), y + (w/2) * math.sin(yaw)])
            pole2 = np.array([x - (w/2) * math.cos(yaw), y - (w/2) * math.sin(yaw)])
            
            collision = False
            # Check vs Robot (Safe zone around spawn)
            if np.linalg.norm([x - self.pose[0], y - self.pose[1]]) < (w/2 + self.robot_radius + 0.8):
                collision = True
            # Check vs Goal
            if np.linalg.norm([x - self.goal[0], y - self.goal[1]]) < (w/2 + 0.8):
                collision = True
            # Check vs Existing obstacles
            for existing in self.obstacles:
                if np.linalg.norm([pole1[0] - existing["pos"][0], pole1[1] - existing["pos"][1]]) < 0.3:
                    collision = True; break
                if np.linalg.norm([pole2[0] - existing["pos"][0], pole2[1] - existing["pos"][1]]) < 0.3:
                    collision = True; break
            
            if not collision:
                # Add two distinct poles to act as the rigid collision points
                self.obstacles.append({"type": "circle", "pos": pole1, "r": r})
                self.obstacles.append({"type": "circle", "pos": pole2, "r": r})
                if not hasattr(self, 'rings'): self.rings = []
                self.rings.append({"pos": np.array([x, y]), "yaw": yaw, "width": w})
                break
            attempts += 1

    def reset(self, seed=None, options=None):
        # Clear rings before the super.reset populates them
        self.rings = []
        # We need to ensure we call our overridden _spawn_obstacle
        super().reset(seed=seed, options=options)
        
        # Clear and initialize action queue with zeros for the delay duration
        self.action_queue = [np.zeros(2) for _ in range(self.action_delay + 1)]
        
        # Note: super().reset calls self._spawn_obstacle in a loop.
        # Since self is a Holonomic360LidarEnv instance, it will call OUR version.
        return self._get_obs(), {}

    def _spawn_goal(self):
        # Goal in the positive-X half of the 8x8 world
        self.goal = np.array([np.random.uniform(2.0, 3.5), np.random.uniform(-1.5, 1.5)])

    def _spawn_robot(self):
        self.pose = np.zeros(3)
        # Robot in the negative-X half of the 8x8 world
        self.pose[0] = np.random.uniform(-3.5, -2.0)
        self.pose[1] = np.random.uniform(-1.5, 1.5)
        self.pose[2] = 0.0  # Facing Positive X (Forward)
        self.target_pos = self.pose[:2].copy()

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
        
        # Calculate clearance from robot edge instead of center
        clearances = min_dists - self.robot_radius
        
        # Return normalized results [0, 1]
        return np.clip(clearances / self.lidar_max_range, 0.0, 1.0)

    def step(self, action):
        """
        Revised step method:
        - Implements 2-step input delay for realism.
        - Provides stricter penalty for obstacle proximity (danger threshold 0.2).
        - Rewards being further away from obstacles for safety.
        """
        self.steps += 1
        self.last_action = action # Action to be used in observation (intended action)
        
        # Restore symmetric movement so it can actually strafe/arc around obstacles
        dx = action[0] * 0.05
        dy = action[1] * 0.05
        
        new_x = self.pose[0] + dx
        new_y = self.pose[1] + dy
        new_x = np.clip(new_x, -4.0, 4.0)
        new_y = np.clip(new_y, -4.0, 4.0)
        
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
            
        goal_reached = dist < 1.0
        
        if goal_reached:
            reward = 300.0
            terminated = True
        elif collision:
            reward = -50.0
            terminated = True
        else:
            # === DOMINANT: Progress toward goal ===
            # ~0.15/step at max speed. This is the primary learning signal.
            reward = 3.0 * (self.prev_distance - dist)
            
            # === MINOR: Forward thrust preference ===
            reward += 0.1 * action[0]
            
            # === MINOR: Step penalty to discourage idling ===
            reward -= 0.01
            
            # === MINOR: Obstacle proximity penalty ===
            # Activation at 0.8m (tighter), max penalty -0.1 (well below progress signal)
            min_clearance = np.min(self._raycast()) * self.lidar_max_range
            if min_clearance < 0.8:
                reward -= 0.1 * ((0.8 - min_clearance) / 0.8) ** 2            
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
        
        # Draw Rings
        for idx, ring in enumerate(getattr(self, 'rings', [])):
            rx, ry = ring["pos"]
            w = ring["width"]
            yaw = ring["yaw"]
            
            p1x = rx + (w/2) * math.cos(yaw)
            p1y = ry + (w/2) * math.sin(yaw)
            p2x = rx - (w/2) * math.cos(yaw)
            p2y = ry - (w/2) * math.sin(yaw)
            
            sx1, sy1 = to_screen(p1x, p1y)
            sx2, sy2 = to_screen(p2x, p2y)
            
            # Alternate ring colors just like the picture!
            color = (255, 50, 50) if idx % 2 == 0 else (150, 255, 50)
            
            # Draw the ring's top bar connecting the poles
            pygame.draw.line(canvas, color, (sx1, sy1), (sx2, sy2), 4)
            # Draw the base poles (radius 0.03)
            pygame.draw.circle(canvas, (200, 200, 50), (sx1, sy1), scale_len(0.03))
            pygame.draw.circle(canvas, (200, 200, 50), (sx2, sy2), scale_len(0.03))
        
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
            # Since r_norm is clearance from edge, add back robot_radius for rendering from center
            r_dist = (r_norm * self.lidar_max_range) + self.robot_radius if r_norm < 1.0 else self.lidar_max_range
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
