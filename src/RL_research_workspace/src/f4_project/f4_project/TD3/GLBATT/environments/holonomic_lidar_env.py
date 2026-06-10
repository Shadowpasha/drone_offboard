import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random

class HolonomicLidarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        # Constants matching original env or approx
        self.goal_range = 6.0
        self.obstacle_range = 3.5 # Slightly reduced to avoid too much clutter in 2D
        self.num_obstacles = 6
        self.start_range = 3.0
        
        # Robot settings
        self.robot_radius = 0.5
        self.lidar_max_range = 12.0
        self.num_lidar_rays = 64
        self.lidar_fov = 120.0 * (np.pi / 180.0) # 120 degrees in radians
        
        # Pygame settings
        self.window_size = 800
        self.world_size = 16.0 # -8 to 8 meters
        self.scale = self.window_size / self.world_size
        self.window = None
        self.clock = None
        
        # Observation Space: 64 Lidar + 6 Goal/State info
        self.observation_space = spaces.Box(-8.0, 8.0, shape=(70,), dtype=np.float64)
        
        # Action Space: [vx, vy]
        self.action_space = spaces.Box(np.array([-1,-1]), np.array([1,1]), dtype=np.float64)
        
        # State variables
        self.pose = np.zeros(3) # x, y, yaw
        self.goal = np.zeros(2)
        # Obstacles: list of dicts {"type": "circle"/"rect", "pos": [x,y], "r": r, "size": [w,h]}
        self.obstacles = [] 
        self.prev_distance = 0.0
        self.last_action = np.zeros(2)
        self.steps = 0
        self.max_steps = 500
        self.target_pos = np.zeros(2)
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
            dist / 10.0,
            heading_diff / np.pi,
            dev_x / self.world_size, # Normalization Patch: keep spatial dims in [-1, 1]
            dev_y / self.world_size
        ], dtype=np.float64)
        
        return np.concatenate([lidar_rays, state_info])

    def _raycast(self):
        rays = np.ones(self.num_lidar_rays) * 1.0
        # FOV is centered around the current heading (pose[2])
        # Wedge from -fov/2 to +fov/2
        start_angle = self.pose[2] - self.lidar_fov / 2
        angles = np.linspace(start_angle, start_angle + self.lidar_fov, self.num_lidar_rays, endpoint=False)
        
        for i, angle in enumerate(angles):
            ray_dir = np.array([math.cos(angle), math.sin(angle)])
            min_dist = self.lidar_max_range
            
            for obs in self.obstacles:
                if obs["type"] == "circle":
                    Ox_Cx = self.pose[0] - obs["pos"][0]
                    Oy_Cy = self.pose[1] - obs["pos"][1]
                    b = 2 * (Ox_Cx * ray_dir[0] + Oy_Cy * ray_dir[1])
                    c = (Ox_Cx**2 + Oy_Cy**2) - obs["r"]**2
                    delta = b**2 - 4*c
                    if delta >= 0:
                        t1 = (-b - math.sqrt(delta)) / 2
                        if 0 < t1 < min_dist:
                            min_dist = t1
                
                elif obs["type"] == "rect":
                    # Ray-AABB Intersection (Slab method)
                    half_w, half_h = obs["size"][0] / 2, obs["size"][1] / 2
                    min_x, max_x = obs["pos"][0] - half_w, obs["pos"][0] + half_w
                    min_y, max_y = obs["pos"][1] - half_h, obs["pos"][1] + half_h
                    
                    inv_dx = 1.0 / ray_dir[0] if ray_dir[0] != 0 else float('inf')
                    inv_dy = 1.0 / ray_dir[1] if ray_dir[1] != 0 else float('inf')
                    
                    t1_x = (min_x - self.pose[0]) * inv_dx
                    t2_x = (max_x - self.pose[0]) * inv_dx
                    t1_y = (min_y - self.pose[1]) * inv_dy
                    t2_y = (max_y - self.pose[1]) * inv_dy
                    
                    t_min = max(min(t1_x, t2_x), min(t1_y, t2_y))
                    t_max = min(max(t1_x, t2_x), max(t1_y, t2_y))
                    
                    if t_max >= 0 and t_min <= t_max:
                        if 0 < t_min < min_dist:
                            min_dist = t_min
            
            rays[i] = min_dist / self.lidar_max_range
        return np.clip(rays, 0.0, 1.0)

    def _check_collision(self, x, y):
        for obs in self.obstacles:
            if obs["type"] == "circle":
                if np.linalg.norm([x - obs["pos"][0], y - obs["pos"][1]]) < (self.robot_radius + obs["r"]):
                    return True
            elif obs["type"] == "rect":
                half_w, half_h = obs["size"][0] / 2, obs["size"][1] / 2
                rect_min = obs["pos"] - [half_w, half_h]
                rect_max = obs["pos"] + [half_w, half_h]
                
                # Find closest point on rect to robot center
                closest_x = max(rect_min[0], min(x, rect_max[0]))
                closest_y = max(rect_min[1], min(y, rect_max[1]))
                
                dist = np.linalg.norm([x - closest_x, y - closest_y])
                if dist < self.robot_radius:
                    return True
        return False

    def step(self, action):
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
            
        goal_reached = dist < 0.5
        
        if goal_reached:
            reward = 20.0 # Standardized Peer Reward (Scale 0.1)
            terminated = True
        elif collision:
            reward = -10.0 # Standardized Peer Penalty (Scale 0.1)
            terminated = True
        else:
            # Reward: Progress + Time + Safety + Smoothness
            reward = 20.0 * (self.prev_distance - dist) - 0.1
            
            # Action Smoothness Penalty
            smoothness_penalty = -0.1 * np.linalg.norm(action - self.prev_action)
            reward += smoothness_penalty
            
            obs = self._get_obs()
            closest_laser = np.min(obs[:self.num_lidar_rays])
            if closest_laser < 0.3:
                 reward += -0.5 * (0.3 - closest_laser)**2
            
            # REWARD SCALING PATCH: Divide all step rewards by 10 for stability
            reward *= 0.1
            
        self.prev_distance = dist
        self.prev_action = action.copy()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, {"reached": goal_reached}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.target_pos = np.zeros(2)
        self._spawn_goal()
        self._spawn_robot()
        self.obstacles = []
        for i in range(self.num_obstacles):
            # Mix shapes: even are circles, odd are rects
            self._spawn_obstacle(shape="circle" if i % 2 == 0 else "rect")
            
        self.prev_distance = np.linalg.norm(self.goal - self.pose[:2])
        self.last_action = np.zeros(2)
        self.prev_action = np.zeros(2)
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), {}

    def _spawn_goal(self):
        # Strict Forward-Goal Patch: Goal is always in the far right sector
        self.goal = np.array([np.random.uniform(4.0, 6.0), np.random.uniform(-1.5, 1.5)])

    def _spawn_robot(self):
        self.pose = np.zeros(3)
        # Strict Forward-Goal Patch: Robot starts in the far left sector
        self.pose[0] = np.random.uniform(-6.0, -4.0)
        self.pose[1] = np.random.uniform(-1.5, 1.5)
        self.pose[2] = 0.0 # Facing Positive X (Forward)
        self.target_pos = self.pose[:2].copy()

    def _spawn_obstacle(self, shape="circle"):
        attempts = 0
        while attempts < 100:
            x = np.random.uniform(-self.obstacle_range, self.obstacle_range)
            y = np.random.uniform(-self.obstacle_range, self.obstacle_range)
            
            if shape == "circle":
                r = 0.5
                collision = np.linalg.norm([x - self.pose[0], y - self.pose[1]]) < (r + self.robot_radius + 0.5)
                collision |= np.linalg.norm([x - self.goal[0], y - self.goal[1]]) < (r + 0.5)
                for existing in self.obstacles:
                    if np.linalg.norm([x - existing["pos"][0], y - existing["pos"][1]]) < 1.0:
                        collision = True; break
                if not collision:
                    self.obstacles.append({"type": "circle", "pos": np.array([x, y]), "r": r})
                    break
            else:
                w, h = 0.8, 0.8
                # Check vs Robot
                collision = (abs(x - self.pose[0]) < (w/2 + self.robot_radius + 0.5)) and (abs(y - self.pose[1]) < (h/2 + self.robot_radius + 0.5))
                collision |= (abs(x - self.goal[0]) < (w/2 + 0.5)) and (abs(y - self.goal[1]) < (h/2 + 0.5))
                for existing in self.obstacles:
                    if np.linalg.norm([x - existing["pos"][0], y - existing["pos"][1]]) < 1.2:
                        collision = True; break
                if not collision:
                    self.obstacles.append({"type": "rect", "pos": np.array([x, y]), "size": np.array([w, h])})
                    break
            attempts += 1

    def render(self):
        if self.render_mode == "rgb_array": return self._render_frame()

    def _render_frame(self):
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
        
        gx, gy = to_screen(*self.goal)
        pygame.draw.circle(canvas, (0, 255, 0), (gx, gy), scale_len(0.5))
        
        for obs in self.obstacles:
            if obs["type"] == "circle":
                osx, osy = to_screen(*obs["pos"])
                pygame.draw.circle(canvas, (100, 100, 100), (osx, osy), scale_len(obs["r"]))
            else:
                # Rectangle (centered)
                w, h = obs["size"] # w is size_x, h is size_y
                # In to_screen: sx relates to y, sy relates to x
                # Screen top-left (min sx, min sy) -> (max world y, max world x)
                max_x = obs["pos"][0] + w/2
                max_y = obs["pos"][1] + h/2
                sx, sy = to_screen(max_x, max_y)
                
                # Pygame rect: width is sx-span (y-size), height is sy-span (x-size)
                pygame.draw.rect(canvas, (100, 100, 100), (sx, sy, scale_len(h), scale_len(w)))
        
        rx, ry = to_screen(*self.pose[:2])
        obs_vals = self._get_obs()
        lasers = obs_vals[:64]
        # Must match _raycast logic exactly
        start_angle = self.pose[2] - self.lidar_fov / 2
        angle_step = self.lidar_fov / 64
        for i, r_norm in enumerate(lasers):
            r_dist = r_norm * self.lidar_max_range
            angle = start_angle + i * angle_step
            end_x = self.pose[0] + r_dist * math.cos(angle)
            end_y = self.pose[1] + r_dist * math.sin(angle)
            ex, ey = to_screen(end_x, end_y)
            pygame.draw.line(canvas, (200, 0, 0), (rx, ry), (ex, ey), 1)
        
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

    def hide_window(self):
        if self.window is not None:
             pygame.display.quit()
             self.window = None
             self.clock = None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
