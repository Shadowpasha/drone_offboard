import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from environments.holonomic_lidar_env import HolonomicLidarEnv

class DifferentialLidarEnv(HolonomicLidarEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        
        # Action Space: [v, w] 
        # v: linear velocity [-1, 1]
        # w: angular velocity [-1, 1]
        self.action_space = spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), dtype=np.float64)
        
        # Differential Drive Constants
        self.max_linear_vel = 1.0
        self.max_angular_vel = 2.0
        self.dt = 0.1
        
        # Challenge Config: Denser and more spread out
        self.num_obstacles = 14
        self.obstacle_range = 6.5 # Spreading them out across the world

    def step(self, action):
        self.steps += 1
        self.last_action = action
        
        v = action[0] * self.max_linear_vel
        w = action[1] * self.max_angular_vel
        
        # 1. Update Heading (yaw)
        new_yaw = self.pose[2] + w * self.dt
        # Keep yaw in [-pi, pi]
        new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Update Position
        dx = v * math.cos(new_yaw) * self.dt
        dy = v * math.sin(new_yaw) * self.dt
        
        new_x = self.pose[0] + dx
        new_y = self.pose[1] + dy
        new_x = np.clip(new_x, -8.0, 8.0)
        new_y = np.clip(new_y, -8.0, 8.0)
        
        collision = self._check_collision(new_x, new_y)
        
        if not collision:
            self.pose[0] = new_x
            self.pose[1] = new_y
            self.pose[2] = new_yaw
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
            # Reward: Progress + Time + Smoothness
            reward = 20.0 * (self.prev_distance - dist) - 0.1
            
            # Action Smoothness (Penalty for high angular velocity and abrupt changes)
            smoothness_penalty = -0.05 * abs(w) - 0.1 * np.linalg.norm(action - self.prev_action)
            reward += smoothness_penalty
            
            obs = self._get_obs()
            closest_laser = np.min(obs[:64])
            if closest_laser < 0.3:
                 reward += -0.5 * (0.3 - closest_laser)**2
            
            # REWARD SCALING PATCH: Divide all step rewards by 10 for stability
            reward *= 0.1
            
        self.prev_distance = dist
        self.prev_action = action.copy()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, {"reached": goal_reached}

    def _render_frame(self):
        # Override to show a tank/differential robot
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
            if obs["type"] == "circle":
                osx, osy = to_screen(*obs["pos"])
                pygame.draw.circle(canvas, (100, 100, 100), (osx, osy), scale_len(obs["r"]))
            else:
                w, h = obs["size"] # w is size_x, h is size_y
                # Top-Left in Screen (Min SX, Min SY) corresponds to (Max World Y, Max World X)
                # Max World X = pos[0] + w/2
                # Max World Y = pos[1] + h/2
                
                # We need to pass the "Top Left" WORLD coordinate that results in the Top-Left SCREEN coordinate.
                # to_screen(x, y) -> sx = C - y*s, sy = C - x*s
                
                max_x = obs["pos"][0] + w/2
                max_y = obs["pos"][1] + h/2
                
                sx, sy = to_screen(max_x, max_y)
                
                # Pygame rect width is Screen X span (World Y span -> h)
                # Pygame rect height is Screen Y span (World X span -> w)
                pygame.draw.rect(canvas, (100, 100, 100), (sx, sy, scale_len(h), scale_len(w)))
        
        # Draw Lidar
        rx, ry = to_screen(*self.pose[:2])
        obs_vals = self._get_obs()
        lasers = obs_vals[:64]
        start_angle = self.pose[2] - self.lidar_fov / 2
        angle_step = self.lidar_fov / 64
        for i, r_norm in enumerate(lasers):
            r_dist = r_norm * self.lidar_max_range
            angle = start_angle + i * angle_step
            end_x = self.pose[0] + r_dist * math.cos(angle)
            end_y = self.pose[1] + r_dist * math.sin(angle)
            ex, ey = to_screen(end_x, end_y)
            pygame.draw.line(canvas, (255, 100, 100), (rx, ry), (ex, ey), 1)

        # Draw Differential Robot (Rectangle with arrow/heading)
        # Rect dimensions
        rw, rl = 0.3, 0.4
        yaw = self.pose[2]
        
        # Corners of the robot body
        corners = [
            (rl/2, rw/2), (rl/2, -rw/2), (-rl/2, -rw/2), (-rl/2, rw/2)
        ]
        
        rotated_corners = []
        for cx, cy in corners:
            rx_c = cx * math.cos(yaw) - cy * math.sin(yaw)
            ry_c = cx * math.sin(yaw) + cy * math.cos(yaw)
            rotated_corners.append(to_screen(self.pose[0] + rx_c, self.pose[1] + ry_c))
            
        pygame.draw.polygon(canvas, (50, 50, 200), rotated_corners)
        
        # Draw wheels
        wheel_w, wheel_l = 0.08, 0.15
        for side in [1, -1]: # Right and Left
            wx, wy = 0, side * (rw/2 + wheel_w/2)
            # Center of wheel in world coords
            wwx = self.pose[0] + wx * math.cos(yaw) - wy * math.sin(yaw)
            wwy = self.pose[1] + wx * math.sin(yaw) + wy * math.cos(yaw)
            
            # Corner of wheels
            w_corners = [
                (wwx + (wheel_l/2)*math.cos(yaw) - (wheel_w/2)*math.sin(yaw), wwy + (wheel_l/2)*math.sin(yaw) + (wheel_w/2)*math.cos(yaw)),
                (wwx + (wheel_l/2)*math.cos(yaw) + (wheel_w/2)*math.sin(yaw), wwy + (wheel_l/2)*math.sin(yaw) - (wheel_w/2)*math.cos(yaw)),
                (wwx - (wheel_l/2)*math.cos(yaw) + (wheel_w/2)*math.sin(yaw), wwy - (wheel_l/2)*math.sin(yaw) - (wheel_w/2)*math.cos(yaw)),
                (wwx - (wheel_l/2)*math.cos(yaw) - (wheel_w/2)*math.sin(yaw), wwy - (wheel_l/2)*math.sin(yaw) + (wheel_w/2)*math.cos(yaw)),
            ]
            scr_w_corners = [to_screen(wc[0], wc[1]) for wc in w_corners]
            pygame.draw.polygon(canvas, (0, 0, 0), scr_w_corners)

        # Draw Heading Arrow
        head_x = self.pose[0] + 0.3 * math.cos(yaw)
        head_y = self.pose[1] + 0.3 * math.sin(yaw)
        hx, hy = to_screen(head_x, head_y)
        pygame.draw.line(canvas, (0, 255, 0), (rx, ry), (hx, hy), 3)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
