import gymnasium as gym
import numpy as np

class AgnosticPIDRewardWrapper(gym.Wrapper):
    """
    A universal Gymnasium wrapper that applies PID-based reward shaping 
    based on 'observation activity' (displacement in state-space).
    
    This detects 'stagnation' or 'stalling' without needing environment-specific goals.
    """
    def __init__(self, env, target_activity=0.02, kp=0.5, ki=0.001, kd=0.01):
        super().__init__(env)
        self.target_activity = target_activity
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_obs = None
        
        # Stagnation tuning parameters
        self.stall_threshold = 0.001
        self.tuning_growth_rate = 0.0001 
        self.tuning_decay_rate = 0.98

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.integral = 0.0
        # Initial error is the gap between starting activity (0) and target
        self.prev_error = self.target_activity 
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Calculate Activity (Observation Displacement)
        # We handle single observation or dictionary/tuple if necessary, 
        # but standard Box space is assumed here.
        if isinstance(obs, np.ndarray) and isinstance(self.prev_obs, np.ndarray):
            displacement = np.linalg.norm(obs - self.prev_obs)
        else:
            displacement = 0.0
            
        # 2. Calculate Error (Target - Actual)
        error = self.target_activity - displacement
        
        # 3. Adaptive Tuning of Ki based on 'Stall'
        # If the agent is physically stuck or not exploring enough
        if displacement < self.stall_threshold:
            self.ki += self.tuning_growth_rate
            self.ki = min(self.ki, 0.05) # Cap to prevent reward explosion
        elif displacement > self.target_activity:
            # Gradually reduce Ki pressure if we are moving well
            self.ki = max(0.001, self.ki * self.tuning_decay_rate)

        # 4. PID Reward Components
        self.integral += error
        self.integral = np.clip(self.integral, -5.0, 5.0) # Anti-windup
        derivative = error - self.prev_error
        
        # PID 'Pressure' signal
        # High pressure means high stagnation
        pid_pressure = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # 5. Apply Pressure to Reward
        # We subtract pressure because high pressure (stagnation) should lower the reward
        reward -= pid_pressure
        
        # Update memory
        self.prev_obs = obs
        self.prev_error = error
        
        return obs, reward, terminated, truncated, info
