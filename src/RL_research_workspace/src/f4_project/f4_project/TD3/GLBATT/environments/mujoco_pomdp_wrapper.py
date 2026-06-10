import gymnasium as gym
import numpy as np

class JointMaskWrapper(gym.ObservationWrapper):
    """
    A Gymnasium wrapper to convert standard MuJoCo environments into POMDPs
    by masking out the entire state (position and velocity) of a specific joint.
    This simulates a realistic sensor failure on a distal linkage.
    """
    def __init__(self, env):
        super().__init__(env)
        
        env_name = env.unwrapped.spec.id
        
        # Hopper-v4 observation vector length is 11.
        # [z, angle_torso, angle_thigh, angle_leg, angle_foot, vel_x, vel_z, vel_torso, vel_thigh, vel_leg, vel_foot]
        # To mask the foot, we remove index 4 (angle) and 10 (velocity).
        if "Hopper" in env_name:
            self.mask_indices = [4, 10]
        else:
            # Fallback for others, mask the last pos and last vel
            # Assume roughly half pos, half vel
            split = self.observation_space.shape[0] // 2
            self.mask_indices = [split - 1, self.observation_space.shape[0] - 1]
            
        self.keep_indices = [i for i in range(self.observation_space.shape[0]) if i not in self.mask_indices]
        
        # Update observation space
        high = self.observation_space.high[self.keep_indices]
        low = self.observation_space.low[self.keep_indices]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
    def observation(self, obs):
        return obs[self.keep_indices]
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Capture unshaped reward for benchmarking
        info['original_reward'] = reward
        
        # Extract the masked joint states (angle, velocity) to serve as ground truth targets
        masked_joint_state = obs[self.mask_indices]
        info['masked_joint_state'] = masked_joint_state
        
        return self.observation(obs), reward, terminated, truncated, info
