import argparse
import gymnasium as gym
import numpy as np
import itertools
import torch
from datetime import datetime
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from train_env_disp_mem import DroneGazeboEnv


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="GazeboIrisEnv-v0",
                    help='Mujoco Gym environment (default: GazeboIrisEnv-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=5000, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=500000, metavar='N',
                    help='maximum number of steps (default: 8000)')
parser.add_argument('--hidden_size', type=int, default=600, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 1000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=True)

args = parser.parse_args()


gym.register(
    id='GazeboIrisEnv-v0',
    entry_point='train_env_disp_mem:DroneGazeboEnv', 
)

state_dim = 4
laser_dim = 30
action_dim = 2
env_name = "GazeboIrisEnv-v0"
policy_file = "laser_2d_new_200_400_Gaussian_21_10_2024_00_34"

env = gym.make(env_name)
# Agent
agent = SAC(laser_dim, state_dim, env.action_space, args)

agent.load_checkpoint(f"checkpoints/sac_checkpoint_{env_name}_{policy_file}")


for _  in range(200):
    state, info = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state["laser"], state["goal"], evaluate=True)

        next_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward


        state = next_state
            