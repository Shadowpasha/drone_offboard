import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gymnasium as gym
import argparse
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from glbatt.architectures import CrossQ_GLBATT
from environments.mujoco_pomdp_wrapper import JointMaskWrapper

class POMDPReplayBuffer(object):
    def __init__(self, state_dim, action_dim, belief_dim, seq_len=50, max_size=int(2e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.seq_len = seq_len

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        self.true_belief = np.zeros((max_size, belief_dim), dtype=np.float32) # Store masked velocities
        
        self.state_seq = np.zeros((max_size, seq_len, state_dim), dtype=np.float32)
        self.next_state_seq = np.zeros((max_size, seq_len, state_dim), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, state_seq, next_state_seq, true_belief):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.true_belief[self.ptr] = true_belief
        
        self.state_seq[self.ptr] = state_seq
        self.next_state_seq[self.ptr] = next_state_seq

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.state_seq[ind]).to(self.device),
            torch.FloatTensor(self.next_state_seq[ind]).to(self.device),
            torch.FloatTensor(self.true_belief[ind]).to(self.device)
        )

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v4", type=str)
    parser.add_argument("--seed", default=1236, type=int)
    parser.add_argument("--start_timesteps", default=10000, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--max_timesteps", default=700000, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=1.0, type=float) 
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=3, type=int) 
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--seq_len", default=30, type=int)
    parser.add_argument("--td3_mode", action="store_true", default=False)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"CrossQ_GLBATT_POMDP_{args.env}_{args.seed}_{timestamp}"
    print(f"---------------------------------------")
    print(f"Policy: CrossQ_GLBATT POMDP, Env: {args.env}, Seed: {args.seed}")
    print(f"---------------------------------------")

    if not os.path.exists("../results"): os.makedirs("../results")
    if args.save_model and not os.path.exists("../models"): os.makedirs("../models")

    # Wrap the standard environment to mask the target joint
    base_env = gym.make(args.env)
    env = JointMaskWrapper(base_env)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # In POMDP mode, our belief indices are not sliced from the state. 
    # Node 0 is dedicated to predicting the "Hazard Horizon" (e.g. falling)
    # The remaining nodes reconstruct the masked distal joint.
    # Hopper-v4 has the foot joint masked (1 angle, 1 velocity = 2 dims)
    masked_dim = 2 if "Hopper" in args.env else 2
    belief_dim = 1 + masked_dim

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "belief_state_indices": list(range(belief_dim)),  # Only used for architectural init
        "td3_mode": args.td3_mode
    }

    policy = CrossQ_GLBATT.CrossQ_GLBATT(**kwargs)

    replay_buffer = POMDPReplayBuffer(state_dim, action_dim, masked_dim, seq_len=args.seq_len)
    writer = SummaryWriter(log_dir=f"../runs/{file_name}")

    state, _ = env.reset(seed=args.seed)
    state_seq = np.zeros((args.seq_len, state_dim))
    for i in range(args.seq_len): state_seq[i] = state
        
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(state_seq) + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

        next_state, reward, done, truncated, info = env.step(action)
        true_belief = info.get('masked_joint_state', np.zeros(masked_dim))
        
        next_state_seq = np.roll(state_seq, -1, axis=0)
        next_state_seq[-1] = next_state
        
        done_bool = float(done) if not truncated else 0
        replay_buffer.add(state, action, next_state, reward, done_bool, state_seq, next_state_seq, true_belief)

        state = next_state
        state_seq = next_state_seq
        episode_reward += reward

        if t >= args.start_timesteps:
            q_value, critic_loss, z_var, attn_dist, intrinsic_reward, safety_loss = policy.train(replay_buffer, args.batch_size)
            if (t + 1) % 100 == 0:
                writer.add_scalar("Loss/Critic", critic_loss, t + 1)
                writer.add_scalar("Loss/Safety_Grounding", safety_loss, t + 1)
                writer.add_scalar("Value/Q", q_value, t + 1)

        if done or truncated:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Reward: {episode_reward:.3f}")
            writer.add_scalar("Reward/Episode", episode_reward, episode_num + 1)

            state, _ = env.reset()
            state_seq = np.zeros((args.seq_len, state_dim))
            for i in range(args.seq_len): state_seq[i] = state
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % args.eval_freq == 0 and args.save_model:
            policy.save(f"../models/{file_name}")

    writer.close()

if __name__ == "__main__":
    train()
