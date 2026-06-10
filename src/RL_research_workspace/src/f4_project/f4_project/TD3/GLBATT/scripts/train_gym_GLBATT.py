import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gymnasium as gym
import argparse
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from glbatt import utils_transformer as utils
from glbatt.architectures import GLBATT_Summary as GLBATT_Lib

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3", type=str)
    parser.add_argument("--seed", default=1236, type=int)
    parser.add_argument("--start_timesteps", default=10000, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--max_timesteps", default=250000, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--seq_len", default=100, type=int)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"GLBATT_Gym_{args.env}_{args.seed}_{timestamp}"
    print(f"---------------------------------------")
    print(f"Policy: GLBATT_Gym, Env: {args.env}, Seed: {args.seed}")
    print(f"---------------------------------------")

    if not os.path.exists("../results"): os.makedirs("../results")
    if args.save_model and not os.path.exists("../models"): os.makedirs("../models")

    # Use gymnasium
    env = gym.make(args.env)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Dynamic target mapping based on environment
    if "BipedalWalker" in args.env:
        # Predict: Hull Angle (0), X_vel (2), Y_vel (3)
        belief_state_indices = [0, 2, 3]
    elif "LunarLander" in args.env:
        # Predict: Y Pos (1), X Vel (2), Y Vel (3), Angle (4)
        belief_state_indices = [1, 2, 3, 4]
    else:
        # Fallback to predicting the entire state vector
        belief_state_indices = list(range(state_dim))

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "belief_state_indices": belief_state_indices
    }

    policy = GLBATT_Lib.GLBATT(**kwargs)
    if args.load_model != "": policy.load(f"../models/{args.load_model}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, seq_len=args.seq_len)
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
        
        next_state_seq = np.roll(state_seq, -1, axis=0)
        next_state_seq[-1] = next_state
        
        # In Gymnasium, done implies terminal, truncated implies time limit. Both mean episode reset.
        done_bool = float(done) if not truncated else 0
        replay_buffer.add(state, action, next_state, reward, done_bool, state_seq, next_state_seq)

        state = next_state
        state_seq = next_state_seq
        episode_reward += reward

        if t >= args.start_timesteps:
            q_value, critic_loss, z_var, attn_dist, intrinsic_reward, safety_loss = policy.train(replay_buffer, args.batch_size)
            if (t + 1) % 100 == 0:
                writer.add_scalar("Loss/Critic", critic_loss, t + 1)
                writer.add_scalar("Loss/Auxiliary_State_Prediction", safety_loss, t + 1)
                writer.add_scalar("Value/Q", q_value, t + 1)
                
                global_attn = attn_dist[:10].sum().item()
                local_attn = attn_dist[10:].sum().item()
                writer.add_scalar('Attention/GlobalUsage', global_attn, t + 1)
                writer.add_scalar('Attention/LocalUsage', local_attn, t + 1)

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
