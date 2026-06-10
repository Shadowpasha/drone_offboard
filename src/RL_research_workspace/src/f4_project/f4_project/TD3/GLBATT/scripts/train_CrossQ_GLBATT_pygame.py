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
from glbatt.architectures import CrossQ_GLBATT as CrossQ_GLBATT_Lib
from environments.holonomic_lidar_env import HolonomicLidarEnv
from environments.holonomic_lidar_env_pid import HolonomicLidarEnvPID
from environments.holonomic_lidar_env_moving import HolonomicLidarEnvMoving
from environments.differential_lidar_env import DifferentialLidarEnv
from environments.complex_envs import DifferentialTrapEnv, DifferentialMovingEnv, HolonomicBlindCorridorEnv



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="moving", choices=["normal", "pid", "moving", "diff", "trap", "diff_moving", "corridor"])
    parser.add_argument("--curriculum", action="store_true", default=True, help="Use curriculum learning for trap env")
    parser.add_argument("--seed", default=1236, type=int)
    parser.add_argument("--start_timesteps", default=10000, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--max_timesteps", default=250000, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=1.0, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=3, type=int)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--render", default=0, type=int)
    parser.add_argument("--seq_len", default=50, type=int)
    parser.add_argument("--td3_mode", action="store_true", default=True)
    args = parser.parse_args()

    file_name = f"CrossQ_GLBATT_{args.env}_{args.seed}"
    print(f"---------------------------------------")
    print(f"Policy: CrossQ_GLBATT, Env: {args.env}, Seed: {args.seed}")
    print(f"---------------------------------------")

    if not os.path.exists("../results"): os.makedirs("../results")
    if args.save_model and not os.path.exists("../models"): os.makedirs("../models")

    render_mode = "human" if args.render else None
    if args.env == "pid": env = HolonomicLidarEnvPID(render_mode=render_mode)
    elif args.env == "moving": env = HolonomicLidarEnvMoving(render_mode=render_mode)
    elif args.env == "diff": env = DifferentialLidarEnv(render_mode=render_mode)
    elif args.env == "diff_moving": env = DifferentialMovingEnv(render_mode=render_mode)
    elif args.env == "trap": 
        env = DifferentialTrapEnv(render_mode=render_mode)
        if args.curriculum:
            print("Curriculum Learning Enabled: Starting at Difficulty 0.0")
            env.set_difficulty(0.0)
    elif args.env == "corridor":
        env = HolonomicBlindCorridorEnv(render_mode=render_mode)
    else: env = HolonomicLidarEnv(render_mode=render_mode)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "belief_state_indices": None,
        "td3_mode": args.td3_mode
    }

    policy = CrossQ_GLBATT_Lib.CrossQ_GLBATT(**kwargs)
    
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"../models/{policy_file}")
    # Replay buffer with configurable sequence length
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, seq_len=args.seq_len)
    
    # Append timestamp to log_dir to prevent TensorBoard event files from overlapping
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir=f"../runs/{file_name}_{timestamp}")



    state, _ = env.reset(seed=args.seed)
    state_seq = np.zeros((args.seq_len, state_dim))
    for i in range(args.seq_len): state_seq[i] = state
        
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    window_successes = []

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(state_seq) + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

        next_state, reward, done, truncated, info = env.step(action)
        
        next_state_seq = np.roll(state_seq, -1, axis=0)
        next_state_seq[-1] = next_state
        
        done_bool = float(done) if episode_timesteps < env.max_steps else 0
        replay_buffer.add(state, action, next_state, reward, done_bool, state_seq, next_state_seq)

        state = next_state
        state_seq = next_state_seq
        episode_reward += reward

        if t >= args.start_timesteps:
            q_value, critic_loss, z_var, attn_dist, intrinsic_reward, safety_loss = policy.train(replay_buffer, args.batch_size)
            if (t + 1) % 100 == 0:
                writer.add_scalar("Loss/Critic", critic_loss, t + 1)
                writer.add_scalar("Loss/Safety_Grounding", safety_loss, t + 1)
                writer.add_scalar("Value/Q", q_value, t + 1)
                writer.add_scalar("Stats/Z_Variance", z_var, t + 1)
                writer.add_scalar("Stats/Intrinsic_Reward", intrinsic_reward, t + 1)
                
                # Log Attention Heatmap (Global vs Local)
                # attn_dist is size 30: [0-9] = Global, [10-29] = Local
                global_attn = attn_dist[:10].sum().item()
                local_attn = attn_dist[10:].sum().item()
                writer.add_scalar('Attention/GlobalUsage', global_attn, t + 1)
                writer.add_scalar('Attention/LocalUsage', local_attn, t + 1)
                
                # Diagnostic Weight Logging
                actor_norm = sum(p.norm(2) for p in policy.actor.parameters() if p.grad is not None)
                critic_norm = sum(p.norm(2) for p in policy.critic.parameters() if p.grad is not None)
                writer.add_scalar("Value/Weights_Actor", actor_norm, t + 1)
                writer.add_scalar("Value/Weights_Critic", critic_norm, t + 1)

        if done or truncated:
            reached = info.get("reached", False)
            window_successes.append(1 if reached else 0)
            if len(window_successes) > 100: window_successes.pop(0)
            success_rate = np.mean(window_successes)

            print(f"Total T: {t+1} Episode Num: {episode_num+1} Reward: {episode_reward:.3f} Success: {reached} Success Rate: {success_rate:.2f}")
            writer.add_scalar("Reward/Episode", episode_reward, episode_num + 1)
            writer.add_scalar("Stats/SuccessRate", success_rate, t + 1)
            


            # Curriculum Update
            if args.env == "trap" and args.curriculum:
                if success_rate > 0.7:
                    new_difficulty = min(1.0, env.difficulty + 0.1)
                    if new_difficulty > env.difficulty:
                        print(f"Curriculum Level Up! New Difficulty: {new_difficulty:.1f}")
                        env.set_difficulty(new_difficulty)
                writer.add_scalar("Stats/Difficulty", env.difficulty, t + 1)

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
