import os
import sys

# ── Fix RTPS payload-too-small error (220 > 207 bytes) ──────────────────────
# Set FastDDS profile BEFORE ANY rclpy/gym imports so the middleware uses the XML.
_test_script_dir = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import TD3
import time
from train_env_disp_mem import DroneGazeboEnv

gym.register(
    id='GazeboIrisEnv-v0',
    entry_point='train_env_disp_mem:DroneGazeboEnv', 
)

def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    # eval_env.seed(seed + 100)
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, info = eval_env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, truncated, info = eval_env.step(action)
            avg_reward += reward
            
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")
    parser.add_argument("--env", default="GazeboIrisEnv-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--load_model", default="TD3_360_1236") # Default model name
    parser.add_argument("--episodes", default=1, type=int)
    parser.add_argument("--no_model", action="store_true", help="Run without loading a model (random policy)")
    parser.add_argument("--goal_x", default=2.0, type=float, help="Specific goal X coordinate")
    parser.add_argument("--goal_y", default=2.0, type=float, help="Specific goal Y coordinate")
    parser.add_argument("--random_goal", action="store_true", help="Use random goal instead of the specified fixed goal")
    
    args = parser.parse_args()
    
    print("---------------------------------------")
    print(f"Testing Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Model: {args.load_model}")
    print("---------------------------------------")

    # if not os.path.exists("./models"):
    #     print("Error: models directory not found.")
    #     exit(1)

    env = gym.make(args.env)

    # State dim = 64 (laser) + 6 (goal info) = 70
    state_dim = 70
    action_dim = 2
    max_action = 1.0
    
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }

    policy = TD3.TD3(**kwargs)
    
    if not args.no_model:
        model_path = f"models/{args.load_model}"
        try:
            policy.load(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
    else:
        print("Running without trained model (untrained policy).")

    for i in range(args.episodes):
        print(f"Episode {i+1}/{args.episodes}")
        goal = None if args.random_goal else (args.goal_x, args.goal_y)
        state, info = env.reset(options={"goal_pos": goal})
        done = False
        episode_reward = 0
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            # time.sleep(0.05) # Optional: slow down for visualization
            
        print(f"Episode Reward: {episode_reward:.3f}")

    # Automated Landing
    print("Mission complete. Pausing before landing...")
    time.sleep(2.0)
    env.unwrapped.land()
    print("Landing command sent.")
    time.sleep(5.0) # Wait for landing to finish

