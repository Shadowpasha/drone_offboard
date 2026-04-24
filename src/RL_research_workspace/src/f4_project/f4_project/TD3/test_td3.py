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
import time

# Jetson Nano Optimization: Limit PyTorch threads to prevent CPU thrashing
# especially in a ROS 2 environment with a lightweight model.
torch.set_num_threads(1)

try:
    from . import TD3
    from .train_env_disp_mem import DroneGazeboEnv
    from .real_drone_env import RealDroneEnv
except (ImportError, ValueError):
    import TD3
    from train_env_disp_mem import DroneGazeboEnv
    from real_drone_env import RealDroneEnv

gym.register(
    id='GazeboIrisEnv-v0',
    entry_point='f4_project.TD3.train_env_disp_mem:DroneGazeboEnv', 
)

gym.register(
    id='RealIrisEnv-v0',
    entry_point='f4_project.TD3.real_drone_env:RealDroneEnv', 
)

def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    # eval_env.seed(seed + 100)
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, info = eval_env.reset()
        done = False
        last_action = None
        while not done:
            action = policy.select_action(np.array(state))
            
            # Apply low pass filter smoothing (a=0.3)
            if last_action is not None:
                action = 0.3 * action + 0.7 * last_action
            last_action = action
            
            state, reward, done, truncated, info = eval_env.step(action)
            avg_reward += reward
            
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")
    parser.add_argument("--env", default="GazeboIrisEnv-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--load_model", default="TD3_360_89123789") # Default model name
    parser.add_argument("--episodes", default=1, type=int)
    parser.add_argument("--no_model", action="store_true", help="Run without loading a model (random policy)")
    parser.add_argument("--goal_x", default=1.0, type=float, help="Local Forward offset")
    parser.add_argument("--goal_y", default=7.0, type=float, help="Local Left offset")
    parser.add_argument("--random_goal", action="store_true", help="Use random goal instead of the specified fixed goal")
    parser.add_argument("--real", action="store_true", default=False, help="Use real drone environment instead of simulation")
    
    args = parser.parse_args()

    # Override env if --real is set
    if args.real:
        args.env = "RealIrisEnv-v0"
    
    print("---------------------------------------")
    print(f"Testing Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Model: {args.load_model}")
    print("---------------------------------------")

    # if not os.path.exists("./models"):
    #     print("Error: models directory not found.")
    #     exit(1)

    env = gym.make(args.env)

    # State dim = 128 (laser) + 6 (goal info) = 134
    state_dim = 134
    action_dim = 2
    max_action = 1.0
    
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }

    policy = TD3.TD3(**kwargs)
    
    if not args.no_model:
        # Improved model path resolution for ROS 2 compatibility
        # 1. Try local models/ directory (source run)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", args.load_model)
        
        # 2. Try fmu/models if not found (development run)
        if not os.path.exists(model_path + "_actor") and not os.path.exists(model_path + "_actor.zip"):
             # Sometimes the install path looks like site-packages/f4_project/TD3/...
             # find_spec or similar could be used but relative is often safer if setup.py is correct
             pass 

        try:
            policy.load(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            # Final fallback: check current working directory's models folder
            cwd_model_path = os.path.join(os.getcwd(), "models", args.load_model)
            try:
                policy.load(cwd_model_path)
                print(f"Loaded model from CWD fallback: {cwd_model_path}")
            except Exception as e2:
                 print(f"Critical Error: Could not load model from any location. ({e2})")
                 env.close()
                 exit(1)
    else:
        print("Running without trained model (untrained policy).")

    try:
        for i in range(args.episodes):
            print(f"Episode {i+1}/{args.episodes}")
            goal = None if args.random_goal else (args.goal_x, args.goal_y)
            state, info = env.reset(options={"goal_pos": goal})
            done = False
            episode_reward = 0
            last_action = None
            while not done:
                action = policy.select_action(np.array(state))
                
                # Apply low pass filter smoothing (a=0.3)
                if last_action is not None:
                    action = 0.3 * action + 0.7 * last_action
                last_action = action
                
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
    
    finally:
        print("Cleaning up...")
        env.close()


if __name__ == "__main__":
    main()

