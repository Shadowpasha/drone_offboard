import os
import sys

# ── Fix RTPS payload-too-small error (220 > 207 bytes) ──────────────────────
# Set FastDDS profile BEFORE ANY rclpy/gym imports so the middleware uses the XML.
_test_script_dir = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import gymnasium as gym
import argparse
import os
import time

# Add GLBATT directory to path
sys.path.append(os.path.join(_test_script_dir, 'GLBATT'))

try:
    from .train_env_disp_mem import DroneGazeboEnv, RealDroneEnv
except (ImportError, ValueError):
    from train_env_disp_mem import DroneGazeboEnv, RealDroneEnv

gym.register(
    id='GazeboIrisEnv-v0',
    entry_point='f4_project.TD3.train_env_disp_mem:DroneGazeboEnv', 
)

gym.register(
    id='RealIrisEnv-v0',
    entry_point='f4_project.TD3.train_env_disp_mem:RealDroneEnv', 
)

class ONNXPolicy:
    def __init__(self, onnx_model_path):
        import onnxruntime as ort
        
        # Optimize CPU threads for Jetson Nano / edge deployment
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        
        self.session = ort.InferenceSession(onnx_model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
    def select_action(self, state):
        state_input = np.array(state, dtype=np.float32)
        if len(state_input.shape) == 1:
            state_input = state_input.reshape(1, -1)
        elif len(state_input.shape) == 2:
            state_input = state_input.reshape(1, state_input.shape[0], -1)
            
        outputs = self.session.run(None, {self.input_name: state_input})
        return outputs[0].flatten()


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
    parser.add_argument("--policy", default="TD3_PINN_Stable")
    parser.add_argument("--env", default="GazeboIrisEnv-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--load_model", default=None)  # Default depends on selected policy
    parser.add_argument("--seq_len", default=50, type=int, help="Sequence window length for GLBATT")
    parser.add_argument("--episodes", default=1, type=int)
    parser.add_argument("--no_model", action="store_true", help="Run without loading a model (random policy)")
    parser.add_argument("--goal_x", default=4.0, type=float, help="Local Forward offset")
    parser.add_argument("--goal_y", default=0.0, type=float, help="Local Left offset")
    parser.add_argument("--random_goal", action="store_true", default=False,  help="Use random goal instead of the specified fixed goal")
    parser.add_argument("--real", action="store_true", default=False, help="Use real drone environment instead of simulation")
    parser.add_argument("--onnx", action="store_true", default=True, help="Load the model as ONNX instead of PyTorch")
    
    args = parser.parse_args()

    # Override env if --real is set
    if args.real:
        args.env = "RealIrisEnv-v0"
    
    # Normalize policy input and set corresponding smart defaults for load_model
    policy_input = args.policy.strip()
    policy_lower = policy_input.lower().replace("_", " ").replace("-", " ")
    
    if "pinn" in policy_lower:
        args.policy = "TD3_PINN_Stable"
        default_model = "pi-td3"
    elif "glbnatt" in policy_lower or "glbatt" in policy_lower:
        args.policy = "GLBATT"
        default_model = "glbatt"
    elif "normal" in policy_lower or policy_lower == "td3":
        args.policy = "TD3_Normal"
        default_model = "td3"
    elif policy_input == "TD3":
        args.policy = "TD3"
        default_model = "td3"
    else:
        # Fallback/custom policy name
        default_model = "td3_10_06_2026_01_56"

    # Automatically set/override the load_model if not explicitly provided
    if args.load_model is None:
        args.load_model = default_model
    
    print("---------------------------------------")
    print(f"Testing Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Model: {args.load_model}")
    print(f"Backend: {'ONNX' if args.onnx else 'PyTorch'}")
    print("---------------------------------------")

    # if not os.path.exists("./models"):
    #     print("Error: models directory not found.")
    #     exit(1)

    env = gym.make(args.env)

    state_dim = 70  # 60 laser rays + 10 goal/state data (matches MuJoCo deployment env)
    
    action_dim = 2
    max_action = 1.0
    
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }

    if args.no_model:
        # Dummy random policy wrapper
        class RandomPolicy:
            def select_action(self, state):
                return env.action_space.sample()
        policy = RandomPolicy()
        print("Running without trained model (untrained random policy).")
        
    elif args.onnx:
        # Load via ONNX Runtime
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", f"{args.load_model}_actor.onnx")
        
        if not os.path.exists(model_path):
            print(f"Error: ONNX file not found at {model_path}")
            print("Please export your PyTorch model to ONNX using export_to_onnx.py first.")
            env.close()
            exit(1)
            
        policy = ONNXPolicy(model_path)
        print(f"Loaded ONNX model from {model_path}")
        
    else:
        # Load via PyTorch
        import torch
        torch.set_num_threads(1)  # Thread limiting for Jetson Nano
        
        try:
            from . import TD3, TD3_Normal, TD3_PINN_Stable
        except (ImportError, ValueError):
            import TD3, TD3_Normal, TD3_PINN_Stable
            
        if args.policy == "TD3":
            policy = TD3.TD3(**kwargs)
        elif args.policy == "TD3_Normal":
            policy = TD3_Normal.TD3(**kwargs)
        elif args.policy == "TD3_PINN_Stable":
            policy = TD3_PINN_Stable.TD3_PINN_Stable(**kwargs)
        elif args.policy in ["GLBATT", "TD3_GLBATT"]:
            from glbatt.architectures import GLBATT_Summary as GLBATT_Lib
            kwargs["belief_state_indices"] = list(range(60, 70))
            policy = GLBATT_Lib.GLBATT(**kwargs)
        else:
            print(f"Unknown policy: {args.policy}")
            env.close()
            exit(1)
            
        # Improved model path resolution for ROS 2 compatibility
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", args.load_model)
        
        try:
            policy.load(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            cwd_model_path = os.path.join(os.getcwd(), "models", args.load_model)
            try:
                policy.load(cwd_model_path)
                print(f"Loaded model from CWD fallback: {cwd_model_path}")
            except Exception as e2:
                 print(f"Critical Error: Could not load model from any location. ({e2})")
                 env.close()
                 exit(1)

    try:
        for i in range(args.episodes):
            print(f"Episode {i+1}/{args.episodes}")
            goal = None if args.random_goal else (args.goal_x, args.goal_y)
            state, info = env.reset(options={"goal_pos": goal})
            done = False
            episode_reward = 0
            last_action = None
            if args.policy in ["GLBATT", "TD3_GLBATT"]:
                state_seq = np.zeros((args.seq_len, state_dim))
                for idx_seq in range(args.seq_len):
                    state_seq[idx_seq] = state

            while not done:
                # --- SAFETY CHECK ON HIGH-RES RAW LIDAR DATA ---
                # Retrieve the raw, full-resolution lidar ranges stored directly from the ROS callback
                raw_laser = getattr(env.unwrapped, 'raw_ranges', np.ones(1080) * 12.0)
                
                # Minimum safe distance: holybro frame (radius 0.25m) + 10" propellers (radius ~0.127m) + 0.2m = 0.577m
                min_safe_dist = 0.25 + 0.127 + 0.2
                
                if np.min(raw_laser) < min_safe_dist:
                    print(f"Failed navigation: Obstacle detected at {np.min(raw_laser):.3f}m (limit: {min_safe_dist:.3f}m) in raw lidar data.")
                    print("Stopping immediately and landing...")
                    # Send zero velocity to stop immediately
                    try:
                        env.step(np.array([0.0, 0.0]))
                    except Exception:
                        pass
                        
                    time.sleep(2.0)
                    try:
                        env.unwrapped.land()
                        print("Landing command sent.")
                    except AttributeError:
                        print("Land method not found on unwrapped environment.")
                        
                    time.sleep(5.0)
                    break
                # ----------------------------------------------------

                if args.policy in ["GLBATT", "TD3_GLBATT"]:
                    action = policy.select_action(state_seq)
                else:
                    action = policy.select_action(np.array(state))
                
                # Apply low pass filter smoothing (a=0.3)
                if last_action is not None:
                    action = 0.3 * action + 0.7 * last_action
                last_action = action
                
                state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                # time.sleep(0.05) # Optional: slow down for visualization

                if args.policy in ["GLBATT", "TD3_GLBATT"]:
                    state_seq = np.roll(state_seq, -1, axis=0)
                    state_seq[-1] = state
                
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

