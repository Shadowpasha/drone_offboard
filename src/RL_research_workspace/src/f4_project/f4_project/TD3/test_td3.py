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
import ctypes

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


class TensorRTPolicy:
    def __init__(self, onnx_model_path):
        import tensorrt as trt
        
        self.onnx_path = onnx_model_path
        self.engine_path = onnx_model_path.replace(".onnx", ".engine")
        
        # Load libcudart
        try:
            self.libcudart = ctypes.CDLL('/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so')
        except OSError:
            try:
                self.libcudart = ctypes.CDLL('libcudart.so')
            except OSError:
                raise RuntimeError("Could not load libcudart.so. Please make sure CUDA is installed.")
                
        self.libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.libcudart.cudaMalloc.restype = int
        self.libcudart.cudaFree.argtypes = [ctypes.c_void_p]
        self.libcudart.cudaFree.restype = int
        self.libcudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.libcudart.cudaMemcpyAsync.restype = int
        self.libcudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.libcudart.cudaStreamCreate.restype = int
        self.libcudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.libcudart.cudaStreamDestroy.restype = int
        self.libcudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.libcudart.cudaStreamSynchronize.restype = int
        
        # Create non-default CUDA stream
        self.stream = ctypes.c_void_p()
        self.libcudart.cudaStreamCreate(ctypes.byref(self.stream))
        
        # Build engine if not exists
        if not os.path.exists(self.engine_path):
            self.build_engine()
            
        # Load engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(self.engine_path, "rb") as f:
            serialized_engine = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.engine_path}")
            
        self.context = self.engine.create_execution_context()
        
        # Buffer pre-allocation variables
        self.d_inputs = {}
        self.d_outputs = {}
        self.allocated_shapes = {}
        self.action_host = None
        self.action_ptr = None

    def cleanup_buffers(self):
        for name, d_ptr in list(self.d_inputs.items()):
            self.libcudart.cudaFree(d_ptr)
        for name, d_ptr in list(self.d_outputs.items()):
            self.libcudart.cudaFree(d_ptr)
        self.d_inputs = {}
        self.d_outputs = {}
        self.allocated_shapes = {}
        self.action_host = None
        self.action_ptr = None

    def __del__(self):
        if hasattr(self, 'libcudart'):
            self.cleanup_buffers()
            if hasattr(self, 'stream') and self.stream.value is not None:
                self.libcudart.cudaStreamDestroy(self.stream)
                self.stream = ctypes.c_void_p()
        
    def build_engine(self):
        import tensorrt as trt
        print(f"Building TensorRT engine for {self.onnx_path}... (this may take up to a minute)")
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("Failed to parse ONNX file:")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed.")
                
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
        
        profile = builder.create_optimization_profile()
        has_dynamic = False
        
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = inp.shape
            is_input_dynamic = any(dim < 0 for dim in shape)
            if is_input_dynamic:
                has_dynamic = True
                if len(shape) == 2:
                    dim1 = shape[1] if shape[1] > 0 else 70
                    min_shape = [1, dim1]
                    opt_shape = [1, dim1]
                    max_shape = [1, dim1]
                elif len(shape) == 3:
                    dim1 = shape[1] if shape[1] > 0 else 50
                    dim2 = shape[2] if shape[2] > 0 else 70
                    min_shape = [1, dim1, dim2]
                    opt_shape = [1, dim1, dim2]
                    max_shape = [1, dim1, dim2]
                else:
                    min_shape = [1] * len(shape)
                    opt_shape = [1] * len(shape)
                    max_shape = [1] * len(shape)
                profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
                
        if has_dynamic:
            config.add_optimization_profile(profile)
            
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build serialized network.")
            
        with open(self.engine_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Engine saved to {self.engine_path}")

    def select_action(self, state):
        import tensorrt as trt
        
        state_input = np.array(state, dtype=np.float32)
        if len(state_input.shape) == 1:
            state_input = state_input.reshape(1, -1)
        elif len(state_input.shape) == 2:
            state_input = state_input.reshape(1, state_input.shape[0], -1)
            
        current_shape = state_input.shape
        
        # Setup buffers if shape changed or first call
        if self.allocated_shapes.get("input") != current_shape:
            self.cleanup_buffers()
            
            # 1. Setup inputs
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.context.set_input_shape(name, current_shape)
                    d_input = ctypes.c_void_p()
                    self.libcudart.cudaMalloc(ctypes.byref(d_input), state_input.nbytes)
                    self.d_inputs[name] = d_input
                    
            # 2. Setup outputs
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.OUTPUT:
                    output_shape = self.context.get_tensor_shape(name)
                    output_nbytes = np.prod(output_shape) * 4 # float32
                    d_output = ctypes.c_void_p()
                    self.libcudart.cudaMalloc(ctypes.byref(d_output), int(output_nbytes))
                    self.d_outputs[name] = d_output
                    
                    # Identify action tensor: name is 'action', or shape ends with 2
                    if name == 'action' or (output_shape[-1] == 2 and self.action_host is None):
                        self.action_host = np.zeros(output_shape, dtype=np.float32)
                        self.action_ptr = d_output
                        
            # Bind all addresses
            for name, d_ptr in self.d_inputs.items():
                self.context.set_tensor_address(name, d_ptr.value)
            for name, d_ptr in self.d_outputs.items():
                self.context.set_tensor_address(name, d_ptr.value)
                
            self.allocated_shapes["input"] = current_shape
            
        # 3. Copy input to device
        for name, d_ptr in self.d_inputs.items():
            self.libcudart.cudaMemcpyAsync(d_ptr, state_input.ctypes.data, state_input.nbytes, 1, self.stream) # H2D = 1
            
        # 4. Execute
        self.context.execute_async_v3(self.stream.value)
        
        # 5. Copy back action
        if self.action_host is not None:
            self.libcudart.cudaMemcpyAsync(self.action_host.ctypes.data, self.action_ptr, self.action_host.nbytes, 2, self.stream) # D2H = 2
            self.libcudart.cudaStreamSynchronize(self.stream)
            action = self.action_host.flatten()
        else:
            action = np.zeros(2, dtype=np.float32)
            
        return action



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
    parser.add_argument("--gpu", action="store_true", default=False, help="Run on GPU using TensorRT (requires --onnx)")
    
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
        # Load via ONNX / TensorRT
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", f"{args.load_model}_actor.onnx")
        
        if not os.path.exists(model_path):
            print(f"Error: ONNX file not found at {model_path}")
            print("Please export your PyTorch model to ONNX using export_to_onnx.py first.")
            env.close()
            exit(1)
            
        if args.gpu:
            try:
                policy = TensorRTPolicy(model_path)
                print(f"Loaded TensorRT GPU engine from/for {model_path}")
            except Exception as e:
                print(f"Failed to load TensorRT policy: {e}")
                print("Falling back to CPU ONNXPolicy...")
                policy = ONNXPolicy(model_path)
        else:
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

