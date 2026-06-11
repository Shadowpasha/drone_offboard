import os
import time
import numpy as np
import argparse
import ctypes
import threading
import psutil
import glob

_libcudart = None
def get_cuda_free_mem_mb():
    global _libcudart
    if _libcudart is None:
        for path in ['/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so', 'libcudart.so']:
            try:
                _libcudart = ctypes.CDLL(path)
                _libcudart.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
                _libcudart.cudaMemGetInfo.restype = int
                break
            except OSError:
                continue
    if _libcudart is not None:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        try:
            res = _libcudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
            if res == 0:
                return free.value / (1024.0 * 1024.0)
        except Exception:
            pass
    return None

def find_gpu_load_node():
    paths = [
        "/sys/devices/platform/bus@0/17000000.gpu/load",
        "/sys/devices/gpu.0/load"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    try:
        matches = glob.glob("/sys/devices/platform/**/*.gpu/load", recursive=True)
        if matches:
            return matches[0]
    except Exception:
        pass
    return None

GPU_LOAD_NODE = find_gpu_load_node()

class GPUMonitor(threading.Thread):
    def __init__(self, interval=0.01):
        super().__init__()
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()
        self.loads = []
        
    def run(self):
        if not GPU_LOAD_NODE:
            return
        while not self.stop_event.is_set():
            try:
                with open(GPU_LOAD_NODE, 'r') as f:
                    val = int(f.read().strip())
                    # load is 0-1000 on Jetson Orin. Convert to 0-100%
                    self.loads.append(val / 10.0)
            except Exception:
                pass
            time.sleep(self.interval)
            
    def stop(self):
        self.stop_event.set()
        try:
            self.join()
        except Exception:
            pass
        if self.loads:
            return np.mean(self.loads)
        return 0.0

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


def benchmark_policy(model_name, policy, state_dim, is_sequence, seq_len=50, num_runs=1000, warmup_runs=100):
    # Generate dummy input matching the shape expected by policy (without batch dimension)
    if is_sequence:
        # GLBATT expects (sequence_length, state_dim) -> (50, 70)
        dummy_state = np.random.randn(seq_len, state_dim).astype(np.float32)
    else:
        # TD3/Normal/PINN expects (state_dim) -> (70,)
        dummy_state = np.random.randn(state_dim).astype(np.float32)

    # Warmup runs to allow policy/session to initialize and optimize
    for _ in range(warmup_runs):
        _ = policy.select_action(dummy_state)

    # Start GPU Monitor thread
    gpu_monitor = GPUMonitor(interval=0.01)
    gpu_monitor.start()

    # Start CPU and RAM tracking
    proc = psutil.Process()
    # Call cpu_percent once to clear state
    try:
        proc.cpu_percent()
    except Exception:
        pass
    start_cpu_times = proc.cpu_times()
    start_time = time.perf_counter()

    # Benchmark runs
    latencies_ms = []
    for _ in range(num_runs):
        start_time_ns = time.perf_counter_ns()
        _ = policy.select_action(dummy_state)
        end_time_ns = time.perf_counter_ns()
        
        # Convert nanoseconds to milliseconds
        latencies_ms.append((end_time_ns - start_time_ns) / 1_000_000.0)

    end_time = time.perf_counter()
    end_cpu_times = proc.cpu_times()
    
    # Stop GPU Monitor and get average load
    avg_gpu_load = gpu_monitor.stop()

    # Calculate CPU usage
    elapsed = end_time - start_time
    cpu_user = end_cpu_times.user - start_cpu_times.user
    cpu_system = end_cpu_times.system - start_cpu_times.system
    total_cpu_time = cpu_user + cpu_system
    # Process CPU percent (single core equivalent, e.g. 100% = 1 core fully busy)
    cpu_percent = (total_cpu_time / elapsed) * 100.0 if elapsed > 0 else 0.0

    # Memory consumption (Resident Set Size)
    final_ram = proc.memory_info().rss / (1024.0 * 1024.0)

    avg_lat = np.mean(latencies_ms)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)
    std_lat = np.std(latencies_ms)
    throughput = 1000.0 / avg_lat

    return {
        "avg": avg_lat,
        "min": min_lat,
        "max": max_lat,
        "std": std_lat,
        "throughput": throughput,
        "cpu_percent": cpu_percent,
        "gpu_percent": avg_gpu_load,
        "ram_mb": final_ram
    }

def main():
    parser = argparse.ArgumentParser(description="Inference latency benchmark for ONNX Runtime CPU and TensorRT GPU.")
    parser.add_argument("--runs", type=int, default=1000, help="Number of benchmark iterations (default: 1000)")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations (default: 100)")
    parser.add_argument("--cpu", action="store_true", help="Run only CPU (ONNX Runtime) benchmark")
    parser.add_argument("--gpu", action="store_true", help="Run only GPU (TensorRT) benchmark")
    args = parser.parse_args()

    # If neither CPU nor GPU flags are specified, default to running both
    run_cpu = args.cpu or not args.gpu
    run_gpu = args.gpu or not args.cpu

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")

    # Define the models to benchmark
    models_to_test = [
        {
            "name": "TD3 (td3)",
            "path": os.path.join(models_dir, "td3_actor.onnx"),
            "is_sequence": False
        },
        {
            "name": "TD3 PINN (pi-td3)",
            "path": os.path.join(models_dir, "pi-td3_actor.onnx"),
            "is_sequence": False
        },
        {
            "name": "GLBATT (glbatt)",
            "path": os.path.join(models_dir, "glbatt_actor.onnx"),
            "is_sequence": True
        }
    ]

    state_dim = 70
    seq_len = 50

    print("========================================================================================================================")
    print("                                            INFERENCE LATENCY & RESOURCE BENCHMARK                                      ")
    print(f" Runs: {args.runs} | Warmup: {args.warmup} | State Dim: {state_dim} | Seq Len: {seq_len}")
    print("========================================================================================================================")
    print(f"{'Model Name':<20} | {'Backend':<9} | {'Avg (ms)':<8} | {'Min (ms)':<8} | {'Max (ms)':<8} | {'Tput (Hz)':<9} | {'CPU (%)':<8} | {'GPU (%)':<8} | {'RAM (MB)':<8} | {'GPU Mem (MB)':<12}")
    print("-" * 120)

    for item in models_to_test:
        if not os.path.exists(item["path"]):
            print(f"[SKIP] {item['name']}: Model file not found at {item['path']}")
            continue

        # 1. Run ONNX CPU benchmark if requested
        if run_cpu:
            try:
                gpu_mem_before = get_cuda_free_mem_mb()
                
                cpu_policy = ONNXPolicy(item["path"])
                res_cpu = benchmark_policy(
                    model_name=item["name"],
                    policy=cpu_policy,
                    state_dim=state_dim,
                    is_sequence=item["is_sequence"],
                    seq_len=seq_len,
                    num_runs=args.runs,
                    warmup_runs=args.warmup
                )
                gpu_mem_after = get_cuda_free_mem_mb()
                
                gpu_mem_used = 0.0
                if gpu_mem_before is not None and gpu_mem_after is not None:
                    gpu_mem_used = max(0.0, gpu_mem_before - gpu_mem_after)
                    
                gpu_mem_str = f"{gpu_mem_used:.1f}" if gpu_mem_before is not None else "N/A"
                print(f"{item['name']:<20} | {'ONNX CPU':<9} | {res_cpu['avg']:<8.4f} | {res_cpu['min']:<8.4f} | {res_cpu['max']:<8.4f} | {res_cpu['throughput']:<9.1f} | {res_cpu['cpu_percent']:<8.1f} | {res_cpu['gpu_percent']:<8.1f} | {res_cpu['ram_mb']:<8.1f} | {gpu_mem_str:<12}")
            except Exception as e:
                print(f"{item['name']:<20} | {'ONNX CPU':<9} | Error running CPU benchmark: {e}")

        # 2. Run TensorRT GPU benchmark if requested
        if run_gpu:
            try:
                gpu_mem_before = get_cuda_free_mem_mb()
                
                gpu_policy = TensorRTPolicy(item["path"])
                res_gpu = benchmark_policy(
                    model_name=item["name"],
                    policy=gpu_policy,
                    state_dim=state_dim,
                    is_sequence=item["is_sequence"],
                    seq_len=seq_len,
                    num_runs=args.runs,
                    warmup_runs=args.warmup
                )
                gpu_mem_after = get_cuda_free_mem_mb()
                
                gpu_mem_used = 0.0
                if gpu_mem_before is not None and gpu_mem_after is not None:
                    gpu_mem_used = max(0.0, gpu_mem_before - gpu_mem_after)
                    
                gpu_mem_str = f"{gpu_mem_used:.1f}" if gpu_mem_before is not None else "N/A"
                print(f"{item['name']:<20} | {'TRT GPU':<9} | {res_gpu['avg']:<8.4f} | {res_gpu['min']:<8.4f} | {res_gpu['max']:<8.4f} | {res_gpu['throughput']:<9.1f} | {res_gpu['cpu_percent']:<8.1f} | {res_gpu['gpu_percent']:<8.1f} | {res_gpu['ram_mb']:<8.1f} | {gpu_mem_str:<12}")
            except Exception as e:
                print(f"{item['name']:<20} | {'TRT GPU':<9} | Error running GPU benchmark: {e}")
        
        print("-" * 120)

    print("========================================================================================================================")

if __name__ == "__main__":
    main()
