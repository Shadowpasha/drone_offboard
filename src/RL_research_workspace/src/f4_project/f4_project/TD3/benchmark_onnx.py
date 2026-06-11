import os
import time
import numpy as np
import onnxruntime as ort

def benchmark_model(model_name, onnx_path, state_dim, is_sequence, seq_len=50, num_runs=1000, warmup_runs=100):
    if not os.path.exists(onnx_path):
        print(f"[SKIP] {model_name}: Model file not found at {onnx_path}")
        return None

    # Configure session options for edge deployment optimization
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Generate dummy input matching the shape
    if is_sequence:
        # GLBATT expects (batch_size, sequence_length, state_dim) -> (1, 50, 70)
        dummy_state = np.random.randn(1, seq_len, state_dim).astype(np.float32)
    else:
        # TD3/Normal/PINN expects (batch_size, state_dim) -> (1, 70)
        dummy_state = np.random.randn(1, state_dim).astype(np.float32)

    # Warmup runs to allow ONNX Runtime to initialize and optimize
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: dummy_state})

    # Benchmark runs
    latencies_ms = []
    for _ in range(num_runs):
        start_time = time.perf_counter_ns()
        _ = session.run(None, {input_name: dummy_state})
        end_time = time.perf_counter_ns()
        
        # Convert nanoseconds to milliseconds
        latencies_ms.append((end_time - start_time) / 1_000_000.0)

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
        "throughput": throughput
    }

def main():
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
    num_runs = 1000
    warmup_runs = 100

    print("=========================================================")
    print("           ONNX RUNTIME LATENCY BENCHMARK                ")
    print(f" Runs: {num_runs} | Warmup: {warmup_runs} | State Dim: {state_dim}")
    print("=========================================================")
    print(f"{'Model Name':<20} | {'Avg (ms)':<9} | {'Min (ms)':<9} | {'Max (ms)':<9} | {'Std (ms)':<9} | {'Throughput (Hz)':<15}")
    print("-" * 84)

    results = {}
    for item in models_to_test:
        res = benchmark_model(
            model_name=item["name"],
            onnx_path=item["path"],
            state_dim=state_dim,
            is_sequence=item["is_sequence"],
            seq_len=seq_len,
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        if res:
            print(f"{item['name']:<20} | {res['avg']:<9.4f} | {res['min']:<9.4f} | {res['max']:<9.4f} | {res['std']:<9.4f} | {res['throughput']:<15.1f}")
            results[item["name"]] = res
    print("=========================================================")

if __name__ == "__main__":
    main()
