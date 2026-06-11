import os
import argparse
import torch
import numpy as np

import sys

# Add GLBATT directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'GLBATT'))

# Import Actor classes from local modules
try:
    from TD3 import Actor as ActorTD3
    from TD3_Normal import Actor as ActorTD3Normal
    from TD3_PINN_Stable import Actor as ActorPINN
    from glbatt.architectures.GLBATT_Summary import GLBATT_Actor
except ImportError:
    sys.path.append(script_dir)
    from TD3 import Actor as ActorTD3
    from TD3_Normal import Actor as ActorTD3Normal
    from TD3_PINN_Stable import Actor as ActorPINN
    from glbatt.architectures.GLBATT_Summary import GLBATT_Actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3_Normal", choices=["TD3", "TD3_Normal", "TD3_PINN_Stable", "GLBATT"])
    parser.add_argument("--model_name", required=True, help="Base name of the model in models/ (e.g. td3, pi-td3, glbatt)")
    parser.add_argument("--state_dim", type=int, default=70)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--max_action", type=float, default=1.0)
    args = parser.parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    # Locate the model actor weights
    possible_paths = [
        os.path.join(models_dir, f"{args.model_name}_actor.pth"),
        os.path.join(models_dir, f"{args.model_name}_actor")
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
            
    if model_path is None:
        print(f"Error: Could not find model weights for '{args.model_name}' under models/ folder.")
        print(f"Tried paths: {possible_paths}")
        return

    # Force CPU device for all policy modules to prevent tracing device mismatch errors
    import TD3
    import TD3_Normal
    import TD3_PINN_Stable
    import glbatt.architectures.GLBATT_Summary as GSummary
    
    cpu_device = torch.device("cpu")
    TD3.device = cpu_device
    TD3_Normal.device = cpu_device
    TD3_PINN_Stable.device = cpu_device
    GSummary.device = cpu_device

    # 1. Instantiate the actor class based on policy type
    if args.policy == "TD3":
        actor = ActorTD3(args.state_dim, args.action_dim, args.max_action)
        dummy_input = torch.randn(1, args.state_dim)
        input_names = ['state']
        output_names = ['action']
        dynamic_axes = {'state': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
        
    elif args.policy == "TD3_Normal":
        actor = ActorTD3Normal(args.state_dim, args.action_dim, args.max_action)
        dummy_input = torch.randn(1, args.state_dim)
        input_names = ['state']
        output_names = ['action']
        dynamic_axes = {'state': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
        
    elif args.policy == "TD3_PINN_Stable":
        actor = ActorPINN(args.state_dim, args.action_dim, args.max_action)
        dummy_input = torch.randn(1, args.state_dim)
        input_names = ['state']
        output_names = ['action']
        dynamic_axes = {'state': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
        
    elif args.policy == "GLBATT":
        belief_state_indices = list(range(60, 70))
        actor = GLBATT_Actor(args.state_dim, args.action_dim, args.max_action, belief_state_indices=belief_state_indices)
        # GLBATT expects sequential inputs: (batch_size, sequence_length, state_dim)
        dummy_input = torch.randn(1, 50, args.state_dim) 
        input_names = ['state_seq']
        # Forward returns (action, belief, attn_weights)
        output_names = ['action', 'belief', 'attn_weights']
        dynamic_axes = {
            'state_seq': {0: 'batch_size', 1: 'seq_len'},
            'action': {0: 'batch_size'},
            'belief': {0: 'batch_size'},
            'attn_weights': {0: 'batch_size'}
        }

    # Load state dict (mapping to CPU for portability)
    device = torch.device("cpu")
    actor.to(device)
    try:
        actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except (TypeError, RuntimeError):
        try:
            actor.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
    actor.eval()

    # Export destination
    output_path = os.path.join(models_dir, f"{args.model_name}_actor.onnx")

    # Perform the export
    print(f"Exporting model from {model_path} to {output_path}...")
    torch.onnx.export(
        actor,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,  # supports newer operators like aten::unflatten used in GLBATT
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print(f"Successfully exported ONNX model to: {output_path}")

if __name__ == "__main__":
    main()
