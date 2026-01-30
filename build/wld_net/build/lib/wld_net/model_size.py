from dehazing_model import Dehazing_Model
import torch
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Dehazing_Model().to(device)

# Calculate parameters and size
num_params = count_parameters(model)
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
model_size_megabytes = param_bytes / (1024**2)

# Create output directory if it doesn't exist
folder_path = "./output/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Write results to file
with open('./output/model_stats.txt', 'w') as f:
    f.write(f'Number of trainable parameters: {num_params:,}\n')
    f.write(f'Model size: {model_size_megabytes:.2f} MB\n')

print(f'Number of trainable parameters: {num_params:,}')
print(f'Model size: {model_size_megabytes:.2f} MB')