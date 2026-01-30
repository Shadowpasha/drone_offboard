from dehazing_model import Dehazing_Model
import torch 
import time
import os

warmup_iterations = 50 # Set the number of warmup iterations
num_iterations = 200 # Set the number of iterations to average over

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device='cuda'
Images=[(128,128), (256,256), (512,512), (1024,1024)]
torch.cuda.empty_cache()
fps_l=[]
latency_l=[]
for i in range(len(Images)):
    t1_shape=(1,3,  *Images[i])
    # Initialize the model
    model = Dehazing_Model().to(device)
    model.eval()


    #Create Loop from  here
    # Create a random input tensor

    with torch.no_grad():
        input = torch.randn(t1_shape).to(device)

        # Warm-up iterations
        for _ in range(warmup_iterations):
            model(input)

        # Number of iterations to average over
        num_iterations = num_iterations

        # Record the start time
        start_time = time.time()
        # Run the model multiple times
        for _ in range(num_iterations):
            model(input)

    # Record the end time
    end_time = time.time()
    input=input.to('cpu')
    torch.cuda.empty_cache()
    # Calculate and print the average latency
    average_latency = (end_time - start_time) / num_iterations


    # Calculate and print the FPS
    fps = 1 / average_latency
    
    latency_l.append(average_latency)
    fps_l.append(fps)
    del model

model = Dehazing_Model().to(device)
num_params = count_parameters(model)
param_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
model_size_megabytes=param_bytes/(1024**2)

folder_path="./output/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    

with open('./output/latency_results.txt', 'w') as f:
    f.write(f'The model has: {num_params: ,} parameters\n')
    f.write(f"Model size: {model_size_megabytes: .2f} megabytes\n\n")
    for i in range(len(Images)):
        f.write(f"Image size: {Images[i]}\n")
        f.write(f"Average Latency: {latency_l[i]: .3f} seconds\n")
        f.write(f"FPS: {fps_l[i]: .1f}\n\n")

    f.close()


