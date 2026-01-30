import torch
import dehazing_model
from pathlib import Path
from dataloader import MyData_Test_Single
import os
import sys
from pathlib import Path
import Feature_Processing
from matplotlib.pyplot import imsave

sys.path.append(str(Path(__file__).resolve().parent.parent))
# Get the directory where the script is located
script_dir = Path(__file__).resolve().parent

#Select size of the input images. Default is 1024x1024
Image_size=(1024,1024)
#Image_size=(512,512)

# Pick the model weights based on the dataset it was trained on. Comment out the other model paths
model_path='./models/RD_dehazing_model_final.pth'  #For model trained on RESIDE dataset
#model_path='./models/OH_dehazing_model_final.pth'  #For model trained on O-HAZE dataset
#model_path='./models/NH_dehazing_model_final.pth'  #For model trained on NH-HAZE dataset
#model_path='./models/DH_dehazing_model_final.pth'  #For model trained on D-HAZE dataset


input_path = script_dir  / '..' / 'test_input'
input_path = os.path.normpath(input_path) 


def process_images(model, test_data, storage_path, Img_size=(1024,1024)):
    test_data=MyData_Test_Single(test_data, Img_size)
    test_data_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    
    
    i=1
    # Loop over the test data
    for hazy_img, gt_img in test_data_dataloader:

        # Input the hazy image to the model
        with torch.no_grad():
            hazy_img=hazy_img.to('cuda')
            gt_img=gt_img.to('cuda')
            hazy_img=Feature_Processing.normalize(hazy_img)
            dehazed_img = model(hazy_img)
        
        hazy_img=Feature_Processing.denormalize(hazy_img)
        dehazed_img=Feature_Processing.denormalize(dehazed_img)
        
        dehazed_img_np = dehazed_img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        imsave(storage_path+ '/dehazed_{}.png'.format(i), dehazed_img_np)
        print(f'Image {i} processed')
        i+=1
    


model = dehazing_model.Dehazing_Model()  # Replace with your model class



folder_path="./output/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    

model.load_state_dict(torch.load(model_path))  # Replace with your model path
model.to('cuda')
# Set the model to evaluation mode
model.eval()

process_images(model, input_path, folder_path, Image_size)
print("All Images Processed")
