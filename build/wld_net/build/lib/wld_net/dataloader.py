import torch
import os
from natsort import natsorted
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torchvision.transforms import Resize


class MyData(Dataset): #In use
    def __init__(self, path, image_size=(512,512)):
        self.filename_original =sorted(os.listdir(path+'//Hazy'), key=len) 
        self.filename_target = sorted(os.listdir(path+'//GT'), key=len)
        
        self.filename_original=natsorted(self.filename_original)
        self.filename_target=natsorted(self.filename_target)
        
        i=0
        while i<len(self.filename_original):
            self.filename_original[i]=path+'/Hazy/'+self.filename_original[i]
            self.filename_target[i]=path+'/GT/'+self.filename_target[i]
            i+=1
        
        self.image_size=image_size
    
    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self,idx):
        filename_o=self.filename_original[idx]
        filename_t=self.filename_target[idx]
        
        
        resize=transforms.Resize(self.image_size)
        #norm=transforms.Normalize([0.5], [0.5])
        
        real=Image.open(filename_o)
        real=resize(real)
        
        condition=Image.open(filename_t)
        condition=resize(condition)
        
        real=transforms.functional.to_tensor(real)
        #real=norm(real)
        condition=transforms.functional.to_tensor(condition)
        #condition=norm(condition)
        
        return real, condition




class MyData_Test(Dataset): #In use
    def __init__(self, path, resize_dimen=(1024,1024)):
        self.filename_original = sorted(os.listdir(path+'//Hazy'), key=len) 
        self.filename_target = sorted(os.listdir(path+'//GT'), key=len)
        
        self.filename_original = natsorted(self.filename_original)
        self.filename_target = natsorted(self.filename_target)
        
        i = 0
        while i < len(self.filename_original):
            self.filename_original[i] = path+'/Hazy/'+self.filename_original[i]
            self.filename_target[i] = path+'/GT/'+self.filename_target[i]
            i += 1
        
        self.resize = Resize((resize_dimen))  # Add this line to initialize the Resize transform
    
    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]
        filename_t = self.filename_target[idx]
        
        real = Image.open(filename_o)
        condition = Image.open(filename_t)
        
        real = self.resize(real)  # Add this line to resize the 'real' image
        condition = self.resize(condition)  # Add this line to resize the 'condition' image
        
        real = transforms.functional.to_tensor(real)
        condition = transforms.functional.to_tensor(condition)
        
        return real, condition



class MyData_Test_Single(Dataset):
    def __init__(self, path, resize_dimen=(1024,1024)):
        # Initialize the filename list first
        self.filename_original = sorted(os.listdir(os.path.join(path, 'Hazy')), key=len)
        self.filename_original = natsorted(self.filename_original)
        
        # Update paths
        self.filename_original = [os.path.join(path, 'Hazy', filename) 
                                for filename in self.filename_original]
        
        # Initialize the resize transform
        self.resize = Resize(resize_dimen)
    
    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]
        
        # Open and process the image
        real = Image.open(filename_o).convert('RGB')
        real = self.resize(real)
        real = transforms.functional.to_tensor(real)
        
        return real, real  # Return twice to match the expected format in test.py