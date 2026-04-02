import torch
import numpy as np

state_array = np.zeros((5,34))
state_array = torch.FloatTensor(
                np.expand_dims(state_array,0))

action = np.zeros((5,2))
action = torch.FloatTensor(
                np.expand_dims(action,0))

sa = torch.cat([state_array, action], 2)
print(sa.size())