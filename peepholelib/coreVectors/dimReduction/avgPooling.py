# torch stuff

import torch
import torchvision

def ChannelWiseMean_conv(act_data): 
    
    act_data = act_data.view(act_data.size(0), act_data.size(1), -1)
    cvs = torch.mean(act_data,2)
    
    return cvs
