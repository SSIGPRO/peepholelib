# torch stuff

import torch
import torchvision

def ChannelWiseMean_conv(act_data): 
    
    act_data = act_data.view(act_data.size(0), act_data.size(1), -1)
    cvs = torch.mean(act_data,2)
    
    return cvs

def cls_token_ViT(act_data):
    """
    Extract the class token from ViT activations.
    """
    # Assuming the class token is the first token in the sequence
    
    return act_data[:, 0, :]

def TokenWiseMean_ViT(act_data):
    """
    Compute the channel-wise mean for ViT activations.
    """
    # Assuming the input is of shape (batch_size, num_tokens, num_channels)
    act_data = act_data[:, 1:, :]  # Exclude the class token
    cvs = torch.mean(act_data, dim=2)  # Mean across tokens
    print("Token-wise mean shape:", cvs.shape)
    
    return cvs