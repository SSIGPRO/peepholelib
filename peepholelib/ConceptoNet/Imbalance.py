import numpy as np

import torch
from torch.utils.data import WeightedRandomSampler

from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(y_train, device):
        """Calculate class weights for imbalanced dataset"""
        # Convert to integers if needed for sklearn compatibility
        y_train_int = y_train.detach().cpu().numpy() 
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_int),
            y=y_train_int
        )
        return torch.FloatTensor(class_weights).to(device)

def create_weighted_sampler(y_train):
        """Create weighted sampler for imbalanced dataset"""
        # Convert to integers if needed
        
        class_counts = torch.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        
        return WeightedRandomSampler(sample_weights, len(sample_weights))