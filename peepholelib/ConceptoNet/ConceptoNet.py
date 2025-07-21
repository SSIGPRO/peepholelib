#python stuff
import pandas as pd
import seaborn as sb
import numpy as np
from math import floor
from matplotlib import pyplot as plt

#Sklearn stuff
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

#torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights, vit_b_16
from cuda_selector import auto_cuda
from torch.nn.functional import softmax as sm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ParametrizableCNN(nn.Module):
    def __init__(self, input_height=100, input_width=None, num_channels=1, output=1, 
                 conv_channels=[32, 64,128], kernel_sizes=[3, 3, 3], 
                 fc_hidden_size=128, dropout_rate=0.5):
        """
        Parametrizable CNN for binary classification with variable input dimensions.
        
        Args:
            input_height (int): Fixed height dimension (default: 100)
            input_width (int): Variable width dimension
            num_channels (int): Number of input channels
            conv_channels (list): Number of channels for each conv layer
            kernel_sizes (list): Kernel sizes for each conv layer
            fc_hidden_size (int): Hidden size for fully connected layer
            dropout_rate (float): Dropout rate
        """
        super(ParametrizableCNN, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = num_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            self.conv_layers.append(conv_layer)
            in_channels = out_channels
        
        # Calculate the size after convolutions
        self.feature_size = self._calculate_conv_output_size(input_height, input_width, len(conv_channels))
        self.final_channels = conv_channels[-1]
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.final_channels * self.feature_size, fc_hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_hidden_size, output)  # Binary classification

    def _calculate_conv_output_size(self, h, w, num_pools):
        """Calculate output size after convolutions and pooling"""
        for _ in range(num_pools):
            h = h // 2
            w = w // 2
        return h * w
    
    def forward(self, x):
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x.squeeze()
    