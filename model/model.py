
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os.path as path
import pandas as pd


class CNNRat(nn.module):
    def __init__(self):
        super(CNNRat, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=3, stride=1, padding=1) 
        self.relu1 = nn.ReLU()

        flatten_size = 50 * 3 * 39
        self.fc1 = nn.Linear(flatten_size, 1000)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.pool1(x) 
        x = F.relu(self.conv(x))
        x = self.pool2(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)    #F.relu(self.fc1(x)) 
        x = self.fc2(x)    #F.relu(self.fc2(x)) 
        output = F.softmax(x, dim=1)
        
        return output