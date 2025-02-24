
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNRat(nn.Module):
    def __init__(self):
        super(CNNRat, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=(3, 3), stride=(1, 1), padding=1) 

        # after pool1 : (3, 24, 53)
        # after conv : (50, 24, 53)
        # after pool2 : (50, 12, 26)
        flatten_size = 50 * 12 * 26
        # flatten_size = 50 * 12 * 5
        self.fc1 = nn.Linear(flatten_size, 1000)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.pool1(x) 
        x = F.relu(self.conv(x))
        x = self.pool2(x)
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        if (not self.training):
            x = F.softmax(x, dim=1)
        output = x
        
        return output