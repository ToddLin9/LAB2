import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

# TODO implement EEGNet model
# class EEGNet(nn.Module):
#     def __init__(self):
#         super(EEGNet, self).__init__()
#         pass

#     def forward(self, x):
#         pass
    
class EEGNet(nn.Module):
    def __init__(self, args):
        super(EEGNet, self).__init__()
        self.args = args
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 51), stride = (1,1), padding = (0,25), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        # Layer 2
        self.conv2 = nn.Conv2d(16, 32, (2, 1), stride = (1,1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pooling2 = nn.AvgPool2d((1, 4), stride = (1,4), padding = 0)
        
        # Layer 3
        self.conv3 = nn.Conv2d(32, 32, (1, 15), stride = (1,1), padding = (0,7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.pooling3 = nn.AvgPool2d((1, 8), stride = (1, 8), padding = 0)
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(736, 2)
        

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = F.elu(self.batchnorm2(x), alpha = self.args.elu_alpha)
        x = F.dropout(self.pooling2(x), self.args.dropout_p)
        
        # Layer 3
        x = self.conv3(x)
        x = F.elu(self.batchnorm3(x), alpha = self.args.elu_alpha)
        x = F.dropout(self.pooling3(x), self.args.dropout_p)
        
        # Flatten tensor
        x = x.view(x.size(0), -1)

        # FC Layer
        x = self.fc1(x)
        
        return x

