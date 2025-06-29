import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a feedforward neural network. 

Please only complete the model definition and do not include any training code.

The model should be a feedforward neural network, that accepts 784 inputs (each image is 28x28, and is flattened for input to the network)
and the output size is 10. Whether you need to normalize outputs using softmax depends on your choice of loss function.

PyTorch documentation is available at https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton 
requires normalized outputs or not.

'''

class FF_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # first FC layer that has 784 inputs and 256 hidden units
        self.fc1     = nn.Linear(in_features=784, out_features=256)
        #used for regulization
        self.dropout = nn.Dropout(p=0.2)
        # final layer that turns 256 hidden units into 10 classes
        self.fc2     = nn.Linear(in_features=256, out_features=10)

    # used generative ai to fix small errors in code
    def forward(self, x):
        x = F.relu(self.fc1(x))   
        x = self.dropout(x)       
        x = self.fc2(x)           
        return x
        
