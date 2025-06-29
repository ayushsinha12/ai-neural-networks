import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a convolutional neural network. 

Please only complete the model definition and do not include any training code.

The model should be a convolutional neural network, that accepts 28x28 grayscale images as input, and outputs a tensor of size 10.
The number of layers/kernels, kernel sizes and strides are up to you. 

Please refer to the following for more information about convolutions, pooling, and convolutional layers in PyTorch:

    - https://deeplizard.com/learn/video/YRhxdVk_sIs
    - https://deeplizard.com/resource/pavq7noze2
    - https://deeplizard.com/resource/pavq7noze3
    - https://setosa.io/ev/image-kernels/
    - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


Whether you need to normalize outputs using softmax depends on your choice of loss function. PyTorch documentation is available at
https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton requires normalized outputs or not.

'''

class Conv_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # used generative ai to get a good estimate of what paramters numbers to use
        # creating the convolutional feature extractors
        self.conv1   = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # using max-pool to reduce spatial dims by 2x each time
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)

        # after two pools, feature maps 3136 features
        self.fc1     = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.dropout = nn.Dropout(p=0.25)

        # this is the final layers which has the 10 classes
        self.fc2     = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # used generative ai to fix small errors in code
        x = F.relu(self.conv1(x))  
        x = self.pool(x)           

        x = F.relu(self.conv2(x))  
        x = self.pool(x)           

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))    
        x = self.dropout(x)
        x = self.fc2(x)            
        return x
        
