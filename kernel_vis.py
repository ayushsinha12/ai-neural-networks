import cv2
import numpy
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))


kernels = conv_net.conv1.weight.data.clone()      
# normalize weights to [0,1]
kmin, kmax = kernels.min(), kernels.max()
kernels = (kernels - kmin) / (kmax - kmin)



# Create a plot that is a grid of images, where each image is one kernel from the conv layer.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, 
# the grid might have 4 rows and 8 columns. Finally, normalize the values in the grid to be 
# between 0 and 1 before plotting.

grid_k = make_grid(kernels, nrow=8, padding=1)
plt.figure(figsize=(8,8))
plt.imshow(grid_k.permute(1,2,0).cpu().numpy(), cmap='gray')
plt.axis('off')



# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.

plt.savefig('kernel_grid.png', bbox_inches='tight')
plt.close()



# Apply the kernel to the provided sample image.
# used generative ai to perfect
img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0					
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)

print(img.shape)

# Apply the kernel to the image
output = F.conv2d(img, conv_net.conv1.weight.data.clone(), bias=None, stride=1, padding=1)


# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
# If not needed for your implementation, you can remove these lines.

output = output.squeeze(0)
output = output.unsqueeze(1)


# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.

omin, omax = output.min(), output.max()
out_norm = (output - omin) / (omax - omin)

# make a grid and plot
grid_f = make_grid(out_norm, nrow=8, padding=1)
plt.figure(figsize=(8,8))
plt.imshow(grid_f.permute(1,2,0).cpu().numpy(), cmap='gray')
plt.axis('off')



# Save the grid to a file named 'image_transform_grid.png'. Add the saved image to the PDF report you submit.

plt.savefig('image_transform_grid.png', bbox_inches='tight')
plt.close()


# Create a feature map progression. You can manually specify the forward pass order or programatically track each activation through the forward pass of the CNN.

# capture first channel at each stage
stages = []
stages.append(('Original', img[0,0].cpu().numpy()))

x1 = F.relu(conv_net.conv1(img))
stages.append(('Conv1', x1[0,0].detach().cpu().numpy()))
p1 = conv_net.pool(x1)
stages.append(('Pool1', p1[0,0].detach().cpu().numpy()))

x2 = F.relu(conv_net.conv2(p1))
stages.append(('Conv2', x2[0,0].detach().cpu().numpy()))
p2 = conv_net.pool(x2)
stages.append(('Pool2', p2[0,0].detach().cpu().numpy()))

# plot them side by side
fig, axes = plt.subplots(1, len(stages), figsize=(4*len(stages), 4))
for ax, (name, data) in zip(axes, stages):
    mn, mx = data.min(), data.max()
    if mx > mn:
        data = (data - mn) / (mx - mn)
    ax.imshow(data, cmap='gray')
    ax.set_title(name)
    ax.axis('off')
fig.tight_layout()

# Save the image as a file named 'feature_progression.png'

plt.savefig('feature_progression.png', bbox_inches='tight')
plt.close()












