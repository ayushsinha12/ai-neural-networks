import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
from ffn import *
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
'''

In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.

'''


'''

PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted. Please do not change the transforms given below - the autograder assumes these.

'''

transform = transforms.Compose([                            
    transforms.ToTensor(),                                  
    transforms.Normalize(mean=[0.5], std=[0.5])             
])

batch_size = 64


'''

PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.

'''

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


'''

PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.

'''


feedforward_net = FF_Net()
conv_net = Conv_Net()



'''

PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.

'''

criterion = nn.CrossEntropyLoss()

optimizer_ffn = optim.Adam(feedforward_net.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(conv_net.parameters(),      lr=0.001)



'''

PART 5:
Train both your models, one at a time! (You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs, but it is not recommended for this assignment.)

'''


num_epochs_ffn = 15
ffn_losses = []
for epoch in range(num_epochs_ffn):  # loop over the dataset multiple times
    running_loss_ffn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Flatten inputs for ffn

        ''' YOUR CODE HERE '''
        inputs = inputs.view(inputs.size(0), -1)

        # zero the parameter gradients
        optimizer_ffn.zero_grad()

        # forward + backward + optimize
        outputs = feedforward_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ffn.step()
        running_loss_ffn += loss.item()

    print(f"Training loss: {running_loss_ffn}")
    ffn_losses.append(running_loss_ffn)

print('Finished Training')

torch.save(feedforward_net.state_dict(), 'ffn.pth')  # Saves model file (upload with submission)


num_epochs_cnn = 10
cnn_losses = []

for epoch in range(num_epochs_cnn):  # loop over the dataset multiple times
    running_loss_cnn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer_cnn.zero_grad()

        # forward + backward + optimize
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()

    print(f"Training loss: {running_loss_cnn}")
    cnn_losses.append(running_loss_cnn)

print('Finished Training')

torch.save(conv_net.state_dict(), 'cnn.pth')  # Saves model file (upload with submission)


'''

PART 6:
Evalute your models! Accuracy should be greater or equal to 80% for both models.

Code to load saved weights commented out below - may be useful for debugging.

'''

# feedforward_net.load_state_dict(torch.load('ffn.pth'))
# conv_net.load_state_dict(torch.load('cnn.pth'))

correct_ffn = 0
total_ffn = 0

correct_cnn = 0
total_cnn = 0

with torch.no_grad():           
    for data in testloader:
        inputs, labels = data

        # FFN eval
        out_ffn = feedforward_net(inputs.view(inputs.size(0), -1))
        _, pred_ffn = torch.max(out_ffn, 1)
        total_ffn   += labels.size(0)
        correct_ffn += (pred_ffn == labels).sum().item()

        # CNN eval
        out_cnn = conv_net(inputs)
        _, pred_cnn = torch.max(out_cnn, 1)
        total_cnn   += labels.size(0)
        correct_cnn += (pred_cnn == labels).sum().item()


print('Accuracy for feedforward network: ', correct_ffn/total_ffn)
print('Accuracy for convolutional network: ', correct_cnn/total_cnn)


'''

PART 7:

Check the instructions PDF. You need to generate some plots. 

'''

# Assume you collected losses: ffn_losses, cnn_losses
# (You can append running_loss_ffn and running_loss_cnn each epoch.)

# 1. Loss curves
plt.figure()
plt.plot(range(1, len(ffn_losses)+1), ffn_losses, label='FFN Loss')
plt.plot(range(1, len(cnn_losses)+1), cnn_losses, label='CNN Loss')
plt.xlabel('Epoch'); plt.ylabel('Training Loss'); plt.legend()
plt.savefig('loss_curves.png'); plt.close()

def plot_example(model, loader, name):
    model.eval()
    for imgs, labels in loader:
        inputs = imgs.view(imgs.size(0), -1) if name=='FFN' else imgs
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        correct = (preds==labels)

        # first correct
        idx = correct.nonzero(as_tuple=False)[0].item()
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(imgs[idx].squeeze(), cmap='gray')
        axes[0].set_title(f'{name} Correct: pred={preds[idx]}, true={labels[idx]}')
        
        # first incorrect
        idx = (~correct).nonzero(as_tuple=False)[0].item()
        axes[1].imshow(imgs[idx].squeeze(), cmap='gray')
        axes[1].set_title(f'{name} Wrong: pred={preds[idx]}, true={labels[idx]}')
        plt.savefig(f'{name.lower()}_examples.png'); plt.close()
        break

plot_example(feedforward_net, testloader, 'FFN')
plot_example(conv_net,        testloader, 'CNN')

# outputs the parameter counts
def count_params(m): return sum(p.numel() for p in m.parameters())
print('FFN params:', count_params(feedforward_net))
print('CNN params:', count_params(conv_net))


y_true_ffn, y_pred_ffn = [], []
y_true_cnn, y_pred_cnn = [], []
with torch.no_grad():
    for imgs, labels in testloader:
        out = feedforward_net(imgs.view(imgs.size(0), -1))
        y_true_ffn.extend(labels.numpy())
        y_pred_ffn.extend(out.argmax(dim=1).numpy())
        out = conv_net(imgs)
        y_true_cnn.extend(labels.numpy())
        y_pred_cnn.extend(out.argmax(dim=1).numpy())
ConfusionMatrixDisplay.from_predictions(y_true_ffn, y_pred_ffn)
plt.title('FFN Confusion Matrix'); plt.savefig('ffn_confusion.png'); plt.close()
ConfusionMatrixDisplay.from_predictions(y_true_cnn, y_pred_cnn)
plt.title('CNN Confusion Matrix'); plt.savefig('cnn_confusion.png'); plt.close()


'''
PART 8:
Compare the performance and characteristics of FFN and CNN models.
'''

'''
In terms of accuracy and parameter efficiency, the convolutional network 
preformed better than the feedforward network. The FFN took about 2.5 minutes 
to train for 15 epochs and had a 86.8% test accuracy. The CNN took about
3.33 minutes to train for 10 epoches and got a 91.5% test accuracy. Because 
the CNN's local receptive fields and weight sharing allowed it to learn
hierarchical edges and texture features, it was far more effective on 
image data compared to FFN. And the CNN has about 422k parameters while the
FFN had 204k parameters. The FFN had slower convergence and worse accuracy 
because it has fully-connected layers that have to learn the global pixel 
relationship directly.
'''