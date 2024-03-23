import torch
import torch.nn as nn
import torch.functional as F
import gc
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler




# Create a basic convnet model taking as input a tensor of size B x 3 x 224 x 224
# and containing the following layers:
# - a 2D convolution with output dim = 64, kernel size = 7, stride = 2, padding = 3
# - a batch norm
# - a ReLU
# - a 2D max pooling with kernel size = 3, stride = 2 and padding = 1
# - a 2D avg pooling with input dim you should guess using .shape and stride = 1
# - a flatten layer
# - a linear layer with input dim you should guess using .shape and output dim = 10
#
# Leave all bias options unchanged (i.e. true by default)
#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 =nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size= 56, stride=1)
        self.flatten = nn.Flatten()
        self.linear= nn.Linear(64, 10) 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


#params

torch.set_printoptions(precision=8)
torch.manual_seed(123) # do not remove this line
X = torch.rand((3, 3, 224, 224))
torch.manual_seed(123) # do not remove this line
m = Net()
batch_size = 64



# Downloading 
# compute mean and variance of the CIFAR10 dataset - do not modify this code
from torchvision import datasets
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.CIFAR10(root='files/', train=True, download=True, transform=train_transform)
print(train_set.data.mean(axis=(0,1,2))/255)
print(train_set.data.std(axis=(0,1,2))/255)


# Run this once to load the train and test data straight into a dataloader class
# that will provide the batches
def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
  
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
    ])

    if test:
        dataset = datasets.CIFAR10(
          root=data_dir, train=False,
          download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


# CIFAR10 dataset 
train_loader, valid_loader = data_loader(data_dir='./files',
                                         batch_size=64)

test_loader = data_loader(data_dir='./files',
                              batch_size=batch_size,
                              test=True)
                              
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_classes = 10


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Build a PlainNet architecture as described in the paper: https://arxiv.org/pdf/1512.03385.pdf
#
# The module takes as input the number of convolutions at each layer (e.g. [6, 7, 11, 5] for a PlainNet-34)
#
# Use a for loop for layers 2-5, e.g.:
#
# layer2 = []
# for _ in range(n):
#     layer2 += [nn.Conv2d(...), nn.BatchNorm(...), nn.ReLU()]
# self.layer2 = nn.Sequential (layer2)
#

class PlainNet(nn.Module):
    def __init__(self, output_dim, layers):
        super().__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), 
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=7, stride=2, padding=3))
        layer2 = []
        for _ in range(layers[0]):
            layer2 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
                       nn.BatchNorm2d(64), 
                       nn.ReLU()]

        self.layer2 = nn.Sequential(*layer2)
        # same for layers 3, 4 and 5
        # Layer 3 , 3*3conv(128,/2) + batchnorm + relu
        layer3=[nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()]
        for _ in range(layers[0]):
            layer3 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
                       nn.BatchNorm2d(128), 
                       nn.ReLU()
                       ]
        self.layer3 = nn.Sequential(*layer3)
        # layer 4, 3*3conv(256,/2) + batchnorm + relu   
        layer4=[nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
                       nn.BatchNorm2d(256), 
                       nn.ReLU()]
        for _ in range(layers[0]):
            layer4 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
                       nn.BatchNorm2d(256), 
                       nn.ReLU()
                       
                       ]
        self.layer4 = nn.Sequential(*layer4)
        # layer 5, 3*3conv(512,/2) + batchnorm + relu
        layer5=[nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
                       nn.BatchNorm2d(512), 
                       nn.ReLU()]
        for _ in range(layers[0]):
            layer5 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
                       nn.BatchNorm2d(512), 
                       nn.ReLU()
                       ]
        self.layer5 = nn.Sequential(*layer5)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Sequential(nn.Linear(512, output_dim),
                                nn.ReLU(),
                                nn.Softmax())
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x= self.avgpool(x)
        x=x.reshape(x.size(0), -1)
        x = self.fc(x)
        # use x = x.reshape(x.size(0), -1) or a nn.Flatten() layer  before the linear layer
        return x

model_plain_18 = PlainNet(num_classes, layers=[4, 3, 3, 3]).to(device) # plain-18
print(f'plain-18: {count_parameters(model_plain_18)}')

# training params
# these parameters should work to train your models
num_epochs = 20
batch_size = 16
learning_rate = 0.01

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_plain_18.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)

# Traininf Function
def train(model):

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        print ('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, loss.item()))
                
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
        
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

if __name__ == '__main__':
    train(model_plain_18)