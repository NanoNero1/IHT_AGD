#SOURCE: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

#SOURCE: https://colab.research.google.com/github/omarsar/pytorch_notebooks/blob/master/pytorch_quick_start.ipynb#scrollTo=3_0Vjq2RHlph


import torch.nn as nn
import torch.nn.functional as F
import torch


### INACTIVE - But we might need it for tasks where we don't use CNNs
class basicNeuralNet(nn.Module):
  def __init__(self,inputDim,outputDim):
    super(basicNeuralNet, self).__init__()
    # Hyper-Parameters
    hiddenSize_1 = 200
    hiddenSize_2 = 50

    # Pytorch Layers
    self.flatten = nn.Flatten()
    self.hidden1 = nn.Linear(inputDim,hiddenSize_1)
    self.hidden2 = nn.Linear(hiddenSize_1,hiddenSize_2)
    self.prob = nn.Linear(hiddenSize_2,outputDim)

    # Activation Functions
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()

  # Forward Pass: outputs a vector [x] of softmax probabilities
  def forward(self,x):
    flat = self.flatten(x)
    hidden_1 = self.relu(self.hidden1(flat))
    hidden_2 = self.relu(self.hidden2(hidden_1))
    prob = self.prob(hidden_2)
    out =  F.log_softmax(prob, dim=1)
    return out

""" Desc: A basic Convolutional Neural Network : these are particularly suitable for image-based problems
"""
class MNIST_convNet(nn.Module):
    def __init__(self):
        super(MNIST_convNet, self).__init__()


        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)
        
    # Forward Pass: outputs a vector [x] of softmax probabilities
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        # Softmax so that we output probabilities (i.e. adds up to 1)
        output = F.log_softmax(x, dim=1)
        
        return output


""" Desc: A basic Convolutional Neural Network : these are particularly suitable for image-based problems
"""
class CIFAR_convNet(nn.Module):
    def __init__(self):
        super(CIFAR_convNet, self).__init__()
        
        # Convolutional layers
        #Init_channels, channels, kernel_size, padding) 
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2,2)
        
        # FC layers
        # Linear layer (64x4x4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        
        # Linear Layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        
        # Flatten the image
        x = x.view(-1, 64*4*4)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
