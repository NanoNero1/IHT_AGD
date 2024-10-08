# SOURCE: https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/MNIST.ipynb#scrollTo=mwpfASvc_v7J
# SOURCE: https://github.com/pytorch/examples/blob/main/mnist/main.py

from torchvision import datasets, transforms
import torch

from pathlib import Path


""" Desc: The MNIST Dataset, sourced from PyTorch, this dataset is for classification of handwritten digits
  Size: 60,000 examples,
  Input: 28x28 pixels,
  Target: digit 0-9,
"""
datasetChoice = "CIFAR"

match datasetChoice:
  case "MNIST":

    # Data Collection and Normalizing so that it's suitable for input to the neural network
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    # Checking if data is already downloaded
    p = Path('../data/MNIST/raw/train-images-idx3-ubyte')
    if p.exists():
      pass
    else:
      raise("Raise Error - data not found")


    dataset1 = datasets.MNIST('../data', train=True, download=True,transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,download=True,transform=transform)

  case "CIFAR":
    # Checking if data is already downloaded
    p = Path('../CIFAR/cifar-10-batches-py/')
    if p.exists():
      pass
    else:
      # To make sure we're not downloading the data every time
      raise("Raise Error - data not found")

    transform = transforms.Compose([transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Select training_set and testing_set
    dataset1 = datasets.CIFAR10("../CIFAR", train= True,download=True,transform = transform)

    dataset2 = datasets.CIFAR10("../CIFAR", train= False,download=True,transform = transform)


# Data Loaders : These also allow us to test performance ad-hoc
train_loader = torch.utils.data.DataLoader(dataset1,batch_size=2048,shuffle=False,drop_last=True,num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset2,batch_size=2048,shuffle=False,drop_last=True,num_workers=2)