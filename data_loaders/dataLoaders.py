# SOURCE: https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/MNIST.ipynb#scrollTo=mwpfASvc_v7J
# SOURCE: https://github.com/pytorch/examples/blob/main/mnist/main.py


""" Desc: The MNIST Dataset, sourced from PyTorch, this dataset is for classification of handwritten digits
  Size: 60,000 examples,
  Input: 28x28 pixels,
  Target: digit 0-9,
"""
datasetChoice = "MNIST"

match datasetChoice:
  case "MNIST":

    # Data Collection and Normalizing so that it's suitable for input to the neural network
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    dataset1 = datasets.MNIST('../data', train=True, download=True,transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,transform=transform)

    # setup information?
  case "CIFAR":
    abort()
    ############################ OOOOOOOOOOOOOOOOH!!!!! ACTUALLY I NEED TO NORMALIZE IT< THE TRANSFORM IS WRONG!!!!
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False,transform=transform)


# Data Loaders : These also allow us to test performance ad-hoc
train_loader = torch.utils.data.DataLoader(dataset1,batch_size=1000,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2,batch_size=1000,shuffle=True)