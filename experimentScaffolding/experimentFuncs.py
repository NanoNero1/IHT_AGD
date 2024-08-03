from IHT_AGD.experimentScaffolding.chooseOptimizer import chooseOptimizer
from IHT_AGD.experimentScaffolding.chooseOptimizer import fixedChooseOptimizer
from IHT_AGD.architectures.convNets import MNIST_convNet
from IHT_AGD.architectures.convNets import basicNeuralNet
import torch
from IHT_AGD.modelTrainTest.trainLoop import train
from torchvision.models import resnet50
import torch.nn.functional as F
import torch.nn as nn

import json 

""" Desc these functions actually run the experiments and capture the model references"""

def runOneExperiment(setup=None,trialNumber=None,datasetChoice="MNIST",**kwargs):
  match datasetChoice:
    case "MNIST":
      model = MNIST_convNet().to(kwargs['device'])
      #model = basicNeuralNet().to(kwargs['device'])
    case "CIFAR":
      model = resnet50().to(kwargs['device'])
      print('the CIFAR MODEL IS LOADED')
    case "IMAGENET":
      model = resnet50().to(kwargs['device'])
      print("THEEEEEEEEEEEEEE MODEL IS NOW RESNET!")

  #optimizer = chooseOptimizer(setup,model,trialNumber,device=kwargs['device'])
  optimizer = fixedChooseOptimizer(setup,model,**(kwargs | {'trialNumber':trialNumber}))
  
  optimizer.loggingInterval = 1

  # This implementation uses a Learning Rate Scheduler
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)

  #We don't need more than these arguments
  #train_net([],model, kwargs['device'], kwargs['train_loader'], optimizer, epochs=setup["epochs"],run=kwargs['run'])
  
  model.train()
  for epoch in range(1, setup["epochs"] + 1):

    #print(optimizer.methodName)

    print(f"!!!!!!!!!!!! THIS IS EPOCH: {epoch}")

    # Call to run one epoch of training
    #train([],model, kwargs['device'], kwargs['train_loader'], optimizer, epoch,trialNumber,run=kwargs['run'])
    

    scheduler.step()
  
  return model

# Trains the network
#def train_net(epochs, path_name, net, optimizer):
def train_net(args, model, device, train_loader, optimizer=None, epochs=1,run=None):
    """ Train the network """
    print(optimizer)
    #writer = SummaryWriter(path_name)
    n_iter = 0

    # Dump info on the network before running any training step
    #tb_dump(0, net, writer)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):  # Loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # NOTE: maybe this is the thing that causes a Neptune NoneError
            print(f"print loss:{loss}")
            ###LOG### Loss
            #abort()
            run[f"trials/{optimizer.trialNumber}/{optimizer.setupID}/loss"].append(loss)

            loss.backward()

            def getNewGrad(parameters):
              model.params = parameters
              newOutput = model(data)
              loss = criterion(newOutput, labels)
              loss.backward()

            optimizer.currentDataBatch = (data,labels)
            optimizer.step()




    print('Finished Training')

def runMainExperiment(setups,epochs=5,trialNumber=0,datasetChoice="MNIST",**kwargs):

  print(kwargs)
  defaults = {"epochs":epochs}

  # Combines the default setup with some added parameters
  setups = [defaults | setup for setup in setups]

  print(setups) # It's nice to see what parameters are actually passed
  all_models = [[] for i in range((len(setups)))]
  for idx in range(len(setups)):
    print(trialNumber)
    all_models[idx] = runOneExperiment(setups[idx],trialNumber=trialNumber,datasetChoice=datasetChoice,**kwargs)
  return all_models

def runPipeline(setups,datasetChoice="MNIST",epochs=1,trials=1,**kwargs):


  
  #print('jasper')
  #print('not 50 minutes ago')
  #abort()
  #Logging Metadata to Neptune
  run = kwargs['run']

  # NOTE: why doesn't this print??
  print('this should print')
  # CHECK: can I send dictionaries directly?
  run["activation"] = "ReLU"

  # Converting to JSON
  with open("setups.json", "w") as outfile: 
    json.dump(setups, outfile)
  
  # NOTE: Neptune does not allow to send dictionaries directly,
  # CHECK: that it doesn't get sent / look into the wrong directory
  run["setupDict"].upload("setups.json")

  for trial in range(trials):
    runMainExperiment(setups,epochs=epochs,trialNumber=trial,datasetChoice=datasetChoice,**kwargs)

