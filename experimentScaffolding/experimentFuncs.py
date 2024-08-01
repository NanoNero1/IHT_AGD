from IHT_AGD.experimentScaffolding.chooseOptimizer import chooseOptimizer
from IHT_AGD.experimentScaffolding.chooseOptimizer import fixedChooseOptimizer
from IHT_AGD.architectures.convNets import MNIST_convNet
from IHT_AGD.architectures.convNets import basicNeuralNet
import torch
from IHT_AGD.modelTrainTest.trainLoop import train
from torchvision.models import resnet50

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
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)
  for epoch in range(1, setup["epochs"] + 1):

    #print(optimizer.methodName)

    # Call to run one epoch of training
    train([],model, kwargs['device'], kwargs['train_loader'], optimizer, epoch,trialNumber,run=kwargs['run'])

    scheduler.step()
  return model

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

