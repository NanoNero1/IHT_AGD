from IHT_AGD.experimentScaffolding.chooseOptimizer import chooseOptimizer
from IHT_AGD.experimentScaffolding.chooseOptimizer import fixedChooseOptimizer
from IHT_AGD.architectures.convNets import MNIST_convNet
import torch
from IHT_AGD.modelTrainTest.trainLoop import train

""" Desc these functions actually run the experiments and capture the model references"""

def runOneExperiment(setup=None,trialNumber=None,datasetChoice="MNIST",**kwargs):

  match datasetChoice:
    case "MNIST":
      model = MNIST_convNet().to(kwargs['device'])
    case "CIFAR":
      abort()
      model = CIFAR_convNet().to(kwargs['device'])

  #optimizer = chooseOptimizer(setup,model,trialNumber,device=kwargs['device'])
  optimizer = fixedChooseOptimizer(setup,model,**(kwargs | trialNumber))
  
  ########################################
  ######################################## TO-DO: make sure the optimizer has kwargs parsed! 
  ######################################## so like test loader is also given to it
  optimizer.loggingInterval = 1

  # This implementation uses a Learning Rate Scheduler
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)
  for epoch in range(1, setup["epochs"] + 1):

    #print(optimizer.methodName)

    # Call to run one epoch of training
    train([],model, kwargs['device'], kwargs['train_loader'], optimizer, epoch,trialNumber,run=kwargs['run'])

    scheduler.step()
  return model

def runMainExperiment(setups,epochs=5,trialNumber=0,**kwargs):

  print(kwargs)
  defaults = {"epochs":epochs}

  # Combines the default setup with some added parameters
  setups = [defaults | setup for setup in setups]

  print(setups) # It's nice to see what parameters are actually passed
  all_models = [[] for i in range((len(setups)))]
  for idx in range(len(setups)):
    print(trialNumber)
    all_models[idx] = runOneExperiment(setups[idx],trialNumber=trialNumber,**kwargs)
  return all_models

def runPipeline(setups,datasetChoice="MNIST",epochs=1,trials=1,**kwargs):

  for trial in range(trials):
    runMainExperiment(setups,epochs=epochs,trialNumber=trial,**kwargs)

