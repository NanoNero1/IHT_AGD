from IHT_AGD.experimentScaffolding.chooseOptimizer import chooseOptimizer
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

  optimizer = chooseOptimizer(setup,model,trialNumber,device=kwargs['device'])
  #optimizer = eval(setup["scheme"])(setup,model,trialNumber)

  # Maybe the idea is that the model can change, but the test loader is global??
  optimizer.test_loader = kwargs['test_loader']

  ########################################
  ######################################## TO-DO: make sure the optimizer has kwargs parsed! 
  ######################################## so like test loader is also given to it
  #for () in kwargs
  # REFURBISH ME!!!!!!!!
  for key, value in kwargs.items():
    if key not in ['device','functionsToHelpTrack','variablesToTrack','run']:
      continue
    setattr(optimizer, key, value)
  optimizer.loggingInterval = 1

  # This implementation uses a Learning Rate Scheduler
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)
  for epoch in range(1, setup["epochs"] + 1):

    #print(optimizer.methodName)

    # Call to run one epoch of training
    train([],model, kwargs['device'], kwargs['train_loader'], optimizer, epoch,trialNumber,run=kwargs['run'])

    scheduler.step()
  return model

