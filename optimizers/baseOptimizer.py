#SOURCE: https://www.geeksforgeeks.org/custom-optimizers-in-pytorch/

#SOURCE: https://github.com/azotlichid/adaptive-opt

import torch
from torch.optim.optimizer import Optimizer, required
import abc

""" Desc: The base optimizer class inherits from the PyTorch Optimizer class. The idea of this design is that we can make brand new
optimizers without having to re-write a lot of the boilerplate code. It also makes it really easy to combine different methods,
e.g. Iterative Hard Thresholding with Accelerated Gradient Descent is a combination of IHT-SGD and vanilla-AGD.
"""

class myOptimizer(Optimizer):
  def __init__(self,params,lr=1.0,sparse=False,demoValue=100,**kwargs):

    # A dummy value to see if inheritance works properly
    self.demoValue = demoValue

    self.lr = lr
    defaults = dict(lr=lr,sparse=sparse)
    super().__init__(params,defaults) # For pytorch's Optimizer class

    self.iteration = 0
    self.trialNumber = None
    self.testAccuracy = None

    self.loggingInterval = 100

    # Varaibles specific to certain classes
    self.currentDataBatch = None

    self.dealWithKwargs(kwargs)

    self.methodName="base_optimizer"
    self.setupID = None

  """ Desc: logs various useful parameters -- is expensive to run on every iteration"""
  def logging(self):

    print(f"==================================================================================== ITERATION: {self.iteration}")

    if self.iteration % self.loggingInterval == 0:
      # Functions to execute
      for function in dir(self):
        if function not in self.functionsToHelpTrack:
          continue
        elif function in self.expensiveFunctions:
          # Do not compute expensive functions on every step
          if self.iteration % 50 == 0:
            pass
          else:
            continue
        eval("self." + function + "()")

      # Variables to Log
      for variable in dir(self):
        if variable not in self.variablesToTrack:
          continue
        elif variable in self.expensiveVariables:
          # Do not compute expensive variables on every step
          if self.iteration % 50 == 0:
            pass
          else:
            continue
        
        self.run[f"trials/{self.trialNumber}/{self.setupID}/{variable}"].append(eval("self."+variable))

  """ Desc: an internal function that calculates the test loss on the logging step #NOTE: expensive to compute, try to make the logging interval high"""
  def getTestAccuracy(self):
    self.model.eval()
    correct = 0
    # The testing accuracy is taken over the entire dataset
    with torch.no_grad():
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(self.test_loader.dataset)
    self.testAccuracy = accuracy

  def easyPrintParams(self):
    for pInd,p in enumerate(self.paramsIter()):
      if pInd != 3:
        continue

  """ Desc: when we add extra kwargs that aren't recognized, we add them to our variables by default"""
  def dealWithKwargs(self,keywordArgs):
    for key, value in keywordArgs.items():
      setattr(self, key, value)

  """ Desc: use it like 'for i in paramsIterator():' """
  def paramsIter(self):
    for group in self.param_groups:
      for p in group['params']:
        yield p


  # Functions for checking Float Safety
  def checkForNAN(self):
    for p in self.paramsIter():
      #torch.isnan(p.data).any() or
      if  torch.isnan(p.grad).any():
        print('nan grad detected')
        abort()
      if torch.isnan(p.data).any():
        print('nan weight detected')
        abort()

  def checkForINF(self):
    for p in self.paramsIter():
      #torch.isnan(p.data).any() or
      if  torch.isinf(p.grad).any():
        print('inf grad detected')
        abort()
      if torch.isinf(p.data).any():
        print('inf weight detected')
        abort()





  ### These methods are mandatory to be overridden after inheritance
  """ Desc: the main function that the optimizer gets called on every iteration """
  @abc.abstractmethod
  def step(self,getNewGrad):
    return None

  """ Desc: what optimization scheme it uses to update weights, e.g. Accelerated Gradient Descent """
  @abc.abstractmethod
  def updateWeights(self):
    pass