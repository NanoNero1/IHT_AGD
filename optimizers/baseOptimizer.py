import torch
from torch.optim.optimizer import Optimizer, required
import abc

""" Desc: The base optimizer class inherits from the PyTorch Optimizer class. The idea of this design is that we can make brand new
optimizers without having to re-write a lot of the boilerplate code. It also makes it really easy to combine different methods,
e.g. Iterative Hard Thresholding with Accelerated Gradient Descent is a combination of IHT-SGD and vanilla-AGD.
"""

class myOptimizer(Optimizer):
  def __init__(self,params,test_loader=None,lr=1.0,sparse=False,demoValue=100,**kwargs):

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

  """ Desc: logs various useful parameters -- is expensive to run on every iteration"""
  def logging(self):
    # NOTE: iteration is checked here to make the code cleaner
    if self.iteration % self.loggingInterval == 0:
      # Functions to execute
      for function in dir(self):
        if function not in self.functionsToHelpTrack:
          continue
        eval("self." + function + "()")

      # Variables to Log
      for variable in dir(self):
        if variable not in self.variablesToTrack:
          continue
        else:
          self.run[f"trials/{self.trialNumber}/{self.methodName}/{variable}"].append(eval("self."+variable))

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


  """ Desc: when we add extra kwargs that aren't recognized, we add them to our variables by default"""
  def dealWithKwargs(self,keywordArgs):
    for key, value in keywordArgs.items():
      setattr(self, key, value)

  """ Desc: use it like 'for i in paramsIterator():' """

  # CHECK: that this efficient, does this just pass references or copy the entire thing?
  # THIS WORKS!
  def paramsIter(self):
    for group in self.param_groups:
      for p in group['params']:
        yield p


  ### These methods are mandatory to be overridden after inheritance
  ### CHECK: there should be an error if they are not implemented
  """ Desc: the main function that the optimizer gets called on every iteration """
  @abc.abstractmethod
  def step(self,getNewGrad):
    return None

  """ Desc: what optimization scheme it uses to update weights, e.g. Accelerated Gradient Descent """
  @abc.abstractmethod
  def updateWeights(self):
    pass