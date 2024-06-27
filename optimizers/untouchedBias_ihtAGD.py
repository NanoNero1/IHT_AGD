import torch
from IHT_AGD.optimizers.ihtAGD import ihtAGD
import numpy as np

class untouchedBias_ihtAGD(ihtAGD):
  def __init__(self,params,**kwargs):
    super().__init__(params,**kwargs)
    self.methodName = "iht_AGD"

  def step(self):
    for p in self.paramsIter():
      print(p.shape)
      
      if len(p.shape) == 1:
       
        abort()
        continue



  def sparsify(self):
    print('The Bias Nodes should not be sparsified')

  def sparsify(self,iterate=None):
    
    #
    cutoff = self.getCutOff(iterate=iterate)

    for p in self.paramsIter():
      print(p.shape)
      
      if len(p.shape) == 1:
       
        abort()
        continue



      state = self.state[p]
      if iterate == None:
        print("!!!!!!!!!!! this should sparsify the params")
        p.data[torch.abs(p) <= cutoff] = 0.0
      else:
        # NOTE: torch.abs(p) is wrong, maybe that's the bug
        (state[iterate])[torch.abs(state[iterate]) <= cutoff] = 0.0




  

# TO - DO:
# for the bias nodes to be untouched, they must 
# a) not be sparsified - sparsify()
# a.a) the cutoff has to be changed too ... technically
# b) not be re-frozen? - refreeze()
# I think thats it?
# oh and look at trackSparsityBias

