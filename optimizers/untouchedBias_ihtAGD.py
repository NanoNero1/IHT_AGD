import torch
from IHT_AGD.optimizers.clipGradientIHTAGD import clipGradientIHTAGD
import numpy as np

class untouchedBias_ihtAGD(clipGradientIHTAGD):
  def __init__(self,params,**kwargs):
    super().__init__(params,**kwargs)
    self.methodName = "iht_AGD"


  def clipGradients(self,clipAmt=0.01):
    torch.nn.utils.clip_grad_value_(self.param_groups[0]['params'],clip_value=clipAmt)
    pass

  def sparsify(self,iterate=None):
    cutoff = self.getCutOff(iterate=iterate)

    for p in self.paramsIter():

      # Here we ignore the bias layers
      if len(p.shape) == 1:
        continue

      state = self.state[p]
      if iterate == None:
        p.data[torch.abs(p) <= cutoff] = 0.0
      else:
        (state[iterate])[torch.abs(state[iterate]) <= cutoff] = 0.0
  
  # NOTE: Refreeze is not only for the PARAMS!
  def refreeze(self,iterate=None):
    for p in self.paramsIter():
      if len(p.shape) == 1:
        continue

      state = self.state[p]
      if iterate == None:
        p.data *= state['xt_frozen']
      else:
        state[iterate] *= state[f"{iterate}_frozen"]

