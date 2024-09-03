import torch
from IHT_AGD.optimizers.ihtAGD import ihtAGD
import numpy as np

###############################################################################################################################################################
# ---------------------------------------------------- IHT-AGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################

class clipGradientIHTAGD(ihtAGD):
  def __init__(self,params,**kwargs):
    super().__init__(params,**kwargs)
    self.methodName = "clipGradientIHTAGD"
    self.alpha = self.beta / self.kappa

  def clipGradients(self,clipAmt=0.01):
    torch.nn.utils.clip_grad_value_(self.param_groups[0]['params'],clip_value=clipAmt)
    pass

  def updateWeightsTwo(self):

    with torch.no_grad():
      for p in self.paramsIter():

        state = self.state[p]


        #First Get z_t+
        state['zt'] = (state['zt'] - (state['zt_oldGrad'] / self.beta) )

        # And then we do the actual update, NOTE: zt is actually z_t+ right now
        state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * state['zt'] + (1.0 / (self.sqKappa + 1.0)) * state['xt']

    # NOTE: careful, this changes the parameters for the model
    self.getNewGrad('zt')
    self.clipGradients()

    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]
        state['zt_oldGrad'] = p.grad.clone().detach()

        # NOTE: p.grad is now the gradient at zt
        p.data = state['xt'] - (1.0 / pow(self.alpha*self.beta , 0.5)) * p.grad

    # We need to keep a separate storage of xt because we replace the actual network parameters
    self.copyXT()