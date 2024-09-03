import torch
from IHT_AGD.optimizers.vanillaSGD import vanillaSGD
import torch.nn.functional as F
from torch import Tensor, nan, inf


###############################################################################################################################################################
# ---------------------------------------------------- VANILLA-AGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################

class vanillaAGD(vanillaSGD):
  def __init__(self,params,**kwargs):
    super().__init__(params,**kwargs)

    # Objective Function Property Variables
    self.alpha = self.beta / self.kappa
    self.sqKappa = pow(self.kappa,0.5)
    self.loss_zt = 0.0
    self.device = kwargs['device']


    for p in self.paramsIter():
      state = self.state[p]

      # CHECK: is is ok to access it like this?
      state['zt'] = torch.zeros_like((p.to(self.device)))
      state['xt'] = p.data.detach().clone()
      state['zt_oldGrad'] = torch.zeros_like((p.to(self.device)))

    self.methodName = "vanilla_AGD"

  def step(self):
    print(f"speed iteration {self.iteration}")
    self.logging()

    # Safe floats check
    self.checkForNAN()
    self.checkForINF()

    self.updateWeights()
    self.iteration += 1

  ##############################################################################

  def updateWeights(self):
    print("AGD updateWeights")
    # Update z_t the according to the AGD equation
    with torch.no_grad():
      for p in self.paramsIter():

        state = self.state[p]

        #First Get z_t+
        state['zt'] = (state['zt'] - (state['zt_oldGrad'] / self.beta) )

        # And then we do the actual update, NOTE: zt is actually z_t+ right now
        state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * state['zt'] + (1.0 / (self.sqKappa + 1.0)) * state['xt']

    # careful, this changes the parameters for the model
    self.getNewGrad('zt')

    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]
        state['zt_oldGrad'] = p.grad.clone().detach()

        # NOTE: p.grad is now the gradient at zt
        p.data = state['xt'] - (1.0 / pow(self.alpha*self.beta , 0.5)) * p.grad

    # We need to keep a separate storage of xt because we replace the actual network parameters
    self.copyXT()

  ##########################################

  # Allows us to compute the gradient from within the optimizer
  def getNewGrad(self,iterate):
    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]
        p.data = state[iterate].clone().detach()

    self.zero_grad()
    data,target = self.currentDataBatch

    newOutput = self.model(data)
    loss = F.nll_loss(newOutput, target)
    loss.backward()

    if iterate == "zt":
      self.loss_zt = float(loss.clone().detach())

  # Copies xt for safekeeping when we swap the parameters
  def copyXT(self):
    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]
        state['xt'] = p.data.clone().detach()
