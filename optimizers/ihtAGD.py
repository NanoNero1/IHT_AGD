import torch
from IHT_AGD.optimizers.vanillaAGD import vanillaAGD
from IHT_AGD.optimizers.ihtSGD import ihtSGD

###############################################################################################################################################################
# ---------------------------------------------------- IHT-AGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################

class ihtAGD(vanillaAGD,ihtSGD):
  def __init__(self,params,**kwargs):
    super().__init__(params,**kwargs)
    self.methodName = "iht_AGD"
    self.alpha = self.beta / self.kappa

  def step(self):
    print(f"speed iteration {self.iteration}")
    self.logging()
    self.compressOrDecompress()
    self.iteration += 1

  #def returnSparse(self):

  # I checked this, it seems to work
  def truncateAndFreeze(self):
    self.updateWeightsSparse()
    print('this should work')
    # define zt



    # Truncate xt
    self.sparsify(iterate='xt')


    # Freeze xt
    self.freeze(iterate='xt')

    pass

  """
  def compressedStep(self):
    print('compressed step')
    self.updateWeightsSparse()
    self.refreeze()
  """

  def updateWeightsSparse(self):
    print("AGD updateWeights (with sparse z_t and x_t)")
    # Update z_t the according to the AGD equation in the note
    with torch.no_grad():
      for p in self.paramsIter():
        state = self.state[p]
        #First Get z_t+
        state['zt'] = (state['zt'] - (state['zt_oldGrad'] / self.beta) )
        #self.sparsify(iterate='zt')


        # And then we do the actual update, NOTE: zt is actually z_t+ right now
        state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * state['zt'] + (1.0 / (self.sqKappa + 1.0)) * state['xt']

        #Find the new z_t
        #state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * (state['zt'] - (state['zt_oldGrad'] / self.beta) ) + (1.0 / (self.sqKappa + 1.0)) * state['xt']

    # CAREFUL! this changes the parameters for the model
    self.getNewGrad('zt')

    with torch.no_grad():
      for p in self.paramsIter():
        # CHECK: Is it still the same state?
        state = self.state[p]
        state['zt_oldGrad'] = p.grad.clone().detach()

        # NOTE: p.grad is now the gradient at zt
        p.data = state['xt'] - (1.0 / pow(self.alpha*self.beta , 0.5)) * p.grad


    self.sparsify()

    for p in self.paramsIter():
        state = self.state[p]
        # we also update xt - NOTE: we do this because we have to put z_t in for the loss computation,
        # so it's nice to store x_t
        state['xt'] = p.data.detach().clone()
