import torch
from IHT_AGD.optimizers.vanillaAGD import vanillaAGD
from IHT_AGD.optimizers.ihtSGD import ihtSGD

###############################################################################################################################################################
# ---------------------------------------------------- IHT-AGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################

class ihtAGD(vanillaAGD,ihtSGD):
  def __init__(self,params,sparsity=0.9,kappa=5.0,beta=50.0,**kwargs):
    super().__init__(params,**kwargs)
    self.methodName = "iht_AGD"
    self.alpha = beta / kappa
    self.beta = beta
    self.kappa = kappa
    #self.model = model

    # BUG: Because of multiple inheritance I need to add this statement below,
    # but I know how to fix this
    self.sparsity=sparsity

  def step(self):
    print(f"speed iteration {self.iteration}")
    self.logging()
    self.compressOrDecompress()
    self.iteration += 1
