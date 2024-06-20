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


  def truncateAndFreeze(self):
    # define zt


    # Truncate xt
    self.sparsify(iterate='xt')


    # Freeze xt
    self.freeze(iterate='xt')

    pass

  def 
