import torch
from IHT_AGD.optimizers.vanillaSGD import vanillaSGD
import numpy as np

###############################################################################################################################################################
# ---------------------------------------------------- IHT-SGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################
class ihtSGD(vanillaSGD):
  def __init__(self, params, sparsifyInterval=10,**kwargs):

    super().__init__(params,**kwargs)
    self.sparsifyInterval = sparsifyInterval

    # Compression, Decompression and Freezing Variables
    self.phaseLength = 80
    self.compressionRatio = 0.5
    self.freezingRatio = 0.2
    self.warmupLength = 3000

    # State Initialization
    for p in self.paramsIter():
      state = self.state[p]
      state['xt_frozen'] = torch.ones_like(p)

    self.methodName = "iht_SGD"

    # Sparsity Tracking
    self.trackSparsity = 0
    self.trackSparsityLinear = 0
    self.trackSparsityBias = 0

  @torch.no_grad()
  def step(self):
    print(f"iteration {self.iteration}")

    self.logging()
    self.easyPrintParams()
    
    # Main function for the optimizer
    self.compressOrDecompress()


    self.easyPrintParams()
    self.iteration +=1

  ########################################################

  def compressOrDecompress(self):
    howFarAlong = (self.iteration - self.warmupLength) % self.phaseLength
    print(f"HowFarAlong: {howFarAlong} / {self.phaseLength}")
    print(f"Iteration: {self.iteration}")

    if self.iteration < self.warmupLength:
      ## WARMUP -- PHASE 0
      self.warmup()
    elif howFarAlong == 0:
      ## FREEZING WEIGHTS -- PHASE 1
      self.truncateAndFreeze()
    elif howFarAlong <= self.phaseLength * self.compressionRatio:
      ## COMPRESSED -- PHASE 2
      self.compressedStep()
    elif howFarAlong > self.phaseLength * self.compressionRatio:
      ## DECOMPRESS -- PHASE 3
      self.decompressed()
    else:
      print("Error, iteration logic is incorrect")

  ### PHASES ###
  def warmup(self):
    print('warmup')
    self.updateWeights()

  def truncateAndFreeze(self):
    print('truncateAndFreeze')
    self.updateWeights()
    self.sparsify()
    self.freeze()

  def compressedStep(self):
    print('compressed step')
    self.updateWeights()
    self.refreeze()

  def decompressed(self):
    print('decompressed')
    self.updateWeights()

  ### UTILITY FUNCTIONS ######################################################################################

  def getCutOff(self,sparsity=None,iterate=None):
    if sparsity == None:
      sparsity = self.sparsity

    concatWeights = torch.zeros((1)).to(self.device)
    for p in self.paramsIter():
      if iterate == None:
        layer = p.data
      else:
        state = self.state[p]
        layer = state[iterate]

      # CHECK: Make sure this flattening doesn't affect the original layer
      flatWeights = torch.flatten(layer)
      concatWeights = torch.cat((concatWeights,flatWeights),0)
    concatWeights = concatWeights[1:] # Removing first zero

    # Converting the sparsity factor into an integer of respective size
    topK = int(len(concatWeights)*(1-sparsity))

    # All the top-k values are sorted in order, we take the last one as the cutoff
    vals, bestI = torch.topk(torch.abs(concatWeights),topK,dim=0)
    cutoff = vals[-1]

    return cutoff
  
  # Reduce an interate to a percentage of non-zero weights
  def sparsify(self,iterate=None):
    cutoff = self.getCutOff(iterate=iterate)

    for p in self.paramsIter():
      state = self.state[p]
      if iterate == None:
        p.data[torch.abs(p) <= cutoff] = 0.0
      else:
        # NOTE: torch.abs(p) is wrong, maybe that's the bug
        (state[iterate])[torch.abs(state[iterate]) <= cutoff] = 0.0
  
  # NOTE: Refreeze is not only for the PARAMS!
  # Makes the iterate match the support it had on the last mask
  def refreeze(self,iterate=None):
    for p in self.paramsIter():
      state = self.state[p]
      if iterate == None:
        p.data *= state['xt_frozen']
      else:
        state[iterate] *= state[f"{iterate}_frozen"]

  # Saves a mask for where an iterate is non-zero
  def freeze(self,iterate=None):
    cutOff = self.getCutOff(iterate=iterate)

    for p in self.paramsIter():
      state = self.state[p]
      if iterate == None:
        layer = p.data
        state['xt_frozen'] = (torch.abs(layer) > 0).type(torch.uint8)
      else:
        layer = state[iterate]
        state[f"{iterate}_frozen"] = (torch.abs(layer) > 0).type(torch.uint8)

  # For logging how sparse the network and specific layers are
  def trackingSparsity(self):
    concatWeights = torch.zeros((1)).to(self.device)
    concatLinear = torch.zeros((1)).to(self.device)
    concatBias = torch.zeros((1)).to(self.device)
    for layer in self.paramsIter():
      inb = torch.abs(layer.data)

      # Total Weights
      flatTotal = torch.flatten(layer.data)
      concatWeights = torch.cat((concatWeights,flatTotal),0)

      if len(layer.data.shape) < 2:
        # Bias Layers
        concatBias = torch.cat((concatBias,layer.data),0)
      else:
        # Linear Layers
        flatLinear = torch.flatten(layer.data)
        concatLinear = torch.cat((concatLinear,flatLinear),0)

      # Sparsity for this layer
      layerSparsity = torch.mean( (torch.abs(layer.data) > 0).type(torch.float) )
      layerName = f"layerSize{torch.numel(layer)}"

      # Track the per-layer sparsity with size
      self.run[f"trials/{self.trialNumber}/{self.setupID}/{layerName}"].append(layerSparsity)

    # Removing the First Zero
    concatBias = concatBias[1:]
    concatWeights = concatWeights[1:]
    concatLinear = concatLinear[1:] 

    # Final sparsity calculations
    nonZeroWeights = (torch.abs(concatWeights) > 0).type(torch.float)
    nonZeroBias = (torch.abs(concatBias) > 0).type(torch.float)
    nonZeroLinear = (torch.abs(concatLinear) > 0).type(torch.float)

    self.trackSparsity = torch.mean(nonZeroWeights)
    self.trackSparsityBias = torch.mean(nonZeroBias)
    self.trackSparsityLinear = torch.mean(nonZeroLinear)
