import torch
from IHT_AGD.optimizers.vanillaSGD import vanillaSGD

###############################################################################################################################################################
# ---------------------------------------------------- IHT-SGD ------------------------------------------------------------------------------------------
###############################################################################################################################################################
class ihtSGD(vanillaSGD):
  def __init__(self, params, sparsifyInterval=10,**kwargs):

    super().__init__(params,**kwargs)
    self.sparsifyInterval = sparsifyInterval

    # Compression, Decompression and Freezing Variables
    self.phaseLength = 40
    self.compressionRatio = 0.5
    self.freezingRatio = 0.2
    self.warmupLength = 30

    # State Initialization
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = 0

        state['xt_frozen'] = torch.ones_like(p)

    self.methodName = "iht_SGD"

    # Sparsity Tracking
    self.trackSparsity = 0
    self.trackSparsityLinear = 0
    self.trackSparsityBias = 0

  @torch.no_grad()
  def step(self):
    print('FIXED IHT SGD')
    print(f"speed iteration {self.iteration}")
    self.logging()
    self.compressOrDecompress()
    self.iteration +=1

  def sparsify(self,iterate=None):
    cutoff = self.getCutOff(iterate=iterate)
    # TO-DO check if cutoff does what it says it does

    for p in self.paramsIter():
      #p.require_grad = False
      p.data[torch.abs(p) <= cutoff] = 0.0
      #p.require_grad = True

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

  ### UTILITY FUNCTIONS ###
  def refreeze(self,iterate=None):
    for p in self.paramsIter():
      state = self.state[p]
      # TO-DO: make into modular string
      #p.mul_(state['xt_frozen'])
      p.data *= state['xt_frozen']

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

    # Converting the sparsity factor into an integer of respective size
    topK = int(len(concatWeights)*(1-sparsity))

    # All the top-k values are sorted in order, we take the last one as the cutoff
    vals, bestI = torch.topk(torch.abs(concatWeights),topK,dim=0)
    cutoff = vals[-1]

    return cutoff

  def freeze(self,iterate=None):
    cutOff = self.getCutOff(iterate=iterate)

    for p in self.paramsIter():
      state = self.state[p]
      if iterate == None:
        layer = p.data
      else:
        layer = state[iterate]

      # NOTE: I CHECKED IT!
      state['xt_frozen'] = (torch.abs(layer) > 0).type(torch.uint8)

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
      layerSparsity = torch.mean( (torch.abs(concatWeights) > 0).type(torch.float) )
      layerName = f"layerSize{torch.numel(layer)}"

      # Track the per-layer sparsity with size
      #setattr(self,f"layerSize{torch.numel(layer)}")
      self.run[f"trials/{self.trialNumber}/{self.methodName}/{layerName}"].append(layerSparsity)


    # Final sparsity calculations
    nonZeroWeights = (torch.abs(concatWeights) > 0).type(torch.float)
    nonZeroBias = (torch.abs(concatBias) > 0).type(torch.float)
    nonZeroLinear = (torch.abs(concatLinear) > 0).type(torch.float)

    self.trackSparsity = torch.mean(nonZeroWeights)
    self.trackSparsityBias = torch.mean(nonZeroBias)
    self.trackSparsityLinear = torch.mean(nonZeroLinear)
