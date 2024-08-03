import torch

class dimitriPytorchSGD(torch.optim.SGD):
    def __init__(self,params,**kwargs):
        super().__init__(params,lr=kwargs['lr'])
        self.trialNumber = 0
        self.methodName = "Dimitri's Pytorch SGD: so that it works with this framework"