import torch

class dimitriPytorchSGD(torch.optim.SGD):
    def __init__(self,params,**kwargs):
        super().__init__(params,**kwargs)
        self.methodName = "Dimitri's Pytorch SGD: so that it works with this framework"