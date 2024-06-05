# SOURCE: https://github.com/pytorch/examples/tree/main/mnist

import torch
import torch.nn.functional as F
from neptune import Run

### note to self: I think I also need to import it here natively, no?
#import trainingMetrics

""" Desc: function for one training epoch. At each step, we query the optimizer and log the training loss to Neptune.
  Output: training loss, testing loss, [! we do not capture testing accuracy here]
"""

def train(args, model, device, train_loader, optimizer, epoch,trialNumber=None,test_loader=None):

    model.train()

    # In case we get NaN, setting this to true should detect this
    # BUG: for some reason this fails to capture NaNs
    torch.autograd.set_detect_anomaly(True)

    # Making sure we know what scheme is implemented
    print(f"Current Scheme: {optimizer.methodName}")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Important to check if the gradients are truly none
        for group in optimizer.param_groups:
          for p in group['params']:
            print(p.grad)

        # Generating Predictions and Calculating Loss
        output = model(data)
        loss = F.nll_loss(output, target)

        ###LOG### Loss
        run[f"trials/{optimizer.trialNumber}/{optimizer.methodName}/loss"].append(loss)

        loss.backward()

        print(loss)


        ###INTERNAL FUNCTIONS TO FEED TO OPTIMIZER###


        #Really, we want to get rid of this!
        # NOTE: we should try to pass data on step? - nope!

        def getNewGrad(parameters):
          model.params = parameters
          newOutput = model(data)
          loss = F.nll_loss(newOutput, target)
          loss.backward()

        ### GET RID OF THIS TOO
        def getNewTestAccuracy():
          return getTestAccuracy(model,device,test_loader)

        # Optimization Step
        optimizer.currentDataBatch = (data,target)
        optimizer.step()


        ###LOG### On every 10 iterations, we can print out some information
        if batch_idx % 10 == 0 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))