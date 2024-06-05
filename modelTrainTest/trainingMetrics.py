import torch
import torch.nn.functional as F

def getTestLoss(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
      data, target = next(iter(test_loader))
      data, target = data.to(device), target.to(device)
      output = model(data)

      # Negative Log-Likelihood Loss
      test_loss += F.nll_loss(output, target, reduction='sum').item()
    # The total loss is divided by the batch size to get the average
    test_loss /= test_loader.batch_size
    return test_loss

def getTestAccuracy(model, device, test_loader):
    model.eval()
    correct = 0
    # The testing accuracy is taken over the entire dataset
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # If the index of the maximum probability matches the target digit, we add it to the corrrect counter variable
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return accuracy