from IHT_AGD.experimentScaffolding.chooseOptimizer import chooseOptimizer

""" Desc these functions actually run the experiments and capture the model references"""

def runOneExperiment(setup=None,trialNumber=None,datasetChoice=None,**kwargs):

  match datasetChoice:
    case "MNIST":
      model = MNIST_convNet().to(device)
    case "CIFAR":
      abort()
      model = CIFAR_convNet().to(device)

  optimizer = chooseOptimizer(setup,model,trialNumber)
  #optimizer = eval(setup["scheme"])(setup,model,trialNumber)

  # Maybe the idea is that the model can change, but the test loader is global??
  optimizer.test_loader = test_loader

  ########################################
  ######################################## TO-DO: make sure the optimizer has kwargs parsed! 
  ######################################## so like test loader is also given to it
  #for () in kwargs
  # REFURBISH ME!!!!!!!!
  for key, value in kwargs.items():
    if key not in ['device','functionsToHelpTrack','variablesToTrack','run']:
      continue
    setattr(optimizer, key, value)
  optimizer.loggingInterval = 1

  # This implementation uses a Learning Rate Scheduler
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)
  for epoch in range(1, setup["epochs"] + 1):

    #print(optimizer.methodName)

    # Call to run one epoch of training
    train([],model, device, train_loader, optimizer, epoch,trialNumber,run=run)

    scheduler.step()
  return model

