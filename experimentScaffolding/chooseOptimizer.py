from IHT_AGD.optimizers.vanillaAGD import vanillaAGD
from IHT_AGD.optimizers.ihtAGD import ihtAGD
from IHT_AGD.optimizers.ihtSGD import ihtSGD
from IHT_AGD.optimizers.vanillaSGD import vanillaSGD



# NOTE: we want to remove this and replace it with a better system,
# nonetheless, this is still useful for now due to a small issue in my setup code
def chooseOptimizer(setup,model,trialNumber,device=None):
  match setup["scheme"]:
    case "vanillaSGD":
         #optimizer = vanillaSGD(model.parameters(),lr=setup["lr"],model=model,beta=setup["beta"])
         optimizer = vanillaSGD(model.parameters(),lr=setup["lr"],model=model,beta=setup["beta"])
    case "ihtSGD":
         #optimizer = ihtSGD(model.parameters(),sparsity=setup["sparsity"],lr=setup["lr"],model=model,beta=setup["beta"])
         optimizer = ihtSGD(model.parameters(),sparsity=setup["sparsity"],lr=setup["lr"],model=model,beta=setup["beta"])
    case "vanillaAGD":
         #I CANT FIX THIS UNTIL DEVICE IS IN THERE!!!!!!!!
         optimizer = vanillaAGD(model.parameters(),kappa=setup["kappa"],beta=setup["beta"],model=model,device=device)
    case "ihtAGD":
         optimizer = ihtAGD(model.parameters(),sparsity=setup["sparsity"],kappa=setup["kappa"],beta=setup["beta"],model=model,device = device)
    case "untouchedIhtAGD":
        pass
        #optimizer = untouchedIhtAGD(model.parameters(),sparsity=setup["sparsity"],kappa=setup["kappa"],beta=setup["beta"],model=model)
    case "pytorchSGD":
        pass
        #optimizer = dimitriPytorchSGD(model.parameters(),beta=3.0)#torch.optim.SGD(model.parameters(), lr=1.0/3.0)
    case _:
        pass
        #action-default
  optimizer.trialNumber = trialNumber
  return optimizer

def fixedChooseOptimizer(setup,model,**kwargs):
  print("test fixed chooose")

  # Joining the kwargs with the setup arguments
  keyWordArgs = setup | kwargs

  # Filtering the kwargs here, just so that we know what data we give the optimizers
  # CAREFUL!: is 'del' a safe function?
  for key in keyWordArgs.keys():
    # The values we allow the optimizer to pick up
    allowedVars = ['lr','sparsity','alpha','beta','kappa','device','scheme','functionsToHelpTrack','variablesToTrack','run','train_loader',
                   'test_loader','trialNumber']
    if key not in allowedVars:
        del keyWordArgs[key]

  # Not the most safe idea, is there another way?
  eval(f"optimizer = {setup['scheme']}(model.parameters(),model=model,**keyWordArgs)")