from IHT_AGD.optimizers.vanillaAGD import vanillaAGD
from IHT_AGD.optimizers.ihtAGD import ihtAGD
from IHT_AGD.optimizers.ihtSGD import ihtSGD
from IHT_AGD.optimizers.vanillaSGD import vanillaSGD
from IHT_AGD.optimizers.untouchedBias_ihtAGD import untouchedBias_ihtAGD
from IHT_AGD.optimizers.clipGradientIHTAGD import clipGradientIHTAGD
from IHT_AGD.optimizers.ztSparse_ihtAGD import ztSparse_ihtAGD
from IHT_AGD.optimizers.nativePytorchSGD import dimitriPytorchSGD
import sys

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# NOTE: This function has been DEPRECATED
# nonetheless it might still be useful for debugging purposes in the future

####################################################################### DEPRECATED #################################################
def chooseOptimizer(setup,model,trialNumber,device=None):
  match setup["scheme"]:
    case "vanillaSGD":
         optimizer = vanillaSGD(model.parameters(),lr=setup["lr"],model=model,beta=setup["beta"])
    case "ihtSGD":
         optimizer = ihtSGD(model.parameters(),sparsity=setup["sparsity"],lr=setup["lr"],model=model,beta=setup["beta"],device=device)
    case "vanillaAGD":
         optimizer = vanillaAGD(model.parameters(),kappa=setup["kappa"],beta=setup["beta"],model=model,device=device)
    case "ihtAGD":
         optimizer = ihtAGD(model.parameters(),sparsity=setup["sparsity"],kappa=setup["kappa"],beta=setup["beta"],model=model,device = device)
    case "untouchedIhtAGD":
        pass
    case "pytorchSGD":
        optimizer = dimitriPytorchSGD(model.parameters())
    case _:
        pass
  optimizer.trialNumber = trialNumber
  return optimizer
#####################################################################DEPRECATED########################################


""" Desc: this function sets up the optimizer and handles the passing of the parameters"""
def fixedChooseOptimizer(setup,model,**kwargs):

  # Joining the kwargs with the setup arguments
  keyWordArgs = setup | kwargs

  # Filtering the kwargs here, just so that we know what data we give the optimizers
  keyWordArgsKeys = keyWordArgs.keys()
  badKeys = []
  allowedVars = ['lr','sparsity','alpha','beta','kappa','device','scheme','functionsToHelpTrack','variablesToTrack','run','train_loader',
                   'test_loader','trialNumber','expensiveVariables','expensiveFunctions']
  for key in keyWordArgsKeys:
    # The values we allow the optimizer to pick up
    if key not in allowedVars:
        badKeys.append(key)

  # Sometimes we want to exclude bad parameters, to avoid overwriting
  for badKey in badKeys:
    del keyWordArgs[badKey] 

  # Initialization of the optimizer
  optimizerClass = str_to_class(setup['scheme'])
  optimizer = optimizerClass(model.parameters(),model=model,**keyWordArgs)
  optimizer.setupID = setup['setupID']

  return optimizer