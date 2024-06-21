setup_vanilla_AGD = {
    "setupID":"vanillaAGD",
    "scheme":"vanillaAGD" ,
    "kappa":5.0,
    "beta":50.0}

setup_vanilla_SGD = {
    "setupID":"vanillaSGD",
    "scheme":"vanillaSGD" ,
    "beta":50.0}

setup_ihtAGD_SP90 = {
    "setupID":"ihtAGD_SP90",
    "scheme":"ihtAGD" ,
    "sparsity":0.90,
    "kappa":5.0,
    "beta":50.0}

setup_ihtSGD_SP90 = {
    "setupID":"ihtSGD_SP90",
    "scheme":"ihtSGD" ,
    "sparsity":0.90,
    "beta":50.0}



setups = [setup_ihtSGD_SP90,setup_ihtAGD_SP90,setup_vanilla_AGD,setup_vanilla_SGD]