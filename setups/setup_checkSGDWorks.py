setup_nativeSGD = {
    "setupID":"setup_nativeSGD",
    "scheme":"dimitriPytorchSGD" ,
    "lr":0.01}

setup_vanillaSGD = {
    "setupID":"setup_vanillaSGD",
    "scheme":"vanillaSGD" ,
    "beta":100.0}

setup_IHTAGD_SP70_K10_B10 = {
    "setupID":"setup_IHTAGD_SP70_K10_B10",
    "scheme":"ihtAGD" ,
    "sparsity":0.70,
    "kappa":10.0,
    "beta":600.0}


setups = [setup_IHTAGD_SP70_K10_B10,
          setup_nativeSGD,
          setup_vanillaSGD ]