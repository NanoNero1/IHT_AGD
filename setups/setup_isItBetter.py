setup_ihtAGD_SP90_K10_B10 = {
    "setupID":"setup_ihtAGD_SP90_K10_B10",
    "scheme":"ihtAGD" ,
    "sparsity":0.90,
    "kappa":10.0,
    "beta":10.0}

setup_ihtSGD_SP90_B5 = {
    "setupID":"setup_ihtSGD_SP90_B5",
    "scheme":"ihtSGD" ,
    "sparsity":0.90,
    "beta":5.0}

setup_vanillaSGD_B5 = {
    "setupID":"setup_vanillaSGD_B5",
    "scheme":"vanillaSGD" ,
    "beta":5.0}


setups = [
          setup_ihtAGD_SP90_K10_B10,
          setup_vanillaSGD_B5,
          setup_ihtSGD_SP90_B5
          ]