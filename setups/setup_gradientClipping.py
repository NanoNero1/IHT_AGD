setup_ihtAGD_SP90_K10_B20 = {
    "setupID":"setup_ihtAGD_SP90_K10_B20",
    "scheme":"ihtAGD" ,
    "sparsity":0.90,
    "kappa":10.0,
    "beta":10.0,
    "gradientClip": True}

setup_ihtSGD_SP90_B5 = {
    "setupID":"setup_ihtSGD_SP90_B5",
    "scheme":"ihtSGD" ,
    "sparsity":0.90,
    "beta":5.0,
    "gradientClip": True}

setups = [setup_ihtAGD_SP90_K10_B20,
          setup_ihtSGD_SP90_B5]