### EXPERIMENT NUMBER 01
### IHT AGD
setup_ihtAGD_SP90_K10_B20 = {
    "setupID":"setup_ihtAGD_SP90_K10_B20",
    "scheme":"ihtAGD" ,
    "sparsity":0.90,
    "kappa":10.0,
    "beta":10.0}

### IHT SGD

setup_ihtSGD_SP90_B5 = {
    "setupID":"setup_ihtSGD_SP90_B5",
    "scheme":"ihtSGD" ,
    "sparsity":0.90,
    "beta":5.0}

### SGD

setup_vanillaSGD_B5 = {
    "setupID":"setup_vanillaSGD_B5",
    "scheme":"vanillaSGD" ,
    "sparsity":0.90,
    "beta":5.0}


setups = [setup_ihtAGD_SP90_K10_B20,
          setup_ihtSGD_SP90_B5,
          setup_vanillaSGD_B5]

