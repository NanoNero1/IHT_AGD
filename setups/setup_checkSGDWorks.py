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
    "kappa":3.0,
    "beta":100.0}

setup_vanillaAGD = {
    "setupID":"setup_vanillaAGD_K10_B10",
    "scheme":"vanillaAGD" ,
    "kappa":3.0,
    "beta":100.0}

setup_clipGradientIHTAGD_SP70_K10_B10 = {
    "setupID":"setup_clipGradientIHTAGD_SP70_K10_B10",
    "scheme":"clipGradientIHTAGD" ,
    "sparsity":0.70,
    "kappa":10.0,
    "beta":60.0}


setups = [setup_vanillaAGD,
          setup_clipGradientIHTAGD_SP70_K10_B10,
          setup_nativeSGD,
          setup_vanillaSGD,
            setup_IHTAGD_SP70_K10_B10]