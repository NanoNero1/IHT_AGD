"""Desc: this file is to check if we can import CUDA"""

import torch
import torchvision
# CUDA Check
print(torch.__version__)
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")
print("this is the new py file")