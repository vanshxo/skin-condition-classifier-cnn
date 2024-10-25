import torch
print(torch.__version__)

# Test if MPS is available for Apple Silicon
print(torch.backends.mps.is_available())


