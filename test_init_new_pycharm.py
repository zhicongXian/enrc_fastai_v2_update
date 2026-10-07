import sys
import os

print("Python:", sys.executable)

print("\nNumPy-related modules BEFORE import:")
print([m for m in sys.modules if m.startswith("numpy")])

print("\nPYTHONPATH:")
print(os.environ.get("PYTHONPATH"))

print("\nCONDA_PREFIX:")
print(os.environ.get("CONDA_PREFIX"))

print("\nsys.path:")
for p in sys.path:
    print(p)

print("\n--- importing NumPy ---")
import numpy as np

print("NumPy version:", np.__version__)
print("NumPy path:", np.__file__)

print("\n--- importing Torch ---")
import torch

print("Torch version:", torch.__version__)
print("Torch path:", torch.__file__)

a = np.array([1, 2, 3])
print("torch.from_numpy:", torch.from_numpy(a))