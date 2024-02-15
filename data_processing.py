import scipy.io
import torch

# Load your .mat file
mat = scipy.io.loadmat("data/DS105-objectviewing-ROI-std4mm-DLY-TempAlign.mat")

print(mat.keys())
