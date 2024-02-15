import scipy.io
import torch

path_str = "data/DS105-objectviewing-ROI-std4mm-DLY-TempAlign.mat"

# Load your .mat file
mat = scipy.io.loadmat(path_str)

print(mat.keys())
