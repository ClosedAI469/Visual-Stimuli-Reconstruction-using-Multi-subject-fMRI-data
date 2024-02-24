"""
The following snippet is a test for the functions and classes in data_processing.py and models.py.
"""

# %% imports
from torch.nn.utils.rnn import pad_packed_sequence, unpad_sequence

from config import ROI_PATH_2_MAT, DEVICE, N_VOXELS_ROI
from data_processing import dataset_scan, dataset_stimulus, dataset_run
import scipy.io as sio

from models import MLP, LSTM

#####################################
# Test 1 

print("###### printing Test 1 ######")
print()

# %% test for dataset_scan
mat = sio.loadmat(ROI_PATH_2_MAT)
print(mat.keys())

ds = dataset_scan(mat)
print(ds)
print(ds["features"])
sample = ds["features"][0:2]
print(sample)
print(len(sample))
print(sample.shape)
shape = sample.shape[-1]
print(shape)

# %% test for MLP
output_dim = 16384
model = MLP(
    shape,
    output_dim,
    int((shape + output_dim) * 0.75),
    int((shape + output_dim) * 0.5),
    int((shape + output_dim) * 0.25),
).to(DEVICE)
print(model)
output = model(sample)
print(output)
print(output.shape)

#####################################

# Test 2 

print("###### printing Test 2 ######")
print()
# %% test for dataset_stimulus
mat = sio.loadmat(ROI_PATH_2_MAT)

print(mat.keys())

ds = dataset_stimulus(mat, DEVICE)

sample = ds["features"][0:2]
packing = ds["packing"][0:2].tolist()

# %% test for LSTM (with packed sequence)
hidden_size = 128
model = LSTM(N_VOXELS_ROI, hidden_size).to(DEVICE)
print(model)

packed = model(sample, packing)
padded = pad_packed_sequence(packed[0], batch_first=True)

output = unpad_sequence(padded[0], padded[1], batch_first=True)

print("#######################")
print(output)
print("#######################")
# print(output.shape)

#####################################
# Test 3

print("###### printing Test 3 ######")
print()
# %% test for dataset_run
mat = sio.loadmat(ROI_PATH_2_MAT)
print(mat.keys())

ds = dataset_run(mat, DEVICE)

sample = ds["features"][0]

# %% test for LSTM (with tensor)
hidden_size = 128
model = LSTM(N_VOXELS_ROI, hidden_size).to(DEVICE)
print(model)

output = model(sample)

print(output)
