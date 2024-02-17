from config import *
from data_processing import dataset_scan
import scipy.io as sio
from models import MLP

mat = sio.loadmat(ROI_PATH_2_MAT)

ds = dataset_scan(mat).with_format("torch", device=DEVICE)

sample = ds["features"][0:2]
print(sample.shape)
shape = sample.shape[-1]
print(shape)
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
