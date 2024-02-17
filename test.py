from config import *
from data_processing import scan_dataset
from models import MLP

ds = scan_dataset(ROI_PATH_2_MAT).with_format("torch", device=DEVICE)

sample = ds["features"][0]
shape = sample.shape[0]
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
