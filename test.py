from config import *
from data_processing import scan_dataset
from models import MLP

ds = scan_dataset(ROI_PATH_2_MAT).with_format("torch", device=DEVICE)

sample = ds["features"][0]
shape = sample.shape[0]
print(shape)
model = MLP(shape, 9, 4 * shape, 2 * shape).to(DEVICE)
print(model)
output = model(sample)
print(output)
