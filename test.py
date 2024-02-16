from config import *
from data_processing import scan_dataset

ds = scan_dataset(ROI_PATH_2_MAT).with_format("torch", device=DEVICE)
