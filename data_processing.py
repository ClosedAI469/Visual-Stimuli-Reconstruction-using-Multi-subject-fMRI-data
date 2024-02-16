"""
This file is used to process the data from the mat file.
It stores various dataset classes, enabling different ways to structure the data for different encoding approaches.
"""

from typing import Type
import torch
import scipy.io
from torch.utils.data import Dataset, dataset, DataLoader
import numpy as np
from numpy import ndarray
from config import *
from models import ScanMLP


def load_mat(path: str):
    """
    Load the mat file from the path.
    """
    return scipy.io.loadmat(path)


class ScanDataset(Dataset):
    """
    In this dataset, easy sample is one scan.
    """

    def __init__(self, data_dict: dict):
        self.data: ndarray = data_dict["data"]
        self.label_one_hot: ndarray = data_dict["mlabel"]
        self.shape: tuple = self.data.shape[1:]

    def __len__(self):
        return self.label_one_hot.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.from_numpy(
            self.label_one_hot[idx]
        )


# TODO test

data_dict = load_mat(ROI_PATH_2_MAT)
ds = ScanDataset(data_dict)

test_size = int(len(ds) * 0.1)

train_ds, val_ds, test_ds = dataset.random_split(
    ds, [len(ds) - test_size * 2, test_size, test_size]
)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=True)

dl = {"train": train_dl, "val": val_dl, "test": test_dl}

input_dims = ds.shape

model = ScanMLP(input_dims, 4 * input_dims[0], 9).to(DEVICE)

sample = ds[0][0].to(DEVICE)
output = model(sample)
print(output)
