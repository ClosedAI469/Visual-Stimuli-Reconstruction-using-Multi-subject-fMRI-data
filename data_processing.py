"""
This file is used to process the data from the mat file.
It stores various dataset classes, enabling different ways to structure the data for different encoding approaches.
"""

import scipy.io as sio
from datasets import Dataset


def scan_dataset(path: str):
    """
    Load the mat file from the path and return a Dataset object.

    :param path: Path to the mat file
    :type path: str
    :return: A Dataset object
    :rtype: Dataset
    """
    mat = sio.loadmat(path, variable_names=("data", "mlabel"))

    return Dataset.from_dict({"features": mat["data"], "labels": mat["mlabel"]})


# Hobie: I think we should use the Dataset class from the huggingface datasets module.
# It's a bit more flexible than the PyTorch Dataset class.

# class ScanDataset(torch.utils.data.Dataset):
#     """
#     A PyTorch dataset. In this dataset, easy sample is one scan.
#     """
#
#     def __init__(self, data_dict: dict):
#         self.data: ndarray = data_dict["data"]
#         self.label_one_hot: ndarray = data_dict["mlabel"]
#         self.shape: tuple = self.data.shape[1:]
#
#     def __len__(self):
#         return self.label_one_hot.shape[0]
#
#     def __getitem__(self, idx):
#         return torch.from_numpy(self.data[idx]), torch.from_numpy(
#             self.label_one_hot[idx]
#         )
