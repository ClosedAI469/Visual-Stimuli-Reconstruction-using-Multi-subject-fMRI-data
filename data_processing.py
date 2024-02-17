"""
This file is used to process Tony's data from https://drive.google.com/drive/u/1/folders/1SNdQ-jCSGvdP1N8iUoTomA2owBShQvWJ.

It stores various data formatter methods,
enabling different ways to structure the data for different encoding approaches.

Before putting the data in one of the data formatters, load the .mat files to a dictionary first.
"""

from datasets import Dataset


def dataset_scan(mat: dict):
    """
    Load the mat file from the path and return a Dataset object.

    Each sample represents a single fMRI scan.

    :param mat: A dictionary containing the data from the mat file is
    :type mat: dict
    :return: A Dataset object
    :rtype: Dataset
    """

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
