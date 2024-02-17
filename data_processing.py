import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from config import *

"""
This file is used to process Tony's data from https://drive.google.com/drive/u/1/folders/1SNdQ-jCSGvdP1N8iUoTomA2owBShQvWJ.

It stores various data formatter methods,
enabling different ways to structure the data for different encoding approaches.

Before putting the data in one of the data formatters, load the .mat files to a dictionary first.
"""

from datasets import Dataset


def dataset_scan(mat: dict, device: torch.device = DEVICE):
    """
    Format the data to a Dataset object in which very sample is a datapoint of fMRI scan.

    :param device: The device to store the data on
    :type device: torch.device
    :param mat: A dictionary containing the data from the mat file
    :type mat: dict
    :return: A Dataset object
    :rtype: Dataset
    """

    return Dataset.from_dict(
        {"features": mat["data"], "labels": mat["mlabel"]}
    ).with_format("torch", device=device)


def dataset_run(mat: dict, device: torch.device):
    """
    Format the data to a Dataset object in which every sample is a sequence of consecutive fMRI scans during a run.

    :param mat: The dictionary containing the data from the mat file
    :type mat: dict
    :param device: The device to store the data on
    :type device: torch.device
    :return: A Dataset object
    :rtype: Dataset
    """
    return Dataset.from_dict(
        {
            "features": mat["data"].reshape(
                (N_RUNS_TOTAL, N_SCANS_PER_RUN) + mat["data"].shape[1:]
            ),
            "labels": mat["mlabel"].reshape(
                (N_RUNS_TOTAL, N_SCANS_PER_RUN) + mat["mlabel"].shape[1:]
            ),
        }
    ).with_format("torch", device=device)


def dataset_stimulus(mat: dict, device: torch.device):
    """
    Format the data to a Dataset object in which every sample is a padded sequence of consecutive fMRI scans
    during which the subject is looking at the same stimulus.

    :param mat: The dictionary containing the data from the mat file
    :type mat: dict
    :param device: The device to store the data on
    :type device: torch.device
    :return: A Dataset object
    :rtype: Dataset
    """
    stimuli = mat["label"].squeeze()
    features_raw = mat["data"]
    labels = mat["mlabel"]
    runs = mat["run"].squeeze()
    subjects = mat["subject"].squeeze()

    n_samples = features_raw.shape[0]
    n_features = features_raw.shape[-1]

    sequences = []
    length_sequences = []
    label_sequences = np.empty((0, labels.shape[-1]))

    prev_stimuli, prev_run, prev_subject = stimuli[0], runs[0], subjects[0]
    i_start = 0

    for i in range(n_samples):
        if (
            stimuli[i] != prev_stimuli
            or runs[i] != prev_run
            or subjects[i] != prev_subject
        ):
            sequences.append(torch.from_numpy(features_raw[i_start:i]).to(device))
            length_sequences.append(i - i_start)
            label_sequences = np.vstack(
                (label_sequences, labels[i - 1]), dtype=np.float32
            )

            i_start = i
            prev_stimuli, prev_run, prev_subject = stimuli[i], runs[i], subjects[i]

    sequences.append(torch.from_numpy(features_raw[i_start:]).to(device))
    length_sequences.append(n_samples - i_start)
    label_sequences = np.vstack((label_sequences, labels[-1]), dtype=np.float32)

    sequences_padded = pad_sequence(sequences, batch_first=True)

    print(sequences_padded.shape)

    return Dataset.from_dict(
        {
            "features": sequences_padded,
            "labels": label_sequences,
            "packing": length_sequences,
        }
    ).with_format("torch", device=device)
