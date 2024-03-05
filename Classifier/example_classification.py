import torch
import scipy.io
import numpy as np
import torchvision
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from torchvision import transforms
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from classifier_dataset import Classifier_Dataset


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(18678, 4096),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(4096),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, fmri, image):
        image = self.flatten(image)
        x = torch.cat([fmri, image], dim=-1)
        out = self.model(x)
        return out
        

def train(classifier, dset, batch_size=16, epochs=1):
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        batches_per_epoch = len(dset)//batch_size
        running_loss = 0
        for i in range(batches_per_epoch):
            fmri, image, targets = dset.get_batch(batch_size=batch_size)
            optimizer.zero_grad()
            outputs = classifier(fmri, image)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"training loss for epoch:{epoch} = {running_loss/batches_per_epoch}")
            

def main():
    # example of how to run this classifier using the train function and classifier_dataset
    mat = scipy.io.loadmat("fMRI_data.mat")
    subjects = mat['subject'][0]
    scans = mat['data']
    file_names = scipy.io.loadmat("stimuli.mat")['stimuli']
    dset = Classifier_Dataset('encoded_stim', scans, file_names)
    model = Classifier()
    train(model, dset)
