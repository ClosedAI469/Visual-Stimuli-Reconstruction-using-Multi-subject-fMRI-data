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


class Classifier_Dataset():
    def __init__(self, encodings_dir, data, encoding_names):
        self.preprocess(data, encoding_names, encodings_dir)
        self.num_inference_steps = 1000
        self.idx = None
        self.i = 0
        self.scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.scheduler.set_timesteps(self.num_inference_steps)

    def __len__(self):
        return len(self.encoding_names)

    def preprocess(self, data, encoding_names, encodings_dir):
        self.encoding_names_set = set()
        self.encoding_names = list()
        self.data = list()
        for i in range(len(encoding_names)):
            f_name = encoding_names[i].strip()
            if f_name != '':
                file_path = os.path.join(encodings_dir, os.path.splitext(f_name)[0]) + ".pt"
                self.encoding_names_set.add(file_path)
                self.encoding_names.append(file_path)
                self.data.append(torch.Tensor(data[i]))
        self.encoding_names_set = np.array(list(self.encoding_names_set))
        self.encoding_names = np.array(self.encoding_names)
        
        self.data = np.array(data)
        

    def prep_epoch(self, batch_size = 16, shuffle=True):
        idx = np.arange(len(self.encoding_names))
        if shuffle:
            np.random.shuffle(idx)
        self.idx = np.array_split(idx, len(self)//batch_size)
        self.i = 0

    def get_batch(self, batch_size = 16, shuffle=True):
        if self.idx is None:
            self.prep_epoch(batch_size, shuffle)
        if self.i >= len(self.idx):
            self.prep_epoch(batch_size, shuffle)
        idx = self.idx[self.i]
        batch_data = torch.Tensor(self.data[idx])
        batch_images = []
        targets = []
        for i in idx:
            if np.random.choice([0,1]) == 1:
                image = torch.load(self.encoding_names[i], map_location='cpu')
                targets.append(1)
            else:
                file = np.random.choice(self.encoding_names_set)
                while file == self.encoding_names[i]:
                    file = np.random.choice(self.encoding_names_set)
                image = torch.load(file, map_location='cpu')
                targets.append(0)
            # noise images
            alpha_cumprod = self.scheduler.alphas_cumprod[np.random.randint(1000)]
            epsilon = torch.randn(image.shape)
            noised_image = (torch.sqrt(alpha_cumprod) * image) + (torch.sqrt(1-alpha_cumprod) * epsilon)
            batch_images.append(noised_image)
            
        batch_images = torch.cat(batch_images)
        self.i += 1
        return batch_data, batch_images, torch.Tensor(targets).unsqueeze(dim=-1)
    
