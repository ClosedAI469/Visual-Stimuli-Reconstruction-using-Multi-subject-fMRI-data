import torch
import scipy.io
import numpy as np
import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast

def make_datasets():
    scans = scipy.io.loadmat('fMRI_data.mat')
    data = scans['data']
    image_names = scipy.io.loadmat('stimuli.mat')['stimuli']
    subjects = scans['subject'][0]
    dir = 'stimuli'

    train_scans = []
    train_fnames = []
    test_scans = []
    test_fnames = []

    prev = ''
    start_index = 0
    for i in range(len(image_names)):
        f_name = image_names[i].strip()
        if f_name != prev:
            if prev != '':
                if subjects[i] != 6:
                    # add the data to the train set
                    train_fnames.append(prev)
                    train_scans.append(torch.from_numpy(data[start_index:i]))
                else:
                    # add the data to the test set
                    test_fnames.append(prev)
                    test_scans.append(torch.from_numpy(data[start_index:i]))
            start_index = i
        prev = f_name
    return train_scans, train_fnames, test_scans, test_fnames ,dir
        

class Model():
    def __init__(self, fMRI_encoder):
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder
        self.fMRI_encoder = fMRI_encoder
        self.text_input = self.get_text_input()

        self.height = 512
        self.width = 512    
        self.num_inference_steps = 100            # Number of denoising steps
        self.guidance_scale =  0.5

        scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        scheduler.set_timesteps(self.num_inference_steps)

    def train(self):
        self.fMRI_encoder.train()

    def eval(self):
        self.fMRI_econder.eval()

    def get_text_input(self):
        prompt = [""]
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)[0]
        text_embeddings = torch.cat([text_embeddings, text_embeddings])
        return text_embeddings
        
    