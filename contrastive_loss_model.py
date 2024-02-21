import torch
from torchvision import transforms
from tqdm.auto import tqdm
from torch import autocast
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

class Contrastive_loss_trainer():
    def __init__(self, fMRI_encoder, dataSet):
        self.fMRI_encoder = fMRI_encoder
        self.image_encoder = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        self.image_encoder.eval()
        self.dataSet = dataSet

    def encode_image(self, x):
        # return tensor of size 16384
        with torch.no_grad():
            output = torch.flatten(self.image_encoder(x).latent_dist.sample(), start_dim=1)
        return output
    
    def NT_Xent_loss(self, fMRI_encoding, image_encoding):
        cos_sim = F.cosine_similarity(fMRI_encoding[:, None], image_encoding[None, :], dim=-1) #outputs a batchsize x batchsize tensor
        return F.softmax(cos_sim, dim=-1)

    def get_fMRI_encoder(self):
        return self.fMRI_encoder

    def train(self, epochs=1, batchsize=16):
        self.fMRI_encoder.train()
        train_loader = DataLoader(self.dataset, batch_size=batchsize, shuffle=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.fMRI_encoder.parameters())

        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                fMRIs, images = data
                encoded_images = self.encode_image(images)

                optimizer.zero_grad()
                
                fMRI_encodings = self.fMRI_encoder(fMRIs)
                cos_sim_matrix = self.NT_Xent_loss(fMRI_encodings, encoded_images)
                targets = torch.eye(batchsize)
                loss = loss_fn(cos_sim_matrix, targets)
                loss.backwards()

                optimizer.step()







