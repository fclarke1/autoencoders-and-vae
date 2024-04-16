from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

class VaeFc(nn.Module):
    '''Variational AutoEncoder using a fully connected model architecture
    '''
    def __init__(self, hidden_dim:int, latent_dim:int, c:float=0.5, seed:int=42):
        super().__init__()
        
        self.rng = np.random.default_rng(seed)
        self.c = c  # c = 1/2k where k is the variance of the normal
        self.LeakyReLU = nn.LeakyReLU()
        
        # deisgned for MNIST images, so needed to be 28*28 image size
        self.encoder_1 = nn.Linear(in_features=28*28, out_features=hidden_dim)
        self.encoder_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.encoder_mean = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        self.encoder_logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        
        self.decoder_1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        self.decoder_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.decoder_3 = nn.Linear(in_features=hidden_dim, out_features=28*28)
    
    
    def encoder(self, x:torch.Tensor):
        # encoder inputs image and outputs Gaussian parameters (mean, logvar)
        x = torch.flatten(x, start_dim=1)
        h = self.LeakyReLU(self.encoder_1(x))
        h = self.LeakyReLU(self.encoder_2(h))
        mean = self.encoder_mean(h)
        logvar = self.encoder_logvar(h)
        return mean, logvar


    def decoder(self, mean:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        # given gaussian parameters, sample z from distribution and pass z through the decoder to get an image
        device = mean.device
        eps = torch.randn_like(mean).to(device)
        z = mean + torch.exp(logvar) * torch.Tensor(eps)
        
        h = self.LeakyReLU(self.decoder_1(z))
        h = self.LeakyReLU(self.decoder_2(h))
        y_hat = torch.sigmoid(self.decoder_3(h))
        y_hat = torch.unflatten(y_hat, 1, (1, 28, 28))
        return y_hat
    
    
    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # given input image, pass through encoder and decoder to output generated image (with latent distribution parameters)
        mean, logvar = self.encoder(x)
        y_hat = self.decoder(mean, logvar)
        return y_hat, mean, logvar
    
    
    def calculate_loss(self, y:torch.Tensor, y_hat:torch.Tensor, mean:torch.Tensor, logvar:torch.Tensor) -> float:
        # loss used for training this model
        reproduction_loss = nn.functional.binary_cross_entropy(y_hat, y, reduction='sum')
        # this part of the loss is what makes this model a VAE. This is derived using ELBO
        variational_loss = - 0.5 * (1 + logvar - mean**2 - logvar.exp()).sum()
        loss = self.c * reproduction_loss + variational_loss
        return loss
    
    
class AutoencoderFc(nn.Module):
    '''Variational AutoEncoder using a fully connected model architecture
    '''
    def __init__(self, hidden_dim:int, latent_dim:int):
        super().__init__()
        self.LeakyReLU = nn.LeakyReLU()
        
        # deisgned for MNIST images, so needed to be 28*28 image size
        self.encoder_1 = nn.Linear(in_features=28*28, out_features=hidden_dim)
        self.encoder_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.encoder_z = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        
        self.decoder_1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        self.decoder_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.decoder_3 = nn.Linear(in_features=hidden_dim, out_features=28*28)
    
    
    def encoder(self, x:torch.Tensor) -> torch.Tensor:
        # encoder inputs image and outputs Gaussian parameters (mean, logvar)
        x = torch.flatten(x, start_dim=1)
        h = self.LeakyReLU(self.encoder_1(x))
        h = self.LeakyReLU(self.encoder_2(h))
        z = self.encoder_z(h)
        return z


    def decoder(self, z:torch.Tensor) -> torch.Tensor: 
        # given latent point z, pass z through the decoder to get an image       
        h = self.LeakyReLU(self.decoder_1(z))
        h = self.LeakyReLU(self.decoder_2(h))
        y_hat = torch.sigmoid(self.decoder_3(h))  # sigmoid so output is in [0,1]
        y_hat = torch.unflatten(y_hat, 1, (1, 28, 28))
        return y_hat
    
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # given input image, pass through encoder and decoder to output generated image
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat
    
    
    def calculate_loss(self, y:torch.Tensor, y_hat:torch.Tensor) -> float:
        # loss used for training this model
        reproduction_loss = nn.functional.binary_cross_entropy(y_hat, y, reduction='sum')
        loss = reproduction_loss
        return loss