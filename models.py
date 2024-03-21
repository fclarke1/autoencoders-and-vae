from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # function that removes dims of 1, and the last 2 dims into one
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class VaeFc(nn.Module):
    def __init__(self, hidden_dim, latent_dim, c=1, seed=42):
        super().__init__()
        # c = 1/2k where k is the variance of the normal
        self.rng = np.random.default_rng(seed)
        self.c = c
        self.LeakyReLU = nn.LeakyReLU()
        
        self.encoder_1 = nn.Linear(in_features=28*28, out_features=hidden_dim)
        self.encoder_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.encoder_mean = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        self.encoder_logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        
        self.decoder_1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        self.decoder_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.decoder_3 = nn.Linear(in_features=hidden_dim, out_features=28*28)
    
    
    def encoder(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.LeakyReLU(self.encoder_1(x))
        x = self.LeakyReLU(self.encoder_2(x))
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mean, logvar


    def decoder(self, mean, logvar):
        q_shape = (mean.shape[0], mean.shape[1])
        q = self.rng.normal(size=q_shape)
        z = mean + torch.exp(logvar) * torch.Tensor(q)
        
        y_hat = self.LeakyReLU(self.decoder_1(z))
        y_hat = self.LeakyReLU(self.decoder_2(y_hat))
        y_hat = torch.sigmoid(self.decoder_3(y_hat))
        y_hat = torch.unflatten(y_hat, 1, (1, 28, 28))
        return y_hat
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        mean, logvar = self.encoder(x)
        y_hat = self.decoder(mean, logvar)
        return y_hat
    
    
    def calculate_loss(self, y, y_hat, mean, logvar):
        reproduction_loss = self.c * nn.functional.mse_loss(y, y_hat)
        
        
        


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 1 channel in, 6 channel out, kernal 5*5
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 channel in, 16 channel out, kernal 3*3
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # 1* 28 * 28
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # maxpool(6 * 28 * 28) -> 6 * 14 * 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # maxpool(16 * 10 * 10) -> 16 * 5 * 5
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class AutoEncoderConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.e_conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 1 channel in, 6 channel out, kernal 5*5
        self.e_conv2 = nn.Conv2d(6, 16, 5)  # 6 channel in, 16 channel out, kernal 3*3
        self.e_fc1 = nn.Linear(16 * 5 * 5, 120)
        self.e_fc2 = nn.Linear(120, 84)
        self.e_fc3 = nn.Linear(84, 6)

        # Decoder
        self.d_fc1 = nn.Linear(6, 84)
        self.d_fc2 = nn.Linear(84, 120)
        self.d_fc3 = nn.Linear(120, 16 * 5 * 5)
        self.d_conv1 = nn.ConvTranspose2d(16, 6, 5)
        self.d_conv2 = nn.Conv2d(6, 1, 5, padding=2)
    
    
    def encoder(self, x):
        # 1 * 28 * 28
        x = F.max_pool2d(F.relu(self.e_conv1(x)), 2)
        # 6 * 14 * 14
        x = F.max_pool2d(F.relu(self.e_conv2(x)), 2)
        # 16 * 5 * 5
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.e_fc1(x))
        # 120
        x = F.relu(self.e_fc2(x))
        # 84
        x = self.e_fc3(x)
        # 4
        return x
    
    def decoder(self, x):
        # 4
        x = F.relu(self.d_fc1(x))
        # 84
        x = F.relu(self.d_fc2(x))
        # 120
        x = F.relu(self.d_fc3(x))
        x = x.view(-1, 16, 5, 5)
        # 16 * 5 * 5
        x = F.relu(F.interpolate(x, scale_factor=2))
        # 16 * 10 * 10
        x = self.d_conv1(x)
        # 6 * 14 * 14
        x = F.relu(F.interpolate(x, scale_factor=2))
        # 6 * 28 * 28
        x = self.d_conv2(x)
        # 1 * 28 * 28
        return x
        
        
    
    
    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        x = self.decoder(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features