from torch.utils.data import DataLoader
from torch import nn, optim
import torch


class Train:
    
    def __init__(self, model:nn.Module, train_loader:DataLoader, validate_loader:DataLoader, device=torch.device('cpu'), loss_fn=nn.CrossEntropyLoss(), epochs:int=5, lr:float=10e-3, is_autoencoder=False):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.epochs = epochs
        self.lr = lr
        self.loss_fn = loss_fn
        self.is_autoencoder = is_autoencoder
    
    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        
        self.model.train()
        
        for epoch in range(self.epochs):
            running_loss = 0
            nb_items = 0
            print(f'\nepoch: {epoch}')
            for data in self.train_loader:
                optimizer.zero_grad()
                x, y = data
                if self.is_autoencoder:
                    y = x
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                nb_items += x.shape[0]
            
            print(f'training loss: {running_loss/nb_items:5.5f}')
            
            running_loss = 0
            nb_items = 0
            for data in self.validate_loader:
                x, y = data
                if self.is_autoencoder:
                    y = x
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                nb_items += x.shape[0]
                running_loss += self.loss_fn(pred, y).item()
            print(f'validation loss: {running_loss/nb_items:5.5f}')
                
                