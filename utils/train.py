from torch.utils.data import DataLoader
from torch import nn, optim
import torch


class Train:
    
    def __init__(self, model:nn.Module, train_loader:DataLoader, validate_loader:DataLoader, loss_fn=nn.CrossEntropyLoss(), epochs:int=5, lr:float=10e-3, use_cuda=False, is_autoencoder=False):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.device = self.device
        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.epochs = epochs
        self.lr = lr
        self.is_autoencoder = is_autoencoder
    
    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        
        for epoch in range(self.epochs):
            self.model.train()
            
            running_loss = 0
            nb_items = 0
            print(f' epoch: {epoch+1}')
            for data in self.train_loader:
                optimizer.zero_grad()
                x, y = data
                if self.is_autoencoder:
                    y = x
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(x)
                if isinstance(pred, tuple):
                    loss = self.model.calculate_loss(y, *pred)
                else:
                    loss = self.model.calculate_loss(y, pred)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                nb_items += x.shape[0]
            print(f'  training loss:   {running_loss/nb_items:5.2f}')
            
            running_loss = 0
            nb_items = 0
            self.model.eval()
            for data in self.validate_loader:
                x, y = data
                if self.is_autoencoder:
                    y = x
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                nb_items += x.shape[0]
                if isinstance(pred, tuple):
                    loss = self.model.calculate_loss(y, *pred)
                else:
                    loss = self.model.calculate_loss(y, pred)
                running_loss += loss.item()
            print(f'  validation loss: {running_loss/nb_items:5.2f}')