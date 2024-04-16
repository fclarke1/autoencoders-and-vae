from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torch


class Train:
    def __init__(self, model:nn.Module, train_loader:DataLoader, validate_loader:DataLoader, writer:SummaryWriter, epochs:int=5, lr:float=10e-3, use_cuda=False):
        ''' wrapper used to train models using the models custom .calculate_loss() function
        '''
        try:
            self.device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
        except:
            print('ERROR: Cuda not available - using cpu instead')
            self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.device = self.device
        self.writer = writer  # logger of losses during training
        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.epochs = epochs
        self.lr = lr
        self.accuracy_thresh = 0.6  # threshold for pixel value when calculating the validation dataset accuracy of generated image
    
    def train(self):
        ''' Train the model
        '''
        # by default use the Adam optimizer for all models
        optimizer = optim.Adam(self.model.parameters())
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # TRAINING RUN
            running_loss = 0
            nb_items = 0  # calculate total number of items so we can normalize the loss
            print(f' epoch: {epoch+1}')
            for data in self.train_loader:
                optimizer.zero_grad()
                x, _ = data
                y = x  # target is the input when training autoencoders
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                
                # if the output pred is a tuple then our model is the VAE which outputs [y_hat, mean, logvar]
                if isinstance(pred, tuple):
                    loss = self.model.calculate_loss(y, *pred)
                else:
                    loss = self.model.calculate_loss(y, pred)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                nb_items += x.shape[0]
                
            print(f'  training loss:   {running_loss/nb_items:5.2f}')  # normalize the loss
            self.writer.add_scalar('Loss/train', running_loss/nb_items, epoch)  # record losses
            
            # VALIDATION RUN
            running_loss = 0
            nb_pixels_correct = 0
            nb_items = 0
            self.model.eval()
            for data in self.validate_loader:
                x, _ = data
                y = x
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                
                # if the output pred is a tuple then our model is the VAE which outputs [y_hat, mean, logvar]
                # for validation record the loss and pixel accuracy (within a threshold)
                if isinstance(pred, tuple):
                    loss = self.model.calculate_loss(y, *pred)
                    nb_pixels_correct += self.pixels_correct(x, pred[0])
                else:
                    loss = self.model.calculate_loss(y, pred)
                    nb_pixels_correct += self.pixels_correct(x, pred)
                
                running_loss += loss.item()
                nb_items += x.shape[0]
            
            print(f'  validation loss: {running_loss/nb_items:5.2f}')
            self.writer.add_scalar('Loss/valid', running_loss/nb_items, epoch)
            self.writer.add_scalar('Accuracy', nb_pixels_correct/(nb_items*(28**2)), epoch)
        self.writer.close()
        
    
    def pixels_correct(self, y:torch.Tensor, y_hat:torch.Tensor) -> int:
        ''' Used to calculate the pixel-wise accuracy of models
        '''
        thresh = self.accuracy_thresh  # threshold is used to map all pixels to either {0, 1}
        y = torch.where(y < thresh, 0, 1)
        y_hat = torch.where(y_hat < thresh, 0, 1)
        # count the total number of pixels that are correct
        pixels_correct = (y * y_hat).sum() + ((y-1) * (y_hat - 1)).sum()
        return pixels_correct
        