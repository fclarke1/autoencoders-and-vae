import argparse
from utils import train, models, data_loader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch


def train_models(args):
    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True)
    
    if args.train_autoencoder or args.train_vae:
        dataloader_train, dataloader_valid = data_loader.CustomDataloaders.MNIST(batch_size=args.batch_size)
    
    if args.train_autoencoder:
        print(f'**** Autoencoder Training ****')
        model_name = f'auto_e{args.epochs}'
        model_path = models_dir / model_name
        writer = SummaryWriter(f'./logs/{model_name}')
        
        model = models.AutoencoderFc(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
        trainer = train.Train(model=model, epochs=args.epochs, train_loader=dataloader_train, validate_loader=dataloader_valid, writer=writer, use_cuda=args.use_cuda, is_autoencoder=True)
        trainer.train()
        torch.save(model.state_dict(), model_path)
    
    if args.train_vae:
        vae_c_list = [float(x) for x in args.vae_c_list.split(',')]
        for c in vae_c_list:
            print(f'\n**** VAE Training (c={c}) ****')
            model_name = f'vae_c{c:.1f}_e{args.epochs}'
            model_path = models_dir / model_name
            writer = SummaryWriter(f'./logs/{model_name}')
            
            model = models.VaeFc(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, c=c)
            trainer = train.Train(model=model, epochs=args.epochs, train_loader=dataloader_train, validate_loader=dataloader_valid, writer=writer, use_cuda=args.use_cuda, is_autoencoder=True)
            trainer.train()
            torch.save(model.state_dict(), model_path)



def main(args):
    train_models(args)
    
    print('\nProgram Complete')
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your autoencoder models and save them in your directory")

    parser.add_argument("--models_dir", type=str, default='models', help="relative directory path to save models, default=models")
    parser.add_argument("--train_autoencoder", action='store_true', help="train the AutoEncoder model")
    parser.add_argument("--train_vae", action='store_true', help="train the VAE model")
    parser.add_argument("--epochs", type=int, default=10, help="number of Epochs to train each model, default=10")
    parser.add_argument("--vae_c_list", type=str, default='1', help="comma separated list of c values to train vae with, eg. '1,2,3', default=1")
    parser.add_argument("--use_cuda", action='store_true', help="use cuda if available")
    parser.add_argument("--hidden_dim", type=int, default=200, help="hidden dimension size of models, default=200")
    parser.add_argument("--latent_dim", type=int, default=2, help="latent dimension size of models, default=2")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training models, default=64")
    args = parser.parse_args()
    
    main(args)
