import argparse
from utils import models, data_loader
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

LATENT_DIM = 2  # visualising in 2d so only works when models are trained with 2 dims for latent space
IMAGE_SIZE = 1  # controls the image size when saved

# image post-processing. Thresholds to create 3 different bands of pixel values {0, 0.7, 1}
THRESH_LOW = 0.55
THRESH_HIGH = 0.6
MID_PIXEL = 0.7


def latent_space_plots(args:argparse.ArgumentParser, dataloader:DataLoader):
    ''' Iterate through all saved models and generate plots exploring their latent space in different ways
    '''
    # setup directories for plots
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    latent_dist_dir = output_dir / 'latent_distribution'
    latent_phase_dir = output_dir / 'latent_phase'
    prediction_plots_dir = output_dir / 'prediction'
    norm_gen_plots_path = output_dir / 'norm_gen'
    latent_dist_dir.mkdir(exist_ok=True)
    latent_phase_dir.mkdir(exist_ok=True)
    prediction_plots_dir.mkdir(exist_ok=True)
    norm_gen_plots_path.mkdir(exist_ok=True)
    
    # setup template models and parse out model names
    model_names = [model.name for model in models_dir.iterdir() if model.is_file()]
    model_autoencoder = models.AutoencoderFc(hidden_dim=args.hidden_dim, latent_dim=LATENT_DIM)
    model_vae = models.VaeFc(hidden_dim=args.hidden_dim, latent_dim=LATENT_DIM)
    print(f'{len(model_names)} models found:\n   {model_names}\n')
    
    # iterate through each model and generate plots
    for model_name in tqdm(model_names, desc='Generating images'):
        model_path = models_dir / model_name
        is_vae = True if model_name[:3] == 'vae' else False
        # use the correct model architecture for the loaded state_dict
        model = model_vae if is_vae else model_autoencoder
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # generate and save plots
        latent_distribution(model, is_vae, dataloader, args.nb_batches, args.batch_size, model_name, latent_dist_dir, is_save=True)
        latent_phase_plot(model, is_vae, model_name, latent_phase_dir, is_save=True)
        prediction_plots(model, dataloader, is_vae, model_name, prediction_plots_dir, is_save=True)
        sample_from_norm(model, is_vae, dataloader, model_name, norm_gen_plots_path, is_save=True)


def latent_phase_plot(model:torch.nn.Module, is_vae:bool, model_name:str=None, output_dir:Path=None, max_range:int=2, steps:int=10, is_save:bool=False):
    ''' generate images across the 2D latent space
        This gives a view of how the generate images are distributed across the space without the label
    '''
    # create the latent space points that will be passed through the model.decoder()
    x = torch.arange(start=-max_range, end=max_range, step=2*max_range/(steps))
    y = torch.arange(start=-max_range, end=max_range, step=2*max_range/(steps))
    x, y = torch.meshgrid([x,y], indexing='xy')

    # stack all latent space points into a batch of the correct dimesions [B, 2]
    batch = torch.vstack([x.flatten(), -y.flatten()]).permute(1,0)
    # if model is VAE then we create a very small varience so we generate an image from the given point
    if is_vae:
        x_hat = model.decoder(batch, -100 * torch.ones_like(batch)).detach()
    else:
        x_hat = model.decoder(batch).detach()
    
    # post-process the image to be within the 3 threshold bands
    x_hat = x_hat.squeeze(1).numpy()
    x_hat = post_process_img(x_hat)

    # plot each generate image in relative position from their latent space position
    fig, axes = plt.subplots(steps, steps, figsize=(steps * IMAGE_SIZE, steps * IMAGE_SIZE))
    for i, ax in enumerate(axes.flatten()):
        img = x_hat[i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if is_save:
        latent_phase_path = output_dir / (model_name + '.png')
        plt.savefig(latent_phase_path)
        plt.close()
        

def post_process_img(img:np.ndarray) -> np.ndarray:
    ''' Given an image array, threshold values within 3 bands
    '''
    return(np.where(img > THRESH_LOW, np.where(img > THRESH_HIGH, 1, MID_PIXEL), 0))


def prediction_plots(model:torch.nn.Module, dataloader:DataLoader, is_vae:bool, model_name:str=None, output_dir:Path=None, cols:int=3, rows:int=2, is_save:bool=False):
    ''' generate a comparision of input images shown next to their generate images
    '''
    # get just 1 batch (assuming batch_size < rows * cols)
    x, _ = next(iter(dataloader))

    if is_vae:
        x_hat, *_ = model(x)
    else:
        x_hat = model(x)
    
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * IMAGE_SIZE * 2, rows * IMAGE_SIZE))
    for i in range(rows):
        # for each input and generated image, concatenate into one image and plot on the figure
        for j in range(cols):
            img_target = x[i * (cols) + (j)][0].detach().numpy()  # Extracting an image from x
            img_hat = x_hat[i * (cols) + (j)][0].detach().numpy()  # Extracting an image from x_hat 
            img = np.concatenate([img_target, img_hat], axis=1)
            img = post_process_img(img)  # post-process image to be in 3 bands
            axs[i, j].imshow(img, cmap='gray')
            axs[i, j].axis('off')
    plt.tight_layout()
    
    if is_save:
        prediction_plot_path = output_dir / (model_name + '.png')
        plt.savefig(prediction_plot_path)
        plt.close()


def sample_from_norm(model:torch.nn.Module, is_vae:bool, dataloader:DataLoader, model_name:str=None, output_dir:Path=None, cols:int=9, rows:int=9, is_save:bool=False):
    ''' sample from a 2D N(0,1) Gaussian, and generate images
    '''
    # sample from 2d Normal distributiion (note: out corvairance is diaganol, therefore each dimension is independent so we can use 1D normal)
    z = torch.randn([cols * rows, 2])
    
    # if VAE we give a very small logvar. This is because we have already sampled from N(0,1) so we don't need to sample again
    if is_vae:
        x_hat = model.decoder(z, -100 * torch.ones_like(z)).detach().numpy()
    else:
        x_hat = model.decoder(z).detach().numpy()
    x_hat = post_process_img(x_hat)
    
    # plot images
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(rows * IMAGE_SIZE, cols * IMAGE_SIZE))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(x_hat[i,0], cmap='gray')
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if is_save:
        norm_gen_plot_path = output_dir / (model_name + '.png')
        plt.savefig(norm_gen_plot_path)
        plt.close()


def latent_distribution(model:torch.nn.Module, is_vae:bool, dataloader:DataLoader, nb_batches:int, batch_size:int, model_name:str=None, output_dir:Path=None, is_save=False):
    ''' Plot a given number (nb_batches * batch_size) of inputs mapped onto the latent space. This shows the distribution of each label in latent space
    '''
    # create tensors to hold the multiple batches of encoded latent space points, and their respective labels
    nb_total = nb_batches * batch_size
    encoded_data = torch.zeros(size=(nb_total, LATENT_DIM))
    labels = torch.zeros(nb_total)
    
    # iterate through required number of batches to calculate the latent space and label of each input 
    for i, batch in enumerate(dataloader):
        x, y = batch
        labels[i * batch_size:(i+1) * batch_size] = y
        pred = model.encoder(x)
        z = pred[0] if is_vae else pred
        encoded_data[i * batch_size: (i+1) * batch_size] = z
        # if we've done the number of required batches break the loop
        if i + 1 == nb_batches:
            break
    
    # convert to numpy
    labels = labels.detach().numpy()
    encoded_data = encoded_data.detach().numpy()

    # 10 distinct colours used to plot the latent points of each character class
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    cmap = ListedColormap(colors)

    # plot all latent points and colour by label
    plt.scatter(encoded_data[:,0], encoded_data[:,1], c=labels, cmap=cmap)
    plt.gca().set_aspect('equal')  # keep plot square
    plt.colorbar()
    
    if is_save:
        latent_dist_path = output_dir / (model_name + '.png')
        plt.savefig(latent_dist_path)
        plt.close()


def main(args:argparse.ArgumentParser):
    _, dataloader = data_loader.CustomDataloaders.MNIST(batch_size=args.batch_size)
    latent_space_plots(args, dataloader)
    print('\nProgram Complete')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Investiagte the 2D latent space of the autoencoder and vae models you've already trained")

    parser.add_argument("--models_dir", type=str, default='models', help="relative directory path to save models, default=models")
    parser.add_argument("--output_dir", type=str, default='images', help="relative directory path to directory of output images, default=images")
    parser.add_argument("--nb_batches", type=int, default=20, help="number of batches to be plotted in latent_distribution plot, default=20")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size when loading data, default=64")
    parser.add_argument("--hidden_dim", type=int, default=200, help="hidden dimension size of models, default=200")
    args = parser.parse_args()
    
    main(args)