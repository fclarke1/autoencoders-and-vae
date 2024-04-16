import argparse
from utils import models, data_loader
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

LATENT_DIM = 2
THRESH_LOW = 0.4
THRESH_HIGH = 0.6
MID_PIXEL = 0.7
IMAGE_SIZE = 2

def latent_space_plots(args, dataloader):
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    model_names = [model.name for model in models_dir.iterdir() if model.is_file()]
    model_autoencoder = models.AutoencoderFc(hidden_dim=args.hidden_dim, latent_dim=LATENT_DIM)
    model_vae = models.VaeFc(hidden_dim=args.hidden_dim, latent_dim=LATENT_DIM)
    
    latent_dist_dir = output_dir / 'latent_distribution'
    latent_phase_dir = output_dir / 'latent_phase'
    prediction_plots_dir = output_dir / 'prediction'
    norm_gen_plots_path = output_dir / 'norm_gen'
    latent_dist_dir.mkdir(exist_ok=True)
    latent_phase_dir.mkdir(exist_ok=True)
    prediction_plots_dir.mkdir(exist_ok=True)
    norm_gen_plots_path.mkdir(exist_ok=True)
    
    print(f'{len(model_names)} models found:\n   {model_names}\n')
    
    for model_name in tqdm(model_names, desc='Generating images'):
        model_path = models_dir / model_name
        is_vae = True if model_name[:3] == 'vae' else False
        model = model_vae if is_vae else model_autoencoder
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        latent_distribution(model, is_vae, latent_dist_dir, model_name, dataloader, args.plot_batches, args.batch_size)
        latent_phase_plot(model, is_vae, latent_phase_dir, model_name)
        prediction_plots(model, dataloader, is_vae, prediction_plots_dir, model_name)
        sample_from_norm(model, is_vae, norm_gen_plots_path, model_name, dataloader)


def latent_phase_plot(model, is_vae, output_dir, model_name, max_range=4, cols=10, is_save=True):
    x = torch.arange(start=-max_range, end=max_range, step=2*max_range/(cols))
    y = torch.arange(start=-max_range, end=max_range, step=2*max_range/(cols))
    x, y = torch.meshgrid([x,y], indexing='xy')

    batch = torch.vstack([x.flatten(), y.flatten()]).permute(1,0)
    if is_vae:
        x_hat = model.decoder(batch, -100 * torch.ones_like(batch)).detach()
    else:
        x_hat = model.decoder(batch).detach()
    x_hat = x_hat.squeeze(1).numpy()
    x_hat = np.where(x_hat > THRESH_LOW, np.where(x_hat > THRESH_HIGH, 1, MID_PIXEL), 0)

    # Create a figure and axes
    fig, axes = plt.subplots(cols, cols, figsize=(cols * IMAGE_SIZE, cols * IMAGE_SIZE))

    # Plot each image in the grid
    for i, ax in enumerate(axes.flatten()):
        img = x_hat[i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if is_save:
        latent_phase_path = output_dir / (model_name + '.png')
        plt.savefig(latent_phase_path)
        plt.close()


def prediction_plots(model, dataloader, is_vae, output_dir, model_name, cols=3, rows=2, is_save=True):
    x, _ = next(iter(dataloader))

    if is_vae:
        x_hat, *_ = model(x)
    else:
        x_hat = model(x)
    # Create a figure and axes for plotting
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * IMAGE_SIZE * 2, rows * IMAGE_SIZE))

    # Loop through each row
    for i in range(rows):
        # Alternate between x and x_hat in each row
        for j in range(cols):
            img_target = x[i * (cols) + (j)][0].detach().numpy()  # Extracting an image from x
            img_hat = x_hat[i * (cols) + (j)][0].detach().numpy()  # Extracting an image from x_hat 
            img = np.concatenate([img_target, img_hat], axis=1)
            img = np.where(img > THRESH_LOW, np.where(img > THRESH_HIGH, 1, MID_PIXEL), 0)
            # Plot the image
            axs[i, j].imshow(img, cmap='gray')
            axs[i, j].axis('off')
            
    

    # Show the plot
    plt.tight_layout()
    if is_save:
        prediction_plot_path = output_dir / (model_name + '.png')
        plt.savefig(prediction_plot_path)
        plt.close()


def sample_from_norm(model, is_vae, output_dir, model_name, dataloader, cols=3, rows=3, is_save=True):
    z = torch.randn([cols * rows, 2])  # covariance of 2D gaussian is diagonal so can sample from 1d multiple times instead from a 2d gaussian
    if is_vae:
        x_hat = model.decoder(z, -100 * torch.ones_like(z)).detach().numpy()
    else:
        x_hat = model.decoder(z).detach().numpy()
    
    # Create a figure and axes for plotting
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(rows * IMAGE_SIZE, cols * IMAGE_SIZE))
    
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(x_hat[i,0], cmap='gray')
        ax.axis('off')
    if is_save:
        norm_gen_plot_path = output_dir / (model_name + '.png')
        plt.savefig(norm_gen_plot_path)
        plt.close()



def latent_distribution(model, is_vae, output_dir, model_name, dataloader, plot_batches, batch_size, is_save=True):
    nb_total = plot_batches * batch_size
    encoded_data = torch.zeros(size=(nb_total, LATENT_DIM))
    labels = torch.zeros(nb_total)
    for i, batch in enumerate(dataloader):
        x, y = batch
        labels[i * batch_size:(i+1) * batch_size] = y
        pred = model.encoder(x)
        z = pred[0] if is_vae else pred
        encoded_data[i * batch_size: (i+1) * batch_size] = z
        if i + 1 == plot_batches:
            break
    labels = labels.detach().numpy()
    encoded_data = encoded_data.detach().numpy()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    cmap = ListedColormap(colors)

    plt.scatter(encoded_data[:,0], encoded_data[:,1], c=labels, cmap=cmap)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    
    if is_save:
        latent_dist_path = output_dir / (model_name + '.png')
        plt.savefig(latent_dist_path)
        plt.close()


def main(args):
    _, dataloader = data_loader.CustomDataloaders.MNIST(batch_size=args.batch_size)
    latent_space_plots(args, dataloader)
    
    print('\nProgram Complete')
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Investiagte the 2D latent space of the autoencoder and vae models you've already trained")

    parser.add_argument("--models_dir", type=str, default='models', help="relative directory path to save models, default=models")
    parser.add_argument("--output_dir", type=str, default='images', help="relative directory path to directory of output images, default=images")
    parser.add_argument("--plot_batches", type=int, default=20, help="number of batches to be plotted on plots, default=20")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size when loading data, default=64")
    parser.add_argument("--hidden_dim", type=int, default=200, help="hidden dimension size of models, default=200")
    
    
    args = parser.parse_args()
    
    main(args)