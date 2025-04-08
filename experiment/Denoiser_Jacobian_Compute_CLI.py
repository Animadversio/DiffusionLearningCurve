# %% [markdown]
# - [x] Load the network reconstruct the model 
# - [x] compute the jacobian of the network given a time input

# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
import os
from os.path import join
import json
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
sys.path.append("/n/home12/binxuwang/Github/DiffusionLearningCurve")
sys.path.append("/Users/binxuwang/Github/DiffusionLearningCurve/")
from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone
from core.toy_shape_dataset_lib import generate_random_star_shape_torch
from core.diffusion_basics_lib import *
from core.diffusion_edm_lib import *
from core.network_edm_lib import SongUNet, DhariwalUNet
from core.DiT_model_lib import *
from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone
from circuit_toolkit.plot_utils import saveallforms, to_imgrid, show_imgrid
from pprint import pprint

saveroot = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve"

# %%
def find_largest_ckpt_step(ckptdir, verbose=True):
    ckpt_files = [f for f in os.listdir(ckptdir) if f.startswith("model_epoch_") and f.endswith(".pth")]
    ckpt_steps = [int(f.split("_")[-1].split(".")[0]) for f in ckpt_files]
    if len(ckpt_steps) == 0:
        if verbose:
            print("No checkpoints found in the directory! check the path: ", ckptdir)
        return None
    else:
        if verbose:
            print(f"Found {len(ckpt_steps)} checkpoints in the directory, largest step is {max(ckpt_steps)}")
        return max(ckpt_steps)


def find_all_ckpt_steps(ckptdir, verbose=True):
    ckpt_files = [f for f in os.listdir(ckptdir) if f.startswith("model_epoch_") and f.endswith(".pth")]
    ckpt_steps = [int(f.split("_")[-1].split(".")[0]) for f in ckpt_files]
    if verbose:
        print(f"Found {len(ckpt_steps)} checkpoints in the directory, largest step is {max(ckpt_steps)}")
    return sorted(ckpt_steps)


# Refactored functions for visualizing gradient maps
from mpl_toolkits.axes_grid1 import ImageGrid

def compute_gradient_map(model, x_shape, sigma_val, output_coords, device, target="denoiser"):
    """
    Compute gradient map for a specific sigma value and output coordinates.
    
    Args:
        model: The denoiser model
        x_shape: Shape of the input tensor (e.g., (3, 32, 32))
        sigma_val: Sigma value for the denoiser
        output_coords: Tuple of (channel, y, x) coordinates for the output pixel
        device: Device to run computation on
    
    Returns:
        Gradient map tensor
    """
    # Set up the output map (focusing on the specified pixel)
    output_map = torch.zeros(x_shape).to(device)
    if output_coords is None:
        # Default to center pixel if not specified
        c, h, w = x_shape
        output_map[:, h//2, w//2] = 1
    else:
        c, y, x = output_coords
        if c is None:  # If channel is None, set all channels
            output_map[:, y, x] = 1
        else:
            output_map[c, y, x] = 1
    
    # Set up the probe point
    x_probe = sigma_val * torch.randn(x_shape).to(device)
    x_probe.requires_grad_(True)
    
    # Compute the denoiser output
    t_sigma = sigma_val * torch.ones(1, device=device)
    denoised = model(x_probe.view(1, *x_shape), t_sigma)
    if target == "denoiser":
        scalar = (output_map * denoised).sum()
    elif target == "score":
        score = (denoised - x_probe.view(1, *x_shape)) / t_sigma[:, None]
        scalar = (output_map * score[0]).sum()
    else:
        raise ValueError(f"Invalid target: {target}")
    
    # Compute scalar output and get gradient
    scalar.backward()
    # Return the gradient
    return x_probe.grad.detach().cpu()


def visualize_gradient_maps(model, x_shape, sigma_values, output_coords=None, target="denoiser",
                           device='cuda', reduction='abs_mean', figsize=(14, 14)):
    """
    Visualize gradient maps for different sigma values.
    
    Args:
        model: The denoiser model
        x_shape: Shape of the input tensor (e.g., (3, 32, 32))
        sigma_values: List of sigma values to test
        output_coords: Tuple of (channel, y, x) coordinates for the output pixel
                      If None, defaults to center pixel
        device: Device to run computation on
        reduction: How to reduce channel dimension ('abs_mean', 'mean', 'max', or None)
        figsize: Figure size for the plot
    
    Returns:
        List of computed gradient maps
    """
    gradient_maps = []
    
    # Compute gradient maps for each sigma
    for sigma_val in sigma_values:
        print(f"Computing gradient map for sigma = {sigma_val}")
        gradient_map = compute_gradient_map(model, x_shape, sigma_val, output_coords, device, target=target)
        
        # Apply reduction if specified
        if reduction == 'abs_mean':
            gradient_map = gradient_map.abs().mean(0)
        elif reduction == 'mean':
            gradient_map = gradient_map.mean(0)
        elif reduction == 'max':
            gradient_map = gradient_map.abs().max(0)[0]
        # If None, keep all channels
        
        gradient_maps.append(gradient_map)
        # Clear gradients for next iteration
        torch.cuda.empty_cache()
    
    # Create a montage of the gradient maps
    rows = int(len(sigma_values)**0.5)
    cols = (len(sigma_values) + rows - 1) // rows  # Ceiling division
    
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.3,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.1)
    
    # Add each gradient map to the grid
    for i, (gradient_map, sigma_val) in enumerate(zip(gradient_maps, sigma_values)):
        if i < len(grid):  # Ensure we don't go out of bounds
            ax = grid[i]
            
            # Handle multi-channel gradient maps
            if len(gradient_map.shape) == 3 and reduction is None:
                # Just show first channel if no reduction
                im = ax.imshow(gradient_map[0], cmap="viridis")
            else:
                im = ax.imshow(gradient_map, cmap="viridis")
                
            ax.set_title(f"σ = {sigma_val}")
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add colorbar
    grid.cbar_axes[0].colorbar(im)
    
    # Add title with output coordinates
    if output_coords:
        c, y, x = output_coords
        channel_str = f"channel {c}" if c is not None else "all channels"
        plt.suptitle(f"Gradient Maps of {target} for Different Sigma Values (Output at {channel_str}, y={y}, x={x})")
    else:
        plt.suptitle(f"Gradient Maps of {target} for Different Sigma Values (Output at center pixel)")
    
    plt.tight_layout()
    plt.show()
    
    return fig, gradient_maps



# %%
# %% [markdown]
# ### Loading CNN
#%%
def create_unet_model(config):
    unet = SongUNet(in_channels=config.channels, 
                out_channels=config.channels, 
                num_blocks=config.layers_per_block, 
                attn_resolutions=config.attn_resolutions, 
                decoder_init_attn=config.decoder_init_attn if 'decoder_init_attn' in config else True,
                model_channels=config.model_channels, 
                channel_mult=config.channel_mult, 
                dropout=config.dropout, 
                img_resolution=config.img_size, 
                label_dim=config.label_dim,
                embedding_type='positional', 
                encoder_type='standard', 
                decoder_type='standard', 
                augment_dim=config.augment_dim, #  no augmentation , 9 for defaults. 
                channel_mult_noise=1, 
                resample_filter=[1,1], 
                )
    pytorch_total_grad_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f'total number of trainable parameters in the Score Model: {pytorch_total_grad_params}')
    pytorch_total_params = sum(p.numel() for p in unet.parameters())
    print(f'total number of parameters in the Score Model: {pytorch_total_params}')
    return unet

# %%
# loading config 
expname = "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_saveckpt_fewsample"
savedir = join(saveroot, expname)
ckptdir = join(savedir, "ckpts")
sample_dir = join(savedir, "samples")
config = edict(json.load(open(f"{savedir}/config.json")))
args = edict(json.load(open(f"{savedir}/args.json")))
pprint(config)
unet = create_unet_model(config)
CNN_precd = EDMCNNPrecondWrapper(unet, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0)
# %% [markdown]
# #### Demo of gradient map
# %% [markdown]
# #### Gradient Map of CNN Unet denoisers

# %%
imgshape = (3, 32, 32)
figdir = join(savedir, "gradient_maps")
os.makedirs(figdir, exist_ok=True)
ckpt_step_list = find_all_ckpt_steps(ckptdir)
for ckpt_step in ckpt_step_list:
    ckpt_path = join(ckptdir, f"model_epoch_{ckpt_step:06d}.pth")
    CNN_precd.load_state_dict(torch.load(ckpt_path))
    device = "cuda"
    CNN_precd = CNN_precd.to(device).eval()
    CNN_precd.requires_grad_(False);
    sigma_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    fig, grad_maps = visualize_gradient_maps(CNN_precd, imgshape, sigma_values, output_coords=(None, 15, 15), target="denoiser");
    saveallforms(figdir, f"gradient_map_cnn_unet_denoiser_ckpt_{ckpt_step:06d}_pos_15_15", fig)
    fig, grad_maps = visualize_gradient_maps(CNN_precd, imgshape, sigma_values, output_coords=(None, 15, 10), target="denoiser");
    saveallforms(figdir, f"gradient_map_cnn_unet_denoiser_ckpt_{ckpt_step:06d}_pos_15_10", fig)
    fig, grad_maps = visualize_gradient_maps(CNN_precd, imgshape, sigma_values, output_coords=(None, 15, 20), target="denoiser");
    saveallforms(figdir, f"gradient_map_cnn_unet_denoiser_ckpt_{ckpt_step:06d}_pos_15_20", fig)
    fig, grad_maps = visualize_gradient_maps(CNN_precd, imgshape, sigma_values, output_coords=(None, 5, 5), target="denoiser");
    saveallforms(figdir, f"gradient_map_cnn_unet_denoiser_ckpt_{ckpt_step:06d}_pos_5_5", fig)
    fig, grad_maps = visualize_gradient_maps(CNN_precd, imgshape, sigma_values, output_coords=(None, 28, 28), target="denoiser");
    saveallforms(figdir, f"gradient_map_cnn_unet_denoiser_ckpt_{ckpt_step:06d}_pos_28_28", fig)
    plt.close("all")




# %% [markdown]
# ### Loading MLP
# %%
# loading config 
expname = "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample"
savedir = join(saveroot, expname)
ckptdir = join(savedir, "ckpts")
sample_dir = join(savedir, "samples")
# config = edict(json.load(open(f"{savedir}/config.json")))
args = edict(json.load(open(f"{savedir}/args.json")))
img_ndim = 3072
MLP_model = UNetBlockStyleMLP_backbone(ndim=img_ndim, nlayers=args.mlp_layers, nhidden=args.mlp_hidden_dim, time_embed_dim=args.mlp_time_embed_dim,)
MLP_precd = EDMPrecondWrapper(MLP_model, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0)
# ckpt_step = find_largest_ckpt_step(ckptdir)
# %% [markdown]
# #### Gradient Map of MLP denoisers
figdir = join(savedir, "gradient_maps")
os.makedirs(figdir, exist_ok=True)
ckpt_step_list = find_all_ckpt_steps(ckptdir)
for ckpt_step in ckpt_step_list:
    ckpt_path = join(ckptdir, f"model_epoch_{ckpt_step:06d}.pth")
    MLP_precd.load_state_dict(torch.load(ckpt_path))
    MLP_precd.requires_grad_(False)
    MLP_precd = MLP_precd.to(device).eval();
    MLP_precd_img = lambda x, sigma: MLP_precd(x.view(x.shape[0], -1), sigma).view(-1,*imgshape) # to have the same signature as the other denoisers
    
    sigma_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    fig, grad_maps = visualize_gradient_maps(MLP_precd_img, imgshape, sigma_values, output_coords=(None, 15, 15), target="denoiser");
    saveallforms(figdir, f"gradient_map_mlp_denoiser_ckpt_{ckpt_step:06d}_pos_15_15", fig)
    fig, grad_maps = visualize_gradient_maps(MLP_precd_img, imgshape, sigma_values, output_coords=(None, 15, 10), target="denoiser");
    saveallforms(figdir, f"gradient_map_mlp_denoiser_ckpt_{ckpt_step:06d}_pos_15_10", fig)
    fig, grad_maps = visualize_gradient_maps(MLP_precd_img, imgshape, sigma_values, output_coords=(None, 15, 20), target="denoiser");
    saveallforms(figdir, f"gradient_map_mlp_denoiser_ckpt_{ckpt_step:06d}_pos_15_20", fig)
    fig, grad_maps = visualize_gradient_maps(MLP_precd_img, imgshape, sigma_values, output_coords=(None, 5, 5), target="denoiser");
    saveallforms(figdir, f"gradient_map_mlp_denoiser_ckpt_{ckpt_step:06d}_pos_5_5", fig)
    fig, grad_maps = visualize_gradient_maps(MLP_precd_img, imgshape, sigma_values, output_coords=(None, 28, 28), target="denoiser");
    saveallforms(figdir, f"gradient_map_mlp_denoiser_ckpt_{ckpt_step:06d}_pos_28_28", fig)
    plt.close("all")


# %% [markdown]
# ### Loading DiT
# loading config 
expname = "FFHQ32_DiT_P2_192D_3H_6L_EDM_saveckpt_fewsample"
savedir = join(saveroot, expname)
ckptdir = join(savedir, "ckpts")
sample_dir = join(savedir, "samples")
config = edict(json.load(open(f"{savedir}/config.json")))
args = edict(json.load(open(f"{savedir}/args.json")))
pprint(config)
DiT_model = DiT(**config)
DiT_precd = EDMDiTPrecondWrapper(DiT_model, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0)
# Find the largest checkpoint step in the directory
# ckpt_step = find_largest_ckpt_step(ckptdir)
# device = "cuda"
# ckpt_path = join(ckptdir, f"model_epoch_{ckpt_step:06d}.pth")
# DiT_precd.load_state_dict(torch.load(ckpt_path))
# DiT_precd = DiT_precd.to(device).eval()
# DiT_precd.requires_grad_(False);

# %%
figdir = join(savedir, "gradient_maps")
os.makedirs(figdir, exist_ok=True)
ckpt_step_list = find_all_ckpt_steps(ckptdir)
for ckpt_step in ckpt_step_list:
    ckpt_path = join(ckptdir, f"model_epoch_{ckpt_step:06d}.pth")
    DiT_precd.load_state_dict(torch.load(ckpt_path))
    DiT_precd = DiT_precd.to(device).eval()
    DiT_precd.requires_grad_(False);
    
    sigma_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    fig, grad_maps = visualize_gradient_maps(DiT_precd, imgshape, sigma_values, output_coords=(None, 15, 15), target="denoiser");
    saveallforms(figdir, f"gradient_map_DiT_denoiser_ckpt_{ckpt_step:06d}_pos_15_15", fig)
    fig, grad_maps = visualize_gradient_maps(DiT_precd, imgshape, sigma_values, output_coords=(None, 15, 10), target="denoiser");
    saveallforms(figdir, f"gradient_map_DiT_denoiser_ckpt_{ckpt_step:06d}_pos_15_10", fig)
    fig, grad_maps = visualize_gradient_maps(DiT_precd, imgshape, sigma_values, output_coords=(None, 15, 20), target="denoiser");
    saveallforms(figdir, f"gradient_map_DiT_denoiser_ckpt_{ckpt_step:06d}_pos_15_20", fig)
    fig, grad_maps = visualize_gradient_maps(DiT_precd, imgshape, sigma_values, output_coords=(None, 5, 5), target="denoiser");
    saveallforms(figdir, f"gradient_map_DiT_denoiser_ckpt_{ckpt_step:06d}_pos_5_5", fig)
    fig, grad_maps = visualize_gradient_maps(DiT_precd, imgshape, sigma_values, output_coords=(None, 28, 28), target="denoiser");
    saveallforms(figdir, f"gradient_map_DiT_denoiser_ckpt_{ckpt_step:06d}_pos_28_28", fig)
    plt.close("all")

