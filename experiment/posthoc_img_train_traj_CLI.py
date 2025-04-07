# %%
import re
import os
from glob import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from os.path import join
import pickle as pkl
from tqdm.auto import tqdm, trange
import sys
sys.path.append("/n/home12/binxuwang/Github/DiffusionLearningCurve")
from core.dataset_lib import load_dataset
from core.img_patch_stats_analysis_lib import compute_crossing_points, sweep_and_create_sample_store, process_img_mean_cov_statistics,\
     process_patch_mean_cov_statistics, plot_variance_trajectories, plot_mean_deviation_trajectories, \
     harmonic_mean, smooth_and_find_threshold_crossing
from core.trajectory_convergence_lib import analyze_and_plot_variance
from circuit_toolkit.plot_utils import saveallforms, to_imgrid
import math
def sweep_and_create_sample_store_sparse(sampledir, step_list=[]):
    """
    Sweeps through the sample directory and loads samples.

    Args:
        sampledir (str): Directory containing sample files.

    Returns:
        dict: Dictionary with epochs as keys and loaded samples as values.
    """
    # sample_paths = sorted(glob(join(sampledir, "samples_epoch_*.pt")), key=lambda x: int(re.findall(r'\d+', x)[0]))
    filename = "samples_epoch_{:06d}.pt"
    sample_store = {}
    for step in tqdm(step_list):
        sample_path = join(sampledir, filename.format(step))
        if not os.path.exists(sample_path):
            print(f"Warning: {sample_path} does not exist")
            continue
        sample_store[step] = torch.load(sample_path)
    return sample_store


def sweep_available_steps(sampledir):
    """
    Sweeps through the sample directory and loads samples.

    Args:
        sampledir (str): Directory containing sample files.

    Returns:
        dict: Dictionary with epochs as keys and loaded samples as values.
    """
    sample_paths = sorted(glob(join(sampledir, "samples_epoch_*.pt")), key=lambda x: int(re.findall(r'\d+', x)[0]))
    available_steps = []
    for sample_path in sample_paths:
        filename = os.path.basename(sample_path)
        match = re.match(r'samples_epoch_(\d+)\.pt', filename)
        if match:
            epoch = int(match.group(1))
            available_steps.append(epoch)
    return available_steps



def find_closest_step(target_step, available_steps):
    return min(available_steps, key=lambda x: abs(x - target_step))


def create_sample_montage(sample_store, steps, imgsize=32, nrow=None, index=1):
    """
    Create a montage of samples from different training steps.
    
    Args:
        sample_store: Dictionary with steps as keys and samples as values
        steps: List of steps to include in the montage
        imgsize: Size of each image
        nrow: Number of images per row in the montage
        
    Returns:
        PIL Image of the montage
    """
    # Create a grid of samples
    sample_collection = []
    for step in steps:
        if step not in sample_store:
            continue
        samples_temp = sample_store[step][index:index+1]
        if samples_temp.ndim == 2:
            samples_temp = samples_temp.view(1, -1, imgsize, imgsize)
        samples_temp = ((samples_temp + 1) / 2).clamp(0, 1)
        sample_collection.append(samples_temp)
    
    sample_collection = torch.cat(sample_collection, dim=0)
    if nrow is None:
        nrow = int(np.sqrt(sample_collection.shape[0]))
    montage = make_grid(sample_collection, nrow=nrow)
    mtg = ToPILImage()(montage)
    return mtg

exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/"

# %%
# Function to find the closest step in sample_store
experiment_names = [
    "FFHQ32_DiT_P2_192D_3H_6L_EDM_pilot",
    "FFHQ32_DiT_P2_384D_6H_6L_EDM_pilot",
    "FFHQ32_DiT_P2_768D_12H_12L_EDM_pilot",
    "FFHQ32_DiT_P2_768D_12H_6L_EDM_pilot",
    "FFHQ32_DiT_P4_384D_6H_6L_EDM_pilot",
    "FFHQ32_DiT_P4_768D_12H_12L_EDM_pilot",
    "FFHQ32_DiT_P4_768D_12H_6L_EDM_pilot",
    "FFHQ32_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm",
    "FFHQ32_fix_words_UNet_MLP_EDM_8L_3072D_lr1e-4",
    "FFHQ32_random_words_jitter_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm",
    "FFHQ32_random_words_jitter_UNet_MLP_EDM_8L_3072D_lr1e-4",
    "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm",
    "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4",
    "FFHQ_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm",
    "FFHQ_random_words_jitter_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm",
    "AFHQ32_DiT_P2_192D_3H_6L_EDM_pilot",
    "AFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm",
    "AFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4",
    "AFHQ_UNet_CNN_EDM_4blocks_wide64_attn_pilot_fixednorm",
    "CIFAR10_UNet_CNN_EDM_3blocks_wide128_attn_pilot_fixednorm",
    "CIFAR10_UNet_CNN_EDM_3blocks_wide128_attn_pilot_fixednorm_smalllr",
    "CIFAR_DiT_P2_192D_3H_6L_EDM_pilot",
    "CIFAR_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm",
    "CIFAR_UNet_MLP_EDM_8L_3072D_lr1e-4",
    
]

experiment_names = [
    "FFHQ32_UNet_CNN_EDM_3blocks_2x_wide128_fixednorm",
    "FFHQ32_UNet_CNN_EDM_2blocks_2x_wide128_fixednorm",
    "FFHQ32_UNet_CNN_EDM_1blocks_2x_wide128_fixednorm",
    "AFHQ32_UNet_CNN_EDM_3blocks_2x_wide128_fixednorm",
    "AFHQ32_UNet_CNN_EDM_2blocks_2x_wide128_fixednorm",
    "AFHQ32_UNet_CNN_EDM_1blocks_2x_wide128_fixednorm",
]

# available_steps = sweep_available_steps(sample_dir)
# steps2show = np.logspace(np.log10(1), np.log10(50000), 100).astype(int)#[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
# steps2show = [find_closest_step(step, available_steps) for step in steps2show]

synopsis_dir = join(exproot, "synopsis")
os.makedirs(synopsis_dir, exist_ok=True)
for expname in experiment_names:
    sample_dir = join(exproot, expname, "samples")
    available_steps = sweep_available_steps(sample_dir)
    steps2show = np.logspace(np.log10(1), np.log10(50000), 100).astype(int)#[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    steps2show = [find_closest_step(step, available_steps) for step in steps2show 
                        if step <= max(available_steps)]
    sample_store_sparse = sweep_and_create_sample_store_sparse(sample_dir, steps2show)
    # Create the montage using the function
    mtg = create_sample_montage(sample_store_sparse, steps2show, index=2, nrow=10)
    mtg.save(join(synopsis_dir, f"{expname}_train_trajectory.png"))

