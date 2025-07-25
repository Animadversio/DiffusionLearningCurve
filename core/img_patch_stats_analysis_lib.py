
from glob import glob
import re
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt

# %%
import sys
import os
import json
from os.path import join
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
from circuit_toolkit.plot_utils import saveallforms, to_imgrid, show_imgrid



def sweep_and_create_sample_store(sampledir):
    """
    Sweeps through the sample directory and loads samples.

    Args:
        sampledir (str): Directory containing sample files.

    Returns:
        dict: Dictionary with epochs as keys and loaded samples as values.
    """
    sample_paths = sorted(glob(join(sampledir, "samples_epoch_*.pt")), key=lambda x: int(re.findall(r'\d+', x)[0]))
    sample_store = {}
    for sample_path in tqdm(sample_paths):
        filename = os.path.basename(sample_path)
        match = re.match(r'samples_epoch_(\d+)\.pt', filename)
        if match:
            epoch = int(match.group(1))
            sample_store[epoch] = torch.load(sample_path)
        else:
            print(f"Warning: could not extract epoch from filename: {filename}")
    return sample_store



def process_img_mean_cov_statistics(train_images, sample_store, savedir, device="cuda", imgshape=(3, 64, 64), save_pkl=True):
    img_shape = train_images.shape[1:]
    img_dim = np.prod(img_shape)
    img_mean = train_images.mean(dim=0)
    img_cov = torch.cov(train_images.view(train_images.shape[0], -1).T)
    img_eigval, img_eigvec = torch.linalg.eigh(img_cov.to(device))
    img_eigval = img_eigval.flip(0)
    img_eigvec = img_eigvec.flip(1)
    img_eigvec = img_eigvec.to(device)
    print(f"img_cov.shape: {img_cov.shape} computed on {train_images.shape[0]} images")
    mean_x_sample_traj = []
    cov_x_sample_traj = []
    diag_cov_x_sample_true_eigenbasis_traj = []
    step_slice = sorted([*sample_store.keys()])
    
    for training_step in tqdm(step_slice):
        x_final = sample_store[training_step]
        if isinstance(x_final, tuple):
            x_final = x_final[0]
        # x_final_patches = extract_patches(x_final.view(x_final.shape[0], *imgshape), patch_size=patch_size, patch_stride=patch_stride)
        x_final_flattened = x_final.view(x_final.shape[0], -1)
        mean_x_sample = x_final_flattened.mean(dim=0)
        cov_x_sample = torch.cov(x_final_flattened.to(device).T)
        mean_x_sample_traj.append(mean_x_sample.cpu())

        # Estimate the variance along the eigenvector of the covariance matrix
        cov_x_sample_true_eigenbasis = img_eigvec.T @ cov_x_sample.to(device) @ img_eigvec
        diag_cov_x_sample_true_eigenbasis = torch.diag(cov_x_sample_true_eigenbasis)
        diag_cov_x_sample_true_eigenbasis_traj.append(diag_cov_x_sample_true_eigenbasis.cpu())
        cov_x_sample_traj.append(cov_x_sample.cpu())
    
    mean_x_sample_traj = torch.stack(mean_x_sample_traj).cpu()
    cov_x_sample_traj = torch.stack(cov_x_sample_traj).cpu()
    diag_cov_x_sample_true_eigenbasis_traj = torch.stack(diag_cov_x_sample_true_eigenbasis_traj).cpu()

    if save_pkl:
        pkl.dump({
            "diag_cov_x_sample_true_eigenbasis_traj": diag_cov_x_sample_true_eigenbasis_traj, 
            "mean_x_sample_traj": mean_x_sample_traj,
            "cov_x_sample_traj": cov_x_sample_traj,
            "img_mean": img_mean.cpu(),
            "img_cov": img_cov.cpu(),
            "img_eigval": img_eigval.cpu(),
            "img_eigvec": img_eigvec.cpu(),
            "step_slice": step_slice
        }, open(f"{savedir}/sample_img_cov_true_eigenbasis_diag_traj.pkl", "wb"))
        print(f"Saved to {savedir}/sample_img_cov_true_eigenbasis_diag_traj.pkl")
    return img_mean, img_cov, img_eigval, img_eigvec, \
        mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj


def extract_patches(images, patch_size, patch_stride, avg_channels=False):
    if avg_channels:
        images = images.mean(dim=1, keepdim=True)
    B, C, H, W = images.shape
    patches = images.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, patch_size, patch_size)
    return patches


def process_patch_mean_cov_statistics(train_images, sample_store, savedir, patch_size=8, patch_stride=4, device="cuda", imgshape=(3, 64, 64), save_pkl=True, avg_channels=False):
    # images = Xtsr.view(Xtsr.shape[0], *imgshape)
    patches = extract_patches(train_images, patch_size=patch_size, patch_stride=patch_stride, avg_channels=avg_channels)
    patch_shape = patches.shape[1:]
    patch_dim = np.prod(patch_shape)
    patch_mean = patches.mean(dim=0)
    patch_cov = torch.cov(patches.view(patches.shape[0], -1).T)
    patch_eigval, patch_eigvec = torch.linalg.eigh(patch_cov.to(device))
    patch_eigval = patch_eigval.flip(0)
    patch_eigvec = patch_eigvec.flip(1)
    patch_eigvec = patch_eigvec.to(device)
    print(f"patch_cov.shape: {patch_eigval.shape} computed on {train_images.shape[0]} images")
    mean_x_patch_sample_traj = []
    cov_x_patch_sample_traj = []
    diag_cov_x_patch_sample_true_eigenbasis_traj = []
    step_slice = sorted([*sample_store.keys()])
    
    for training_step in tqdm(step_slice):
        x_final = sample_store[training_step]
        if isinstance(x_final, tuple):
            x_final = x_final[0]
        x_final_patches = extract_patches(x_final.view(x_final.shape[0], *imgshape), patch_size=patch_size, patch_stride=patch_stride, avg_channels=avg_channels)
        x_final_patches = x_final_patches.view(x_final_patches.shape[0], -1)
        mean_x_patch_sample = x_final_patches.mean(dim=0)
        cov_x_patch_sample = torch.cov(x_final_patches.to(device).T)
        mean_x_patch_sample_traj.append(mean_x_patch_sample.cpu())
        
        # Estimate the variance along the eigenvector of the covariance matrix
        cov_x_patch_sample_true_eigenbasis = patch_eigvec.T @ cov_x_patch_sample.to(device) @ patch_eigvec
        diag_cov_x_patch_sample_true_eigenbasis = torch.diag(cov_x_patch_sample_true_eigenbasis)
        diag_cov_x_patch_sample_true_eigenbasis_traj.append(diag_cov_x_patch_sample_true_eigenbasis.cpu())
        cov_x_patch_sample_traj.append(cov_x_patch_sample.cpu())
    
    mean_x_patch_sample_traj = torch.stack(mean_x_patch_sample_traj).cpu()
    cov_x_patch_sample_traj = torch.stack(cov_x_patch_sample_traj).cpu()
    diag_cov_x_patch_sample_true_eigenbasis_traj = torch.stack(diag_cov_x_patch_sample_true_eigenbasis_traj).cpu()

    if save_pkl:
        pkl.dump({
            "diag_cov_x_patch_sample_true_eigenbasis_traj": diag_cov_x_patch_sample_true_eigenbasis_traj, 
        "mean_x_patch_sample_traj": mean_x_patch_sample_traj,
        "cov_x_patch_sample_traj": cov_x_patch_sample_traj,
        "patch_mean": patch_mean.cpu(),
        "patch_cov": patch_cov.cpu(),
        "patch_eigval": patch_eigval.cpu(),
        "patch_eigvec": patch_eigvec.cpu(),
        "step_slice": step_slice
        }, open(f"{savedir}/sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}{'_avgchn' if avg_channels else ''}_cov_true_eigenbasis_diag_traj.pkl", "wb"))
        print(f"Saved to {savedir}/sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}{'_avgchn' if avg_channels else ''}_cov_true_eigenbasis_diag_traj.pkl")
    return patch_mean, patch_cov, patch_eigval, patch_eigvec, mean_x_patch_sample_traj, cov_x_patch_sample_traj, diag_cov_x_patch_sample_true_eigenbasis_traj
    
# Example usage:
# process_patch_statistics(Xtsr, sample_store, savedir, device)

def plot_variance_trajectories(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj, patch_eigval, slice2plot,
                               patch_size, patch_stride, savedir, dataset_name="FFHQ64", figsize=(6, 4)):
    ndim = patch_eigval.shape[0]
    if isinstance(slice2plot, slice):
        eigidx2plot = range(ndim)[slice2plot]
    elif isinstance(slice2plot, (list, tuple, np.ndarray, torch.Tensor)):
        eigidx2plot = slice2plot
    else:
        raise ValueError(f"Invalid slice2plot type: {type(slice2plot)}")
    max_eigid = max(eigidx2plot)
    plt.figure(figsize=figsize)
    plt.plot(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj[:, eigidx2plot], alpha=0.7)
    for i, eigid in enumerate(eigidx2plot):
        plt.axhline(patch_eigval[eigid].item(), color=f"C{i}", linestyle="--", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Variance")
    plt.title(f"Variance of learned patches ({patch_size}x{patch_size}, stride={patch_stride}) on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {patch_eigval[i].item():.2f}" for i in eigidx2plot], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}_cov_true_eigenbasis_diag_traj_raw_top{max_eigid}")
    plt.show()

    plt.figure(figsize=figsize)
    diag_cov_x_patch_sample_true_eigenbasis_traj_normalized = diag_cov_x_patch_sample_true_eigenbasis_traj / patch_eigval
    plt.plot(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj_normalized[:, eigidx2plot], alpha=0.7)
    plt.axhline(1, color="k", linestyle="--", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Variance [normalized by target variance]")
    plt.title(f"Variance of learned patches ({patch_size}x{patch_size}, stride={patch_stride}) on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {patch_eigval[i].item():.2f}" for i in eigidx2plot], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}_cov_true_eigenbasis_diag_traj_normalized_top{max_eigid}")
    plt.show()

# Example usage:
# plot_variance_trajectories(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj, patch_eigval, patch_size, patch_stride, savedir)

def plot_mean_deviation_trajectories(step_slice, mean_x_patch_sample_traj, patch_mean, patch_eigvec, patch_eigval, 
                                     slice2plot, patch_size, patch_stride, savedir, dataset_name="FFHQ64"):
    patch_mean_vec = patch_mean.view(-1)
    mean_deviation_traj = (mean_x_patch_sample_traj - patch_mean_vec) @ patch_eigvec.cpu()
    MSE_per_mode_traj = mean_deviation_traj.pow(2)
    MSE_per_mode_traj_normalized = MSE_per_mode_traj / patch_eigval

    ndim = patch_eigval.shape[0]
    if isinstance(slice2plot, slice):
        eigidx2plot = range(ndim)[slice2plot]
    elif isinstance(slice2plot, (list, tuple, np.ndarray, torch.Tensor)):
        eigidx2plot = slice2plot
    else:
        raise ValueError(f"Invalid slice2plot type: {type(slice2plot)}")
    max_eigid = max(eigidx2plot)    

    plt.figure()
    plt.plot(step_slice, mean_deviation_traj[:, eigidx2plot], alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Training step")
    plt.ylabel("mean deviation")
    plt.title(f"Mean deviation of learned patches ({patch_size}x{patch_size}, stride={patch_stride}) on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {patch_eigval[i].item():.2f}" for i in eigidx2plot], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}_mean_dev_eigenbasis_traj_raw_top{max_eigid}")
    plt.show()

    plt.figure()
    plt.plot(step_slice, MSE_per_mode_traj[:, eigidx2plot], alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Squared error of mean deviation")
    plt.title(f"Mean deviation of learned patches ({patch_size}x{patch_size}, stride={patch_stride}) on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {patch_eigval[i].item():.2f}" for i in eigidx2plot], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}_mean_SE_eigenbasis_traj_raw_top{max_eigid}")
    plt.show()

    plt.figure()
    plt.plot(step_slice, MSE_per_mode_traj_normalized[:, eigidx2plot], alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Squared error of mean deviation\n[normalized by target variance]")
    plt.title(f"Mean deviation of learned patches ({patch_size}x{patch_size}, stride={patch_stride}) on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {patch_eigval[i].item():.2f}" for i in eigidx2plot], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}_mean_SE_eigenbasis_traj_normalized_top{max_eigid}")
    plt.show()

# Example usage:
# plot_mean_deviation_trajectories(step_slice, mean_x_patch_sample_traj, patch_mean, patch_eigvec, patch_eigval, slice(None, 10, 1), patch_size, patch_stride, savedir)




def process_pnts_mean_cov_statistics(train_pnts, sample_store, savedir, device="cuda",):
    train_X_mean = train_pnts.mean(dim=0)
    train_X_cov = torch.cov(train_pnts.T)
    train_X_eigval, train_X_eigvec = torch.linalg.eigh(train_X_cov.to(device))
    train_X_eigval = train_X_eigval.flip(0)
    train_X_eigvec = train_X_eigvec.flip(1)
    train_X_eigvec = train_X_eigvec.to(device)
    print(f"train_X_eigval.shape: {train_X_eigval.shape} computed on {train_pnts.shape[0]} samples")
    mean_x_sample_traj = []
    cov_x_sample_traj = []
    diag_cov_x_sample_true_eigenbasis_traj = []
    step_slice = sorted([*sample_store.keys()])
    for training_step in tqdm(step_slice):
        x_final = sample_store[training_step]
        if isinstance(x_final, tuple):
            x_final = x_final[0]
        x_final_patches = x_final.view(x_final.shape[0], -1)
        mean_x_sample = x_final_patches.mean(dim=0)
        cov_x_sample = torch.cov(x_final_patches.to(device).T)
        mean_x_sample_traj.append(mean_x_sample.cpu())
        # Estimate the variance along the eigenvector of the covariance matrix
        cov_x_sample_true_eigenbasis = train_X_eigvec.T @ cov_x_sample.to(device) @ train_X_eigvec
        diag_cov_x_sample_true_eigenbasis = torch.diag(cov_x_sample_true_eigenbasis)
        diag_cov_x_sample_true_eigenbasis_traj.append(diag_cov_x_sample_true_eigenbasis.cpu())
        cov_x_sample_traj.append(cov_x_sample.cpu())
    
    mean_x_sample_traj = torch.stack(mean_x_sample_traj).cpu()
    cov_x_sample_traj = torch.stack(cov_x_sample_traj).cpu()
    diag_cov_x_sample_true_eigenbasis_traj = torch.stack(diag_cov_x_sample_true_eigenbasis_traj).cpu()

    pkl.dump({
        "diag_cov_x_sample_true_eigenbasis_traj": diag_cov_x_sample_true_eigenbasis_traj, 
        "mean_x_sample_traj": mean_x_sample_traj,
        "cov_x_sample_traj": cov_x_sample_traj,
        "train_X_mean": train_X_mean.cpu(),
        "train_X_cov": train_X_cov.cpu(),
        "train_X_eigval": train_X_eigval.cpu(),
        "train_X_eigvec": train_X_eigvec.cpu(),
        "step_slice": step_slice
    }, open(f"{savedir}/sample_pnts_cov_true_eigenbasis_diag_traj.pkl", "wb"))
    print(f"Saved to {savedir}/sample_pnts_cov_true_eigenbasis_diag_traj.pkl")
    return train_X_mean, train_X_cov, train_X_eigval, train_X_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj


def plot_sample_pnts_variance_trajectories(step_slice, diag_cov_x_sample_true_eigenbasis_traj, train_X_eigval, slice2plot,
                               savedir, dataset_name="Gaussian"):
    ndim = train_X_eigval.shape[0]
    max_eigid = max(range(ndim)[slice2plot])    
    plt.figure()
    plt.plot(step_slice, diag_cov_x_sample_true_eigenbasis_traj[:, slice2plot], alpha=0.7)
    for i, eigid in enumerate(range(ndim)[slice2plot]):
        plt.axhline(train_X_eigval[eigid].item(), color=f"C{i}", linestyle="--", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Variance")
    plt.title(f"Variance of learned samples on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {train_X_eigval[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_pnts_cov_true_eigenbasis_diag_traj_raw_top{max_eigid}")
    plt.show()

    plt.figure()
    diag_cov_x_sample_true_eigenbasis_traj_normalized = diag_cov_x_sample_true_eigenbasis_traj / train_X_eigval
    plt.plot(step_slice, diag_cov_x_sample_true_eigenbasis_traj_normalized[:, slice2plot], alpha=0.7)
    plt.axhline(1, color="k", linestyle="--", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Variance [normalized by target variance]")
    plt.title(f"Variance of learned samples on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {train_X_eigval[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_pnts_cov_true_eigenbasis_diag_traj_normalized_top{max_eigid}")
    plt.show()

# Example usage:
# plot_variance_trajectories(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj, patch_eigval, patch_size, patch_stride, savedir)

def plot_sample_pnts_mean_deviation_trajectories(step_slice, mean_x_sample_traj, train_X_mean, train_X_eigvec, train_X_eigval, 
                                     slice2plot, savedir, dataset_name="Gaussian"):
    train_X_mean_vec = train_X_mean.view(-1)
    mean_deviation_traj = (mean_x_sample_traj - train_X_mean_vec) @ train_X_eigvec.cpu()
    MSE_per_mode_traj = mean_deviation_traj.pow(2)
    MSE_per_mode_traj_normalized = MSE_per_mode_traj / train_X_eigval.cpu()
    ndim = train_X_eigval.shape[0]
    max_eigid = max(range(ndim)[slice2plot])    

    plt.figure()
    plt.plot(step_slice, mean_deviation_traj[:, slice2plot], alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Training step")
    plt.ylabel("mean deviation")
    plt.title(f"Mean deviation of learned samples on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {train_X_eigval[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_pnts_mean_dev_eigenbasis_traj_raw_top{max_eigid}")
    plt.show()

    plt.figure()
    plt.plot(step_slice, MSE_per_mode_traj[:, slice2plot], alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Squared error of mean deviation")
    plt.title(f"Mean deviation of learned samples on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {train_X_eigval[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_pnts_mean_SE_eigenbasis_traj_raw_top{max_eigid}")
    plt.show()

    plt.figure()
    plt.plot(step_slice, MSE_per_mode_traj_normalized[:, slice2plot], alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Squared error of mean deviation\n[normalized by target variance]")
    plt.title(f"Mean deviation of learned samples on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {train_X_eigval[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_pnts_mean_SE_eigenbasis_traj_normalized_top{max_eigid}")
    plt.show()
    

# from trajectory_convergence_lib import *
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def smooth_and_find_threshold_crossing(trajectory, threshold, first_crossing=False, smooth_sigma=2):
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.cpu().numpy()
    smoothed_trajectory = gaussian_filter1d(trajectory, sigma=smooth_sigma)
    # determine the direction of the crossing
    direction = 1 if smoothed_trajectory[0] > threshold else -1
    if direction == 1:
        crossing_indices = np.where(smoothed_trajectory < threshold)[0]
    else:
        crossing_indices = np.where(smoothed_trajectory > threshold)[0]
    if len(crossing_indices) > 0:
        return crossing_indices[0] if first_crossing else crossing_indices[-1], direction
    else:
        return None, direction



def smooth_and_find_range_crossing(trajectory, LB, UB, smooth_sigma=2):
    """
    Smooths the trajectory and finds the first crossing into the range [LB, UB].
    
    Parameters:
        trajectory (np.ndarray or torch.Tensor): The input trajectory data.
        LB (float or np.ndarray or torch.Tensor): Lower bound of the range.
        UB (float or np.ndarray or torch.Tensor): Upper bound of the range.
        smooth_sigma (float): Standard deviation for Gaussian kernel used in smoothing.
        
    Returns:
        crossing_index (int or None): The index where the trajectory first enters the range.
        direction (str or None): Direction of crossing ('upward' or 'downward').
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    if isinstance(LB, torch.Tensor):
        LB = LB.cpu().numpy()
    if isinstance(UB, torch.Tensor):
        UB = UB.cpu().numpy()
    
    # Smooth the trajectory
    smoothed_trajectory = gaussian_filter1d(trajectory, sigma=smooth_sigma)
    
    # Ensure LB <= UB
    if np.any(LB > UB):
        raise ValueError("Lower bound LB must be less than or equal to upper bound UB.")
    
    # Initialize direction and crossing_index
    crossing_index = None
    direction = None
    
    # Iterate through the trajectory to find the first crossing into [LB, UB]
    for i in range(1, len(smoothed_trajectory)):
        prev = smoothed_trajectory[i-1]
        current = smoothed_trajectory[i]
        
        # Check if previous point was outside the range
        was_below = prev < LB
        was_above = prev > UB
        was_inside = LB <= prev <= UB
        
        # Current point is inside the range
        is_inside = LB <= current <= UB
        
        if not is_inside and was_inside:
            # Exiting the range, not entering
            continue
        if is_inside and not was_inside:
            # Entering the range
            if was_below:
                direction = -1 # 'upward'
            elif was_above:
                direction = 1 # 'downward'
            else:
                # In case previous point was not strictly above or below
                direction = 'unknown'
            crossing_index = i
            break  # Stop after finding the first crossing
    
    return crossing_index, direction



def harmonic_mean(A, B):
    return 2 / (1 / A + 1 / B)


def compute_crossing_points(patch_eigval, diag_cov_x_patch_sample_true_eigenbasis_traj, step_slice, smooth_sigma=2, 
                            threshold_type="harmonic_mean", threshold_fraction=0.2):
    num_trajectories = diag_cov_x_patch_sample_true_eigenbasis_traj.shape[1]
    crossing_steps = []
    directions = []
    for i in range(num_trajectories):
        trajectory = diag_cov_x_patch_sample_true_eigenbasis_traj[:, i]
        if threshold_type == "range":
            threshold = np.array([patch_eigval[i] * (1 - threshold_fraction), patch_eigval[i] * (1 + threshold_fraction)])
            crossing_idx, direction = smooth_and_find_range_crossing(trajectory, threshold[0], threshold[1], smooth_sigma=smooth_sigma)
        else:
            if threshold_type == "harmonic_mean":
                threshold = harmonic_mean(patch_eigval[i], trajectory[0])
            elif threshold_type == "mean":
                threshold = (patch_eigval[i] + trajectory[0]) / 2
            elif threshold_type == "geometric_mean":
                threshold = np.sqrt(patch_eigval[i] * trajectory[0])
            crossing_idx, direction = smooth_and_find_threshold_crossing(trajectory, threshold, first_crossing=True, smooth_sigma=smooth_sigma)
        if crossing_idx is not None:
            crossing_steps.append(step_slice[crossing_idx])
            directions.append(direction)
        else:
            print(f"No crossing found for mode {i}")
            crossing_steps.append(np.nan)
            directions.append(0)
    df = pd.DataFrame({"Variance": patch_eigval.cpu().numpy(), "emergence_step": crossing_steps, "direction": directions})
    # translate direction 1 -> decrease, -1 -> increase
    df["Direction"] = df["direction"].map({1: "decrease", -1: "increase"})
    return df
