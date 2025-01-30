
from glob import glob
import re
import os
import torch
import numpy as np
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

def extract_patches(images, patch_size, patch_stride):
    B, C, H, W = images.shape
    patches = images.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, patch_size, patch_size)
    return patches


def process_patch_mean_cov_statistics(train_images, sample_store, savedir, patch_size=8, patch_stride=4, device="cuda", imgshape=(3, 64, 64)):
    # images = Xtsr.view(Xtsr.shape[0], *imgshape)
    patches = extract_patches(train_images, patch_size=patch_size, patch_stride=patch_stride)
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
        x_final_patches = extract_patches(x_final.view(x_final.shape[0], *imgshape), patch_size=patch_size, patch_stride=patch_stride)
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

    pkl.dump({
        "diag_cov_x_patch_sample_true_eigenbasis_traj": diag_cov_x_patch_sample_true_eigenbasis_traj, 
        "mean_x_patch_sample_traj": mean_x_patch_sample_traj,
        "cov_x_patch_sample_traj": cov_x_patch_sample_traj,
        "patch_mean": patch_mean.cpu(),
        "patch_cov": patch_cov.cpu(),
        "patch_eigval": patch_eigval.cpu(),
        "patch_eigvec": patch_eigvec.cpu(),
        "step_slice": step_slice
    }, open(f"{savedir}/sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}_cov_true_eigenbasis_diag_traj.pkl", "wb"))
    print(f"Saved to {savedir}/sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}_cov_true_eigenbasis_diag_traj.pkl")
    return patch_mean, patch_cov, patch_eigval, patch_eigvec, mean_x_patch_sample_traj, cov_x_patch_sample_traj, diag_cov_x_patch_sample_true_eigenbasis_traj
    
# Example usage:
# process_patch_statistics(Xtsr, sample_store, savedir, device)

def plot_variance_trajectories(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj, patch_eigval, slice2plot,
                               patch_size, patch_stride, savedir, dataset_name="FFHQ64"):
    ndim = patch_eigval.shape[0]
    max_eigid = max(range(ndim)[slice2plot])    
    plt.figure()
    plt.plot(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj[:, slice2plot], alpha=0.7)
    for i, eigid in enumerate(range(ndim)[slice2plot]):
        plt.axhline(patch_eigval[eigid].item(), color=f"C{i}", linestyle="--", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Variance")
    plt.title(f"Variance of learned patches ({patch_size}x{patch_size}, stride={patch_stride}) on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {patch_eigval[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}_cov_true_eigenbasis_diag_traj_raw_top{max_eigid}")
    plt.show()

    plt.figure()
    diag_cov_x_patch_sample_true_eigenbasis_traj_normalized = diag_cov_x_patch_sample_true_eigenbasis_traj / patch_eigval
    plt.plot(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj_normalized[:, slice2plot], alpha=0.7)
    plt.axhline(1, color="k", linestyle="--", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Variance [normalized by target variance]")
    plt.title(f"Variance of learned patches ({patch_size}x{patch_size}, stride={patch_stride}) on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {patch_eigval[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
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
    max_eigid = max(range(ndim)[slice2plot])    

    plt.figure()
    plt.plot(step_slice, mean_deviation_traj[:, slice2plot], alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Training step")
    plt.ylabel("mean deviation")
    plt.title(f"Mean deviation of learned patches ({patch_size}x{patch_size}, stride={patch_stride}) on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {patch_eigval[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}_mean_dev_eigenbasis_traj_raw_top{max_eigid}")
    plt.show()

    plt.figure()
    plt.plot(step_slice, MSE_per_mode_traj[:, slice2plot], alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Squared error of mean deviation")
    plt.title(f"Mean deviation of learned patches ({patch_size}x{patch_size}, stride={patch_stride}) on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {patch_eigval[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
    saveallforms(savedir, f"sample_patch_{patch_size}x{patch_size}_stride_{patch_stride}_mean_SE_eigenbasis_traj_raw_top{max_eigid}")
    plt.show()

    plt.figure()
    plt.plot(step_slice, MSE_per_mode_traj_normalized[:, slice2plot], alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Squared error of mean deviation\n[normalized by target variance]")
    plt.title(f"Mean deviation of learned patches ({patch_size}x{patch_size}, stride={patch_stride}) on true eigenbasis | {dataset_name}")
    plt.gca().legend([f"Eig{i} = {patch_eigval[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
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