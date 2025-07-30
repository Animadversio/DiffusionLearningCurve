#%%
%load_ext autoreload
%autoreload 2
#%%
import sys
import os
from os.path import join
import json
import pickle as pkl
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import trange, tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
sys.path.append("/n/home12/binxuwang/Github/DiffusionLearningCurve")
sys.path.append("/Users/binxuwang/Github/DiffusionLearningCurve/")
from core.dataset_lib import load_dataset
from core.toy_shape_dataset_lib import generate_random_star_shape_torch
from core.diffusion_basics_lib import *
# from core.diffusion_edm_lib import *
from core.network_edm_lib import SongUNet, DhariwalUNet
from core.DiT_model_lib import *
from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone
from core.trajectory_convergence_lib import analyze_and_plot_variance, fit_regression_log_scale
from core.img_patch_stats_analysis_lib import compute_crossing_points, sweep_and_create_sample_store, process_img_mean_cov_statistics,\
     process_patch_mean_cov_statistics, plot_variance_trajectories, plot_mean_deviation_trajectories, \
     harmonic_mean, smooth_and_find_threshold_crossing
from core.trajectory_convergence_lib import analyze_and_plot_variance
from core.trajectory_convergence_lib import smooth_and_find_threshold_crossing, compute_crossing_points
from core.img_patch_stats_analysis_lib import plot_variance_trajectories
from circuit_toolkit.plot_utils import saveallforms, to_imgrid, show_imgrid
from pprint import pprint

#%%

Xtsr, imgsize = load_dataset("ffhq-32x32", normalize=True)
#%% Load in the Sample Stores FFHQ MLP 
synopsis_dir = r"/n/home12/binxuwang/Github/DiffusionLearningCurve/figures/models_synopsis"
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/"
expname = "FFHQ32_ResNet_CNN_EDM_1layers_wide128"
explist = glob.glob(join(exproot, "FFHQ32*"))
explist = [os.path.basename(exp) for exp in explist]
# for expname in explist:
for expname in [
    "FFHQ32_UNet_CNN_EDM_3blocks_1x_wide8_pilot_fixednorm",
    "FFHQ32_UNet_CNN_EDM_2blocks_1x_wide8_pilot_fixednorm", 
    "FFHQ32_UNet_CNN_EDM_1blocks_1x_wide8_pilot_fixednorm",
]:
# [
    # "FFHQ32_ResNet_CNN_EDM_1layers_wide128",
    # "FFHQ32_ResNet_CNN_EDM_2layers_wide128",
    # "FFHQ32_ResNet_CNN_EDM_3layers_wide128",
    # "FFHQ32_ResNet_CNN_EDM_1layers_wide256",
    # "FFHQ32_ResNet_CNN_EDM_3layers_wide256",
    # "FFHQ32_ResNet_CNN_EDM_1layers_wide4",
    # "FFHQ32_ResNet_CNN_EDM_1layers_wide8",
    # "FFHQ32_ResNet_CNN_EDM_1layers_wide32",
    # "FFHQ32_ResNet_CNN_EDM_1layers_wide6",
    # "FFHQ32_ResNet_CNN_EDM_1layers_wide12",
    # "FFHQ32_ResNet_CNN_EDM_1layers_wide16",
    # "FFHQ32_ResNet_CNN_EDM_3layers_wide6",
    # "FFHQ32_ResNet_CNN_EDM_2layers_wide6",
    # "FFHQ32_ResNet_CNN_EDM_5layers_wide6",
    # "FFHQ32_UNet_CNN_EDM_1blocks_1x_wide128_fixednorm",
    # "FFHQ32_UNet_CNN_EDM_1blocks_2x_wide128_fixednorm",
    # "FFHQ32_UNet_CNN_EDM_2blocks_2x_wide128_fixednorm",
    # "FFHQ32_UNet_CNN_EDM_3blocks_2x_wide128_fixednorm",
    # "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_fixednorm_saveckpt",
    # "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm",
    # "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm_DSM",
    # "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm_ESM",
    # "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_saveckpt_fewsample",
    # "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_saveckpt_fewsample_longtrain",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_B1024_fixseed_DSM",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_B1024_fixseed_ESM",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_B4096_fixseed_DSM",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_B4096_fixseed_ESM",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_DSM",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_ESM",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_fixseed_DSM",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_fixseed_ESM",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample",
    # "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample_longtrain",
    # "FFHQ32_DiT_P2_192D_3H_6L_EDM_pilot",
    # "FFHQ32_DiT_P2_192D_3H_6L_EDM_pilot_DSM",
    # "FFHQ32_DiT_P2_192D_3H_6L_EDM_pilot_ESM",
    # "FFHQ32_DiT_P2_192D_3H_6L_EDM_saveckpt_fewsample",
    # "FFHQ32_DiT_P2_192D_3H_6L_EDM_saveckpt_fewsample_longtrain",
    # "FFHQ32_DiT_P2_384D_6H_6L_EDM_pilot",
    # "FFHQ32_DiT_P2_384D_6H_6L_EDM_pilot_DSM",
    # "FFHQ32_DiT_P2_384D_6H_6L_EDM_pilot_ESM",
    # "FFHQ32_DiT_P2_384D_6H_6L_EDM_saveckpt_fewsample_longtrain",
    # "FFHQ32_DiT_P2_768D_12H_12L_EDM_pilot",
    # "FFHQ32_DiT_P2_768D_12H_6L_EDM_pilot",
    # "FFHQ32_DiT_P4_384D_6H_6L_EDM_pilot",
    # "FFHQ32_DiT_P4_768D_12H_12L_EDM_pilot",
    # "FFHQ32_DiT_P4_768D_12H_6L_EDM_pilot",
    # "FFHQ32_ResNet_CNN_EDM_1layers_wide128",
    # "FFHQ32_ResNet_CNN_EDM_1layers_wide256",
    # "FFHQ32_ResNet_CNN_EDM_2layers_wide128",
    # "FFHQ32_ResNet_CNN_EDM_3layers_wide128",
    # "FFHQ32_ResNet_CNN_EDM_3layers_wide256",
# FFHQ32_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm
# FFHQ32_fix_words_UNet_MLP_EDM_8L_3072D_lr1e-4
# FFHQ32_random_words_jitter_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm
# FFHQ32_random_words_jitter_UNet_MLP_EDM_8L_3072D_lr1e-4
# ]:
    try:
        print(expname)
        savedir = join(exproot, expname)
        figdir = join(savedir, "figures")
        os.makedirs(figdir, exist_ok=True)
        sampledir = join(savedir, "samples")
        sample_store = sweep_and_create_sample_store(sampledir)
        # save some samples to the synopsis dir
        step_slice = sorted(sample_store.keys())
        to_imgrid(((1 + sample_store[step_slice[-1]][:25]) * 0.5).clamp(0, 1), \
            ).save(join(synopsis_dir, f"{expname}_final_samples.png"))
        # save the samples to the savedir
        #%
        imgshape = (3, 32, 32)
        patch_size, patch_stride = 32, 1
        img_mean, img_cov, img_eigval, img_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj = \
            process_img_mean_cov_statistics(Xtsr, sample_store, savedir, device="cuda", imgshape=imgshape, save_pkl=False)
        torch.save({"step_slice": list(step_slice),
                    "img_mean": img_mean, 
                    "img_eigval": img_eigval, 
                    "img_eigvec": img_eigvec, 
                    "mean_x_sample_traj": mean_x_sample_traj, 
                    # "cov_x_sample_traj": cov_x_sample_traj, 
                    "diag_cov_x_sample_true_eigenbasis_traj": diag_cov_x_sample_true_eigenbasis_traj}, 
                   join(savedir, "sample_img_cov_true_eigenbasis_diag_traj_pruned.pth"))
        #%
        emer_time_df = compute_crossing_points(img_eigval.cpu(), diag_cov_x_sample_true_eigenbasis_traj.cpu(), range(len(step_slice)), smooth_sigma=1, threshold_type="geometric_mean")
        # find the eigen index that the initial variance and final variance are similar
        var_ratio = diag_cov_x_sample_true_eigenbasis_traj[0,:] / img_eigval.cpu()
        eig_idx_tooclose = (var_ratio > 0.5) & (var_ratio < 2)
        eig_idx_tooclose_vec = eig_idx_tooclose.nonzero()[:,0]
        plt.figure(figsize=(5.5, 5))
        plt.pcolor(diag_cov_x_sample_true_eigenbasis_traj.log().cpu().numpy().T, cmap="Spectral", rasterized=True)
        plt.colorbar(label="log(Variance)")
        plt.scatter(emer_time_df["emergence_step"].to_numpy(), emer_time_df.index.to_numpy(), marker=".", c="k", s=3, alpha=0.2, rasterized=True)
        plt.axhline(eig_idx_tooclose_vec[0], color="r", linestyle="--")
        plt.axhline(eig_idx_tooclose_vec[-1], color="r", linestyle="--")
        plt.title(f"Diagonal of Covariance Matrix in True Eigenbasis\n{expname}")
        plt.xlabel("Training Step")
        plt.ylabel("Eigenvector index")
        plt.gca().invert_yaxis()  # Flip the y-axis
        # annotate x-axis with step_slice
        # Only show a subset of ticks to avoid overcrowding
        tick_indices = np.linspace(0, len(step_slice)-1, 8, dtype=int)
        plt.xticks(tick_indices, [step_slice[i] for i in tick_indices], rotation=-30, ha="left")
        saveallforms([figdir, synopsis_dir], f"eigenframe_variance_heatmap_{expname}_rasterized", dpi=300)
        plt.show()
        #%
        emer_time_df_step = compute_crossing_points(img_eigval.cpu(), diag_cov_x_sample_true_eigenbasis_traj.cpu(), step_slice, smooth_sigma=1, threshold_type="geometric_mean")
        emer_time_df_step.to_csv(join(synopsis_dir, f"emergence_time_df_step_{expname}.csv"))
        init_var = diag_cov_x_sample_true_eigenbasis_traj[0,:]
        mean_init_var = init_var.mean().item()
        exclude_mask = (emer_time_df_step.Variance / mean_init_var > 0.5) & (emer_time_df_step.Variance / mean_init_var < 2)

        analyze_and_plot_variance(emer_time_df_step, x_col="emergence_step", y_col="Variance", hue_col="Direction", log_x=True, log_y=True, figsize=(5, 5), 
                                exclude_mask=exclude_mask, reverse_equation=True, fit_label_format='{direction} fit: $\\tau = {a:.2f} \lambda^{{{b:.2f}}}$', 
                                title=f'{expname}\nVariance vs Emergence Step with Fitted Lines', xlabel='Emergence Step', ylabel='Variance', 
                                alpha=0.6, annotate=False, annotate_offset=(0, 0), fit_line_kwargs=None, 
                                scatter_kwargs={"rasterized": True, "linewidth": 0.3}, 
                                ax=None,
                                )
        plt.axhline(0.5 * mean_init_var, color="k", linestyle="--")
        plt.axhline(2 * mean_init_var, color="k", linestyle="--")
        # fill the area between the two lines
        XLIM = plt.gca().get_xlim()
        plt.fill_betweenx([0.5 * mean_init_var, 2 * mean_init_var], [XLIM[0], XLIM[0]], [XLIM[1], XLIM[1]], 
                        color="gray", alpha=0.3, label="Var. within 0.5-2x of init.")
        print("Direction: increase",)
        df_split = emer_time_df_step[~exclude_mask].query("Direction == 'increase'")
        fit_dict_increase = fit_regression_log_scale(df_split.Variance, df_split.emergence_step)
        print("Direction: decrease",)
        df_split = emer_time_df_step[~exclude_mask].query("Direction == 'decrease'")
        fit_dict_decrease = fit_regression_log_scale(df_split.Variance, df_split.emergence_step)
        # annotate the R2 of the fit
        plt.text(0.05, 0.8, f"R2: {fit_dict_increase['r_squared']:.2f} (N={fit_dict_increase['N']})", transform=plt.gca().transAxes, ha="left", va="center", color="r")
        plt.text(0.05, 0.5, f"R2: {fit_dict_decrease['r_squared']:.2f} (N={fit_dict_decrease['N']})", transform=plt.gca().transAxes, ha="left", va="center", color="blue")
        plt.legend()
        saveallforms([figdir, synopsis_dir], f"convergence_time_vs_variance_scaling_{expname}_rasterized", dpi=300)
        plt.show()
        #%
        plt.figure(figsize=(6, 6))
        for step_id in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, len(step_slice)-1]:
            if step_id < len(step_slice):
                plt.plot(img_eigval.cpu().numpy(), 
                        diag_cov_x_sample_true_eigenbasis_traj.cpu().numpy()[step_id,:], 
                        label=f"step {step_slice[step_id]}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Eigenvalue")
        plt.ylabel("Variance")
        plt.title(f"Variance of Eigenprojected Samples\n{expname}")
        plt.legend()
        saveallforms([figdir, synopsis_dir], f"generated_spectral_evolution_{expname}", dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error for {expname}: {e}")
        continue
    #%%