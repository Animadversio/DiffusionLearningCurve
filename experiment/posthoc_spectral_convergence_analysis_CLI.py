
import os
from os.path import join
import pickle as pkl
import json
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import sys
sys.path.append("/n/home12/binxuwang/Github/DiffusionLearningCurve")
from core.img_patch_stats_analysis_lib import compute_crossing_points, sweep_and_create_sample_store, process_img_mean_cov_statistics,\
     process_patch_mean_cov_statistics, plot_variance_trajectories, plot_mean_deviation_trajectories, \
     harmonic_mean, smooth_and_find_threshold_crossing
from core.trajectory_convergence_lib import analyze_and_plot_variance
from core.dataset_lib import load_dataset
from circuit_toolkit.plot_utils import saveallforms

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Post-hoc analysis of diffusion model training')
    parser.add_argument('--expname', type=str, required=True,
                        help='Experiment name (e.g., FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm)')
    parser.add_argument('--dataset', type=str, default="ffhq-32x32",
                        help='Dataset name (e.g., ffhq-32x32, afhq-32x32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate used during training (default: 1e-4)')
    return parser.parse_args()

# cnn_experiments = [
# #     ("FFHQ", "FFHQ_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
# #     ("FFHQ_fix_words", "FFHQ_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
# #     ("FFHQ_random_words_jitter", "FFHQ_random_words_jitter_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
#     ("ffhq-32x32", "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
#     ("ffhq-32x32-fix_words", "FFHQ32_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
#     ("ffhq-32x32-random_word_jitter", "FFHQ32_random_words_jitter_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
#     ("afhq-32x32", "AFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
#     # ("CIFAR", "CIFAR_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm")
# ]

args = parse_args()
dataset_name = args.dataset
expname = args.expname
lr = args.lr
exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/"
try:
    train_run_args = json.load(open(join(exproot, expname, "args.json"), "r"))
    lr = train_run_args["lr"]
    dataset_name = train_run_args["dataset_name"]
except FileNotFoundError:
    print(f"args.json file not found for {expname} using default lr 1e-4") 
    lr = 1e-4


print(f"Processing experiment: {expname} with dataset {dataset_name} [lr: {lr}]")
savedir = join(exproot, expname)
figdir = join(savedir, "figures")
os.makedirs(figdir, exist_ok=True)
sampledir = join(savedir, "samples")

Xtsr, imgsize = load_dataset(dataset_name, normalize=True)
sample_store = sweep_and_create_sample_store(sampledir)
imgshape = list(Xtsr.shape[1:]) #(3, imgsize, imgsize)
patch_size, patch_stride = imgsize, 1
step_slice = sorted(sample_store.keys())
print(f"Saved samples at steps: {step_slice}")

img_mean, img_cov, img_eigval, img_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj = \
        process_img_mean_cov_statistics(Xtsr, sample_store, savedir, device="cuda", imgshape=imgshape, save_pkl=False)
pkl.dump({"step_slice": step_slice, "img_mean": img_mean, "img_cov": img_cov, "img_eigval": img_eigval, "img_eigvec": img_eigvec, 
            "mean_x_sample_traj": mean_x_sample_traj, "diag_cov_x_sample_true_eigenbasis_traj": diag_cov_x_sample_true_eigenbasis_traj}, 
            open(join(savedir, f"{dataset_name}_img_mean_cov_statistics.pkl"), "wb"))


for slice2plot in [slice(0, 9, 1), slice(0, 30, 3), slice(0, 100, 10), slice(5, 100, 10), slice(2, 500, 50), slice(0, 1000, 100), slice(2, 3000, 300)]:
    plot_variance_trajectories(step_slice, diag_cov_x_sample_true_eigenbasis_traj, img_eigval.cpu(), slice2plot,
                                patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name=dataset_name)
    plt.close("all")

for slice2plot in [slice(0, 9, 1), slice(0, 30, 3), slice(0, 100, 10), slice(5, 100, 10), slice(2, 500, 50), slice(0, 1000, 100), slice(2, 3000, 300)]:
    plot_mean_deviation_trajectories(step_slice, mean_x_sample_traj, img_mean, img_eigvec.cpu(), img_eigval.cpu(), 
            slice2plot, patch_size, patch_stride, savedir=figdir, dataset_name=dataset_name)
    plt.close("all")
    
lr_step_slice = np.array(step_slice) * lr
for threshold_type in ["harmonic_mean", "geometric_mean", ]:
    df = compute_crossing_points(img_eigval.cpu(), diag_cov_x_sample_true_eigenbasis_traj, 
                                    lr_step_slice, smooth_sigma=1, threshold_type=threshold_type, )
    df.to_csv(f"{figdir}/synopsis_image_eigenmode_emergence_{threshold_type}_vs_variance.csv", index=False)
    figh = analyze_and_plot_variance(df, x_col='emergence_step', y_col='Variance', 
                    hue_col='Direction', palette={"increase": "red", "decrease": "blue"}, 
                    log_x=True, log_y=True, figsize=(6, 6), fit_label_format='{direction} fit: $x = {a:.1e}y^{{{b:.2f}}}$', 
                    reverse_equation=True, annotate=False, annotate_offset=(0, 0), 
                    title=f'Variance vs Emergence Time | {dataset_name}', 
                    xlabel=f'Mode emergence step * lr | {threshold_type}', ylabel='Eigenmode variance', alpha=0.5, fit_line_kwargs=None, scatter_kwargs=None, ax=None)
    saveallforms(figdir, f"synopsis_image_eigenmode_emergence_{threshold_type}_vs_variance_fitline_lr_reverse")



steps = [1, 5, 10, 50, 100, 500, 1500, 5000, 10000, 50000]
for step in steps:
    if step not in sample_store:
        print(f"Step {step} not in sample store")
        continue
    print(f"Processing step {step}, shape: {sample_store[step].shape}")
    
    # Create single image sample
    samples_temp = sample_store[step][:1].view(-1, 3, imgsize, imgsize)
    samples_temp = ((samples_temp + 1) / 2).clamp(0, 1)
    montage = make_grid(samples_temp, nrow=8)
    ToPILImage()(montage).save(join(figdir, f"sample_step_{step}_single.png"))
    
    samples_temp = sample_store[step][:64].view(-1, 3, imgsize, imgsize)
    samples_temp = ((samples_temp + 1) / 2).clamp(0, 1)
    montage = make_grid(samples_temp, nrow=8)
    ToPILImage()(montage).save(join(figdir, f"sample_step_{step}.png"))
# except Exception as e:
#     print(f"Error processing {expname}: {e}")

# try:
#     del Xtsr, img_mean, img_cov, img_eigval, img_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj
#     del sample_store
# except NameError:
#     pass
