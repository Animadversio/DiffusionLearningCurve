#!/usr/bin/env python3
"""
Ground Truth Score vs Denoising Score Matching Comparison CLI

This script compares two approaches for learning score functions:
1. Ground Truth Score Learning: Direct supervision with analytical scores
2. Denoising Score Matching: Standard EDM training objective

Usage:
    python experiment/ground_truth_vs_denoising_score_CLI.py --exp_name comparison_test --n_samples 5000 --nepochs 1000
"""

import sys
import os
sys.path.append("/n/home12/binxuwang/Github/DiffusionLearningCurve")

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from core.gaussian_mixture_lib import GaussianMixture
from core.diffusion_edm_lib import (
    UNetBlockStyleMLP_backbone, 
    EDMPrecondWrapper,
    EDMLoss,
    train_score_model_custom_loss,
    edm_sampler
)


class GroundTruthScoreLoss:
    """Loss function for direct score matching against analytical ground truth"""
    def __init__(self, gmm):
        self.gmm = gmm
    
    def __call__(self, model, X):
        X_np = X.detach().cpu().numpy()
        true_scores = self.gmm.score(X_np)
        true_scores_torch = torch.tensor(true_scores, dtype=torch.float32, device=X.device)
        
        # Dummy time input (not used for ground truth score)
        t_dummy = torch.zeros(X.shape[0], device=X.device)
        pred_scores = model(X, t_dummy)
        
        loss = F.mse_loss(pred_scores, true_scores_torch, reduction='none')
        return loss


def langevin_sampler(score_model, n_samples, n_steps=1000, step_size=0.01, init_noise_scale=2.0, device='cpu'):
    """Simple Langevin dynamics sampler for ground truth score model"""
    with torch.no_grad():
        # Initialize with noise
        x = torch.randn(n_samples, 2, device=device) * init_noise_scale
        t_dummy = torch.zeros(n_samples, device=device)
        
        for i in tqdm(range(n_steps), desc="Langevin sampling"):
            score = score_model(x, t_dummy)
            noise = torch.randn_like(x) * np.sqrt(2 * step_size)
            x = x + step_size * score + noise
            
    return x.cpu().numpy()


def compute_sample_metrics(true_samples, gen_samples):
    """Compute various metrics to compare sample quality"""
    metrics = {}
    
    # 1. Mean and covariance comparison
    true_mean = np.mean(true_samples, axis=0)
    gen_mean = np.mean(gen_samples, axis=0)
    metrics['mean_error'] = float(np.linalg.norm(true_mean - gen_mean))
    
    true_cov = np.cov(true_samples.T)
    gen_cov = np.cov(gen_samples.T)
    metrics['cov_frobenius_error'] = float(np.linalg.norm(true_cov - gen_cov, 'fro'))
    
    # 2. Wasserstein distances (1D marginals)
    metrics['wasserstein_x'] = float(wasserstein_distance(true_samples[:, 0], gen_samples[:, 0]))
    metrics['wasserstein_y'] = float(wasserstein_distance(true_samples[:, 1], gen_samples[:, 1]))
    
    # 3. Nearest neighbor distances (coverage)
    dists_true_to_gen = cdist(true_samples, gen_samples)
    min_dists_coverage = np.min(dists_true_to_gen, axis=1)
    metrics['coverage_mean'] = float(np.mean(min_dists_coverage))
    
    # 4. Precision (how close generated samples are to true samples)
    dists_gen_to_true = cdist(gen_samples, true_samples)
    min_dists_precision = np.min(dists_gen_to_true, axis=1)
    metrics['precision_mean'] = float(np.mean(min_dists_precision))
    
    return metrics


def create_gaussian_mixture(mixture_type='default'):
    """Create a Gaussian mixture model"""
    if mixture_type == 'default':
        mus = [np.array([-2.0, -1.0]), np.array([2.0, 1.0]), np.array([0.0, 2.5])]
        covs = [np.array([[0.8, 0.2], [0.2, 0.8]]), 
                np.array([[1.2, -0.4], [-0.4, 1.2]]),
                np.array([[0.6, 0.0], [0.0, 0.6]])]
        weights = [0.4, 0.4, 0.2]
    elif mixture_type == 'simple':
        mus = [np.array([-1.5, 0.0]), np.array([1.5, 0.0])]
        covs = [np.array([[0.5, 0.0], [0.0, 0.5]]), 
                np.array([[0.5, 0.0], [0.0, 0.5]])]
        weights = [0.5, 0.5]
    elif mixture_type == 'complex':
        mus = [np.array([-2.0, -2.0]), np.array([2.0, 2.0]), 
               np.array([-2.0, 2.0]), np.array([2.0, -2.0]), 
               np.array([0.0, 0.0])]
        covs = [np.array([[0.4, 0.1], [0.1, 0.4]]) for _ in range(5)]
        weights = [0.2] * 5
    else:
        raise ValueError(f"Unknown mixture type: {mixture_type}")
    
    return GaussianMixture(mus, covs, weights)


def plot_comparison(X_train, gt_samples, dsm_samples, output_dir):
    """Plot comparison of original data and generated samples"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    axes[0].scatter(X_train[:, 0], X_train[:, 1], alpha=0.6, s=10)
    axes[0].set_title('Original Training Data')
    axes[0].axis('equal')
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-4, 5)
    
    # Ground truth model samples
    axes[1].scatter(gt_samples[:, 0], gt_samples[:, 1], alpha=0.6, s=10, color='red')
    axes[1].set_title('Ground Truth Model Samples')
    axes[1].axis('equal')
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-4, 5)
    
    # Denoising model samples
    axes[2].scatter(dsm_samples[:, 0], dsm_samples[:, 1], alpha=0.6, s=10, color='green')
    axes[2].set_title('Denoising Model Samples')
    axes[2].axis('equal')
    axes[2].set_xlim(-5, 5)
    axes[2].set_ylim(-4, 5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_trajectories(gt_loss_traj, dsm_loss_traj, output_dir):
    """Plot training loss trajectories"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(gt_loss_traj, label='Ground Truth Score Loss', alpha=0.8)
    plt.plot(dsm_loss_traj, label='Denoising Score Matching Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.yscale('log')
    
    # Skip first 100 epochs for clearer view
    start_epoch = min(100, len(gt_loss_traj) // 4)
    plt.subplot(1, 2, 2)
    plt.plot(gt_loss_traj[start_epoch:], label='Ground Truth Score Loss', alpha=0.8)
    plt.plot(dsm_loss_traj[start_epoch:], label='Denoising Score Matching Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (after epoch {start_epoch})')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Ground Truth vs Denoising Score Matching Comparison')
    parser.add_argument('--exp_name', type=str, required=True, 
                        help='Experiment name for output directory')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of training samples to generate')
    parser.add_argument('--nepochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--nlayers', type=int, default=4,
                        help='Number of MLP layers')
    parser.add_argument('--nhidden', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--mixture_type', type=str, default='default',
                        choices=['default', 'simple', 'complex'],
                        help='Type of Gaussian mixture to use')
    parser.add_argument('--n_gen_samples', type=int, default=2000,
                        help='Number of samples to generate for evaluation')
    parser.add_argument('--langevin_steps', type=int, default=500,
                        help='Number of Langevin sampling steps')
    parser.add_argument('--edm_steps', type=int, default=50,
                        help='Number of EDM sampling steps')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: outputs/{exp_name})')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = f"outputs/{args.exp_name}"
    else:
        output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    config = vars(args)
    config['device'] = str(device)
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Starting experiment: {args.exp_name}")
    print(f"Output directory: {output_dir}")
    
    # Create Gaussian mixture model
    print(f"Creating {args.mixture_type} Gaussian mixture...")
    gmm = create_gaussian_mixture(args.mixture_type)
    
    # Generate training data
    print(f"Generating {args.n_samples} training samples...")
    X_train, components, _ = gmm.sample(args.n_samples)
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    
    # Train Ground Truth Score Model
    print("Training Ground Truth Score Model...")
    gt_model = UNetBlockStyleMLP_backbone(
        ndim=2, nlayers=args.nlayers, nhidden=args.nhidden, time_embed_dim=32
    )
    gt_loss_fn = GroundTruthScoreLoss(gmm)
    
    gt_model_trained, gt_loss_traj = train_score_model_custom_loss(
        X_train_torch, gt_model, gt_loss_fn,
        lr=args.lr, nepochs=args.nepochs, batch_size=args.batch_size, device=device
    )
    
    print(f"Ground truth model final loss: {gt_loss_traj[-1]:.6f}")
    
    # Train Denoising Score Matching Model
    print("Training Denoising Score Matching Model...")
    dsm_model = UNetBlockStyleMLP_backbone(
        ndim=2, nlayers=args.nlayers, nhidden=args.nhidden, time_embed_dim=32
    )
    dsm_model_precd = EDMPrecondWrapper(dsm_model, sigma_data=0.5)
    edm_loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=0.5)
    
    dsm_model_trained, dsm_loss_traj = train_score_model_custom_loss(
        X_train_torch, dsm_model_precd, edm_loss_fn,
        lr=args.lr, nepochs=args.nepochs, batch_size=args.batch_size, device=device
    )
    
    print(f"Denoising model final loss: {dsm_loss_traj[-1]:.6f}")
    
    # Generate samples
    print("Generating samples...")
    print("  - Ground Truth Model (Langevin sampling)...")
    gt_samples = langevin_sampler(
        gt_model_trained, args.n_gen_samples, 
        n_steps=args.langevin_steps, device=device
    )
    
    print("  - Denoising Model (EDM sampling)...")
    with torch.no_grad():
        noise_init = torch.randn(args.n_gen_samples, 2, device=device)
        dsm_samples = edm_sampler(
            dsm_model_trained, noise_init, 
            num_steps=args.edm_steps, sigma_min=0.002, sigma_max=80, rho=7
        ).cpu().numpy()
    
    # Compute metrics
    print("Computing evaluation metrics...")
    gt_metrics = compute_sample_metrics(X_train, gt_samples)
    dsm_metrics = compute_sample_metrics(X_train, dsm_samples)
    
    # Save results
    results = {
        'gt_final_loss': float(gt_loss_traj[-1]),
        'dsm_final_loss': float(dsm_loss_traj[-1]),
        'gt_metrics': gt_metrics,
        'dsm_metrics': dsm_metrics,
        'gt_loss_trajectory': [float(x) for x in gt_loss_traj],
        'dsm_loss_trajectory': [float(x) for x in dsm_loss_traj]
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save samples
    np.save(f"{output_dir}/training_data.npy", X_train)
    np.save(f"{output_dir}/gt_samples.npy", gt_samples)
    np.save(f"{output_dir}/dsm_samples.npy", dsm_samples)
    
    # Save models
    torch.save(gt_model_trained.state_dict(), f"{output_dir}/gt_model.pth")
    torch.save(dsm_model_trained.state_dict(), f"{output_dir}/dsm_model.pth")
    
    # Generate plots
    print("Generating plots...")
    plot_comparison(X_train, gt_samples, dsm_samples, output_dir)
    plot_loss_trajectories(gt_loss_traj, dsm_loss_traj, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'GT Model':<15} {'DSM Model':<15}")
    print("-"*60)
    print(f"{'Final Loss':<25} {gt_loss_traj[-1]:<15.6f} {dsm_loss_traj[-1]:<15.6f}")
    for key in gt_metrics.keys():
        print(f"{key:<25} {gt_metrics[key]:<15.6f} {dsm_metrics[key]:<15.6f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("✓ Training complete!")


if __name__ == "__main__":
    main()