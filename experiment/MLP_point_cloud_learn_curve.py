%load_ext autoreload
%autoreload 2
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/n/home12/binxuwang/Github/DiffusionLearningCurve")
sys.path.append("/Users/binxuwang/Github/DiffusionLearningCurve/")
from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone,UNetBlockStyleMLP_backbone_NoFirstNorm
from core.toy_shape_dataset_lib import generate_random_star_shape_torch
from core.diffusion_basics_lib import *
from core.diffusion_edm_lib import *
from core.img_patch_stats_analysis_lib import *
from core.gaussian_mixture_lib import GaussianMixture_torch
from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone_NoFirstNorm
from core.diffusion_esm_edm_lib import delta_GMM_score, delta_GMM_denoiser, EDMDeltaGMMScoreLoss
import os
import pickle as pkl
from circuit_toolkit.plot_utils import saveallforms
saveroot = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve"

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


savedir = join(saveroot, "DSM_vs_Delta_ESM_loss_highdim_GMM_128dim_512batch_data_cmp")
os.makedirs(savedir, exist_ok=True)

# Generate randomized gaussian mixture at ndim dimensions
ndim = 128  # Set desired dimensionality
npnts = 2048  # Set desired number of points
n_components = 3  # Number of Gaussian components

torch.manual_seed(42)
# Generate random means
mus = [torch.randn(ndim) * 2.0 for _ in range(n_components)]
# Generate random covariance matrices (positive definite)
covs = []
for _ in range(n_components):
    # Generate random matrix and make it positive definite
    A = torch.randn(ndim, ndim)
    cov = A @ A.T + torch.eye(ndim) * 0.1  # Add small diagonal term for numerical stability
    covs.append(cov)
# Generate random weights
weights = torch.rand(n_components) + 0.5  # weights between 0.5 and 1.5
weights = weights.tolist()
gmm = GaussianMixture_torch(mus, covs, weights)
pnts, _, _ = gmm.sample(npnts)
pnts = pnts / pnts.std() * 0.5
# For high-dimensional data, we can only plot first 2 dimensions
if ndim >= 2:
    density = gmm.pdf(pnts)
    plt.scatter(pnts[:, 0], pnts[:, 1], c=density, cmap="viridis", alpha=0.6)
    plt.xlabel(f"Dimension 0")
    plt.ylabel(f"Dimension 1")
    plt.title(f"GMM samples ({npnts} points, {ndim}D, showing first 2 dims)")
    plt.colorbar(label="Density")
    plt.show()
else:
    print(f"Generated {npnts} samples in {ndim}D space")
    

if os.path.exists(join(savedir, "training_data.pth")):
    training_data = torch.load(join(savedir, "training_data.pth"))
    pnts = training_data["pnts"]
    mus = training_data["mus"]
    covs = training_data["covs"]
    weights = training_data["weights"]
else:
    torch.save({"pnts": pnts, "mus": mus, "covs": covs, "weights": weights}, join(savedir, "training_data.pth"))

device = get_device()
sigma_data = pnts.std().item()
# torch.save({"pnts": pnts, "mus": mus, "covs": covs, "weights": weights}, join(savedir, "training_data.pth"))

nlayers = 6
nhidden = 192
time_embed_dim = 64
batch_size = 8192
nepochs = 100000
sample_store = {}
loss_store = {}
def sampling_callback_fn(epoch, loss, model, manual_seed=0):
    loss_store[epoch] = loss
    noise_init = torch.randn(2000, ndim, generator=torch.Generator(device=device).manual_seed(manual_seed), device=device)
    x_out = edm_sampler(model, noise_init, 
                    num_steps=20, sigma_min=0.002, sigma_max=80, rho=7, return_traj=False) # , x_traj, x0hat_traj, t_steps
    sample_store[epoch] = x_out.cpu()#,# x_traj.cpu(), x0hat_traj.cpu(), t_steps.cpu()

record_step_list = [*range(0, 100000, 1000)]#[500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000]
device = get_device()
ndim = pnts.shape[1]
cov_empirical = torch.cov(pnts.T, correction=1)
print(f"Point cloud dataset {pnts.shape[0]} samples, {ndim} features | batch size: {batch_size}, epochs: {nepochs} | nlayers: {nlayers}, nhidden: {nhidden}, time_embed_dim: {time_embed_dim}")
model_esm = UNetBlockStyleMLP_backbone_NoFirstNorm(ndim=ndim, nlayers=nlayers, nhidden=nhidden, time_embed_dim=time_embed_dim,)
model_precd_esm = EDMPrecondWrapper(model_esm, sigma_data=sigma_data, sigma_min=0.002, sigma_max=80, rho=7.0)
edm_delta_gmm_loss_fn = EDMDeltaGMMScoreLoss(train_Xmat=pnts.to(device), P_mean=-1.2, P_std=1.2, sigma_data=sigma_data)
model_precd_esm, loss_traj_esm = train_score_model_custom_loss(pnts, model_precd_esm, edm_delta_gmm_loss_fn, 
                                    lr=0.0001, nepochs=nepochs, batch_size=batch_size, device=device, 
                                    callback=sampling_callback_fn, callback_freq=0, 
                                    callback_step_list=record_step_list)
torch.save(loss_traj_esm, join(savedir, f"loss_traj_esm_B{batch_size}.pth"))
torch.save(sample_store, join(savedir, f"sample_store_esm_B{batch_size}.pth"))
torch.save(model_precd_esm, join(savedir, f"model_precd_esm_B{batch_size}.pth"))


sample_store = {}
loss_store = {}
def sampling_callback_fn(epoch, loss, model, manual_seed=0):
    loss_store[epoch] = loss
    noise_init = torch.randn(2000, ndim, generator=torch.Generator(device=device).manual_seed(manual_seed), device=device)
    x_out = edm_sampler(model, noise_init, 
                    num_steps=20, sigma_min=0.002, sigma_max=80, rho=7, return_traj=False) # , x_traj, x0hat_traj, t_steps
    sample_store[epoch] = x_out.cpu()#, x_traj.cpu(), x0hat_traj.cpu(), t_steps.cpu()
    
model_dsm = UNetBlockStyleMLP_backbone_NoFirstNorm(ndim=ndim, nlayers=nlayers, nhidden=nhidden, time_embed_dim=time_embed_dim,)
model_precd_dsm = EDMPrecondWrapper(model_dsm, sigma_data=sigma_data, sigma_min=0.002, sigma_max=80, rho=7.0)
edm_loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=sigma_data)
model_precd_dsm, loss_traj_dsm = train_score_model_custom_loss(pnts, model_precd_dsm, edm_loss_fn, 
                                    lr=0.0001, nepochs=nepochs, batch_size=batch_size, device=device, 
                                    callback=sampling_callback_fn, callback_freq=0, 
                                    callback_step_list=record_step_list)

torch.save(loss_traj_dsm, join(savedir, f"loss_traj_dsm_B{batch_size}.pth"))
torch.save(sample_store, join(savedir, f"sample_store_dsm_B{batch_size}.pth"))
torch.save(model_precd_dsm, join(savedir, f"model_precd_dsm_B{batch_size}.pth"))


