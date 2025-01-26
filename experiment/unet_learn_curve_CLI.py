# %%
import sys
import os
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
from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone
from core.toy_shape_dataset_lib import generate_random_star_shape_torch
from core.diffusion_basics_lib import *
from core.diffusion_edm_lib import *
from core.network_edm_lib import SongUNet, DhariwalUNet

from circuit_toolkit.plot_utils import saveallforms, to_imgrid, show_imgrid


# %% [markdown]
# ### Utils functions

# %%
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# %%
class EDMCNNPrecondWrapper(nn.Module):
    def __init__(self, model, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
    def forward(self, X, sigma, cond=None, ):
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        ## https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
        # unsqueze sigma to have same dimension as X (which may have 2-4 dim) 
        sigma_vec = sigma.view([-1, ] + [1, ] * (X.ndim - 1))
        c_skip = self.sigma_data ** 2 / (sigma_vec ** 2 + self.sigma_data ** 2)
        c_out = sigma_vec * self.sigma_data / (sigma_vec ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma_vec ** 2).sqrt()
        c_noise = sigma.log() / 4
        model_out = self.model(c_in * X, c_noise.view(-1), cond=cond) # this is required for EDM Unet model. 
        return c_skip * X + c_out * model_out

# %%
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

# %% [markdown]


# %% [markdown]
# ### Convolutional Architecture
from pprint import pprint
import argparse
from typing import List, Tuple
def parse_range(range_str: List[str]) -> Tuple[int, int, int]:
    """
    Parses a list of strings into a tuple of three integers representing a range.

    Args:
        range_str (List[str]): List containing start, end, and step as strings.

    Returns:
        Tuple[int, int, int]: Parsed (start, end, step).

    Raises:
        argparse.ArgumentTypeError: If the input is invalid.
    """
    if len(range_str) != 3:
        raise argparse.ArgumentTypeError("Each range must have exactly three integers: start end step.")
    try:
        start, end, step = map(int, range_str)
    except ValueError:
        raise argparse.ArgumentTypeError("All range values must be integers.")
    if start >= end:
        raise argparse.ArgumentTypeError(f"Start ({start}) must be less than end ({end}).")
    if step <= 0:
        raise argparse.ArgumentTypeError(f"Step ({step}) must be a positive integer.")
    return (start, end, step)


def generate_record_times(ranges: List[Tuple[int, int, int]]) -> List[int]:
    """
    Generates a list of record times based on the provided ranges.

    Args:
        ranges (List[Tuple[int, int, int]]): List of ranges defined by (start, end, step).

    Returns:
        List[int]: Generated record times.
    """
    record_times = []
    for start, end, step in ranges:
        record_times.extend(range(start, end, step))
    return record_times


def parse_args():
    parser = argparse.ArgumentParser(description="UNet Learning Curve Experiment")
    parser.add_argument("--exp_name", type=str, default="MNIST_UNet_CNN_EDM", help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--nsteps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--layers_per_block", type=int, default=1, help="Layers per block")
    parser.add_argument("--model_channels", type=int, default=16, help="Model channels")
    parser.add_argument("--channel_mult", type=int, nargs='+', default=[1,2,3,4], help="Channel multiplier")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--decoder_init_attn", type=bool, default=False, help="Decoder initial attention")
    parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[], help="Attention resolutions")
    parser.add_argument("--eval_sample_size", type=int, default=1000, help="Evaluation sample size")
    parser.add_argument("--record_frequency", type=int, default=0, help="Evaluation sample frequency")
    parser.add_argument(
        '-r', '--record_step_range',
        metavar=('START', 'END', 'STEP'),
        type=int,
        nargs=3,
        action='append',
        default=[(0, 10, 1), (10, 50, 5), (50, 100, 10), (100, 500, 25), (500, 2500, 50), (2500, 5000, 100), (5000, 10000, 250)],
        help="Define a range with start, end, and step. Can be used multiple times. Evaluation sample frequency"
    )
    return parser.parse_args()

args = parse_args()

exp_name = args.exp_name
batch_size = args.batch_size
nsteps = args.nsteps
layers_per_block = args.layers_per_block
model_channels = args.model_channels
channel_mult = args.channel_mult
decoder_init_attn = args.decoder_init_attn
attn_resolutions = args.attn_resolutions
lr = args.lr
eval_sample_size = args.eval_sample_size
record_frequency = args.record_frequency
record_step_range = args.record_step_range
ranges = []
for r in record_step_range:
    try:
        parsed_range = parse_range(r)
        ranges.append(parsed_range)
    except argparse.ArgumentTypeError as e:
        raise argparse.ArgumentTypeError(str(e))
record_times = generate_record_times(ranges)
print(f"record_frequency: {record_frequency}")
print(f"record_step_range: {record_step_range}")
print(f"record_times: {record_times}")
saveroot = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve"
savedir = f"{saveroot}/{exp_name}"
os.makedirs(savedir, exist_ok=True)
device = get_device()

# ### Loading data
# load MNIST dataset, make it a B x 32 x 32 tensor
mnist_dataset = torchvision.datasets.MNIST(root='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Data', 
                                           train=True, download=True, 
                                           transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]))
mnist_Xtsr = torch.stack([mnist_dataset[i][0] for i in range(len(mnist_dataset))])
print(mnist_Xtsr.shape) # 60000 x 32 x 32

Xtsr = (mnist_Xtsr.to(device) - 0.5) / 0.5
pnts = Xtsr.view(Xtsr.shape[0], -1)
ndim = pnts.shape[1]
X_mean = pnts.mean(dim=0)
cov_empirical = torch.cov(pnts.T, correction=1)
# diagonalize the covariance matrix
cov_empirical_eigs, cov_empirical_evecs = torch.linalg.eigh(cov_empirical)
cov_empirical_eigs = cov_empirical_eigs.flip(0)
cov_empirical_evecs = cov_empirical_evecs.flip(1)
rot = cov_empirical_evecs
diag_var = cov_empirical_eigs
assert torch.allclose(rot @ torch.diag(diag_var) @ rot.T, cov_empirical, atol=5e-5)
pkl.dump({"diag_var": diag_var.cpu(), 
          "rot": rot.cpu(), 
          #   "cov": cov.cpu(), 
          "cov_empirical": cov_empirical.cpu(),
          "X_mean": X_mean.cpu(),
          "train_pnts": pnts.cpu()}, open(f"{savedir}/train_data_cov_info.pkl", "wb"))

# %%
sample_store = {}
loss_store = {}
def sampling_callback_fn(epoch, loss, model):
    loss_store[epoch] = loss
    noise_init = torch.randn(eval_sample_size, *imgshape).to(device)
    x_out, x_traj, x0hat_traj, t_steps = edm_sampler(model, noise_init, 
                    num_steps=20, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True)
    sample_store[epoch] = x_out.cpu(), # x_traj.cpu(), x0hat_traj.cpu(), t_steps.cpu()


device = get_device()
Xtsr = (mnist_Xtsr.to(device) - 0.5) / 0.5
pnts = Xtsr.view(Xtsr.shape[0], -1)
imgshape = Xtsr.shape[1:]
ndim = pnts.shape[1]
cov_empirical = torch.cov(pnts.T, correction=1)
print(f"MNIST dataset {pnts.shape[0]} samples, {ndim} features")
config = edict(
    channels=1,
    img_size=32,
    layers_per_block=layers_per_block,
    decoder_init_attn=decoder_init_attn,
    attn_resolutions=attn_resolutions,
    model_channels=model_channels,
    channel_mult=channel_mult,
    dropout=0.0,
    label_dim=0,
    augment_dim=0,
)
pprint(config)

json.dump(config, open(f"{savedir}/config.json", "w"))
unet = create_unet_model(config)
model_precd = EDMCNNPrecondWrapper(unet, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0)
edm_loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=0.5)
model_precd, loss_traj = train_score_model_custom_loss(Xtsr, model_precd, edm_loss_fn, 
                                    lr=lr, nepochs=nsteps, batch_size=batch_size, device=device, 
                                    callback=sampling_callback_fn, callback_freq=record_frequency, callback_step_list=record_times)

noise_init = torch.randn(1000, *imgshape).to(device)
x_out, x_traj, x0hat_traj, t_steps = edm_sampler(model_precd, noise_init, 
                num_steps=40, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True)

x_traj.shape
scaling = 1 / (t_steps ** 2 + 1).sqrt()
scaled_x_traj = (scaling[:, None, None, None, None] * x_traj).cpu()

mtg = to_imgrid(((x_out.cpu()[:64]+1)/2).clamp(0, 1), nrow=8, padding=1)
mtg.save(f"{savedir}/learned_samples_final.png")

# %%
pkl.dump(sample_store, open(f"{savedir}/sample_store.pkl", "wb"))
pkl.dump(loss_store, open(f"{savedir}/loss_store.pkl", "wb"))
torch.save(model_precd.model.state_dict(), f"{savedir}/model_final.pth")

# %% [markdown]
# #### Analyze the dynamics of spectral space

# %%
true_cov_eigs = diag_var.cpu()
mean_x_sample_traj = []
cov_x_sample_traj = []
diag_cov_x_sample_true_eigenbasis_traj = []
step_slice = [*sample_store.keys()]
for training_step in tqdm(step_slice):
    x_final, = sample_store[training_step]
    x_final = x_final.to(device).view(x_final.shape[0], -1)
    mean_x_sample = x_final.mean(dim=0)
    mean_x_sample_traj.append(mean_x_sample.cpu())
    cov_x_sample = torch.cov(x_final.T)
    # try estimate the variance along the eigenvector of the covariance matrix
    cov_x_sample_true_eigenbasis = rot.T @ cov_x_sample @ rot
    diag_cov_x_sample_true_eigenbasis = torch.diag(cov_x_sample_true_eigenbasis)
    diag_cov_x_sample_true_eigenbasis_traj.append(diag_cov_x_sample_true_eigenbasis.cpu())
    cov_x_sample_traj.append(cov_x_sample.cpu())
    
mean_x_sample_traj = torch.stack(mean_x_sample_traj).cpu()
cov_x_sample_traj = torch.stack(cov_x_sample_traj).cpu()
diag_cov_x_sample_true_eigenbasis_traj = torch.stack(diag_cov_x_sample_true_eigenbasis_traj).cpu()

pkl.dump({"diag_cov_x_sample_true_eigenbasis_traj": diag_cov_x_sample_true_eigenbasis_traj.cpu(), 
          "mean_x_sample_traj": mean_x_sample_traj.cpu(),
          "cov_x_sample_traj": cov_x_sample_traj.cpu(),
          "true_cov_eigs": true_cov_eigs.cpu(),
          "step_slice": step_slice}, open(f"{savedir}/sample_cov_true_eigenbasis_diag_traj.pkl", "wb"))

# %%
# plot the convergence of the mean 
mean_X_loss = (mean_x_sample_traj.cuda() - X_mean).pow(2).mean(dim=1)
plt.plot(step_slice, mean_X_loss.cpu().numpy(), label="MSE of mean")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("Mean square error")
plt.title("Convergence of the mean of learned samples")
saveallforms(savedir, "sample_mean_convergence_MSE_traj_loglog")
plt.show()

# %%
# Project the difference of mean on the true eigenbasis
mean_X_diff_true_basis = (mean_x_sample_traj.cuda() - X_mean) @ rot
MSE_X_diff_true_basis = mean_X_diff_true_basis.pow(2).cpu()
slice2plot = slice(None, 20, 2)
plt.plot(step_slice, MSE_X_diff_true_basis.numpy()[:, slice2plot], label="Eigenbasis difference", alpha=0.5)
plt.ylim(5E-3, None)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("Eigenbasis difference")
plt.title("Convergence of the mean of learned samples")
saveallforms(savedir, "sample_mean_convergence_eigenbasis_MSE_traj_top20_loglog")
plt.show()

# %%
# Project the difference of mean on the true eigenbasis
mean_X_diff_true_basis = (mean_x_sample_traj.cuda() - X_mean) @ rot
MSE_X_diff_true_basis = mean_X_diff_true_basis.pow(2).cpu()
slice2plot = slice(None, 100, 10)
plt.plot(step_slice, MSE_X_diff_true_basis.numpy()[:, slice2plot], label="Eigenbasis difference", alpha=0.5)
plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("MSE of mean ")
plt.title("Convergence of the mean of learned samples")
plt.gca().legend([f"Eig{i} = {true_cov_eigs[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
saveallforms(savedir, "sample_mean_convergence_eigenbasis_MSE_traj_top100")
plt.show()

# %%
true_cov_eigs = diag_var.cpu()
diag_cov_normalized = (diag_cov_x_sample_true_eigenbasis_traj / true_cov_eigs)
slice2plot = slice(None, 100, 10)
plt.plot(step_slice, diag_cov_x_sample_true_eigenbasis_traj[:, slice2plot], alpha=0.7)
for i, eigid in enumerate(range(ndim)[slice2plot]):
    plt.axhline(true_cov_eigs[eigid].item(), color=f"C{i}", linestyle="--", alpha=0.7)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Training step")
plt.ylabel("Variance")
plt.gca().legend([f"Eig{i} = {true_cov_eigs[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Variance of learned samples on true eigenbasis | MNIST")
saveallforms(savedir, "sample_cov_true_eigenbasis_diag_traj_raw_top100")
plt.show()

# %%
true_cov_eigs = diag_var.cpu()
diag_cov_normalized = (diag_cov_x_sample_true_eigenbasis_traj / true_cov_eigs)
slice2plot = slice(None, 500, 50)
plt.plot(step_slice, diag_cov_x_sample_true_eigenbasis_traj[:, slice2plot], alpha=0.7)
for i, eigid in enumerate(range(ndim)[slice2plot]):
    plt.axhline(true_cov_eigs[eigid].item(), color=f"C{i}", linestyle="--", alpha=0.7)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Training step")
plt.ylabel("Variance")
plt.gca().legend([f"Eig{i} = {true_cov_eigs[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Variance of learned samples on true eigenbasis | MNIST")
saveallforms(savedir, "sample_cov_true_eigenbasis_diag_traj_raw_top500")
plt.show()

# %%
true_cov_eigs = diag_var.cpu()
diag_cov_normalized = (diag_cov_x_sample_true_eigenbasis_traj / true_cov_eigs)
slice2plot = slice(None, 100, 10)
plt.plot(step_slice, diag_cov_normalized[:, slice2plot], alpha=0.7)
plt.axhline(1, color="k", linestyle="--", alpha=0.7)
# for i, eigid in enumerate(range(ndim)[slice2plot]):
#     plt.axhline(true_cov_eigs[eigid].item(), color=f"C{i}", linestyle="--", alpha=0.7)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Training step")
plt.ylabel("Variance [normalized by target variance]")
plt.gca().legend([f"Eig{i} = {true_cov_eigs[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Variance of learned samples on true eigenbasis | MNIST")
saveallforms(savedir, "sample_cov_true_eigenbasis_diag_traj_normalized_top100")
plt.show()

# %%
true_cov_eigs = diag_var.cpu()
diag_cov_normalized = (diag_cov_x_sample_true_eigenbasis_traj / true_cov_eigs)
slice2plot = slice(None, 500, 50)
plt.plot(step_slice, diag_cov_normalized[:, slice2plot], alpha=0.7)
plt.axhline(1, color="k", linestyle="--", alpha=0.7)
# for i, eigid in enumerate(range(ndim)[slice2plot]):
#     plt.axhline(true_cov_eigs[eigid].item(), color=f"C{i}", linestyle="--", alpha=0.7)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Training step")
plt.ylabel("Variance [normalized by target variance]")
plt.gca().legend([f"Eig{i} = {true_cov_eigs[i].item():.2f}" for i in range(ndim)[slice2plot]], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Variance of learned samples on true eigenbasis | MNIST")
saveallforms(savedir, "sample_cov_true_eigenbasis_diag_traj_normalized_top500")
plt.show()