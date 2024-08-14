#%%
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm, trange
#%%
# Utility functions
def generate_multivariate_gaussian(n_sample, n_dim, Lambda, U, mu, device='cpu'):
    """
    Generate multivariate Gaussian data using PyTorch.
    """
    Z = th.randn(n_sample, n_dim, device=device)
    samples = (Z * th.sqrt(Lambda)) @ U.t() + mu
    return samples


def random_orthogonal(n):
    q, r = th.linalg.qr(th.randn(n, n))
    return q, r


def generate_power_law_sequence(size, alpha, x_min):
    """
    Generate a random sequence following a power law distribution.
    """
    r = np.random.rand(size)
    x = x_min * (1 - r) ** (-1 / (alpha - 1))
    return x


# Visualization function
def plot_results(stats_df, W_dev_per_mode_col):
    figh1, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.flatten()
    for i in range(4):
        sns.lineplot(stats_df, x='step', y='loss', ax=axs[i], label='loss')
        sns.lineplot(stats_df, x='step', y='D_dev', ax=axs[i], label='D_dev')
        sns.lineplot(stats_df, x='step', y='W_dev', ax=axs[i], label='W_dev')
        if i % 2 == 1:
            axs[i].set_xscale('log')
        if i >= 2:
            axs[i].set_yscale('log')
        plt.legend()
    plt.suptitle(f'General loss and Weight deviations from optimal')
    plt.tight_layout()
    
    figh2, axs = plt.subplots(1, 4, figsize=(18, 4.5))
    axs = axs.flatten()
    for i, plot_func in enumerate([plt.plot, plt.semilogy, plt.semilogx, plt.loglog]):
        plt.sca(axs[i])
        for eigenN in [0, 5, 10, 20, 50, 100, 200, 300, 400, 500]:
            plot_func(stats_df['step'], W_dev_per_mode_col[:, eigenN], label=f'Mode {eigenN}')
        plt.xlabel('Step')
        plt.ylabel('Deviation')
        plt.legend()
    plt.suptitle('Deviation of W to optimal W^* per mode')
    plt.tight_layout()
    plt.show()
    return figh1, figh2


def get_Gaussian_Cov(n_dim, ):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    U, _ = random_orthogonal(n_dim) 
    U = U.to(device)
    Lambda = np.random.exponential(scale=1.0, size=(n_dim, ))
    Lambda = sorted(Lambda, reverse=True)
    Lambda = th.tensor(Lambda).float().to(device)
    Cov = U @ th.diag(Lambda) @ U.T
    return Cov, U, Lambda

# Training function
def train_model(denoiser, U, Lambda, mu, sigma, batch_size, total_steps, lr, optim_fun):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    # U, _ = random_orthogonal(n_dim)
    # U = U.to(device)
    # Lambda = np.random.exponential(scale=1.0, size=(n_dim, ))
    # Lambda = sorted(Lambda, reverse=True)
    # Lambda = th.tensor(Lambda).float().to(device)
    # mu = th.zeros(n_dim).to(device)
    Cov = U @ th.diag(Lambda) @ U.T

    W_star_solu = Cov @ th.inverse(Cov + sigma ** 2 * th.eye(n_dim).to(device))
    b_star_solu = mu - mu @ W_star_solu 

    optimizer = optim_fun(denoiser.parameters(), lr=lr)
    stats_traj = []
    W_dev_per_mode_col = []

    for step in trange(total_steps):
        samples = generate_multivariate_gaussian(batch_size, n_dim, Lambda, U, mu, device=device)
        noised_samples = samples + sigma * th.randn_like(samples)
        
        Dx = denoiser(noised_samples)
        
        loss = th.mean((Dx - samples) ** 2, dim=(0, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with th.no_grad():
            W_eff = denoiser.get_W_eff()
            W_deviation = (W_star_solu - W_eff).pow(2).mean().sqrt()
            W_deviation_per_mode = ((W_star_solu - W_eff) @ U).pow(2).sum(dim=0).sqrt()
            b_deviation = (b_star_solu - denoiser.b).pow(2).mean().sqrt()
            denoiser_ideal = noised_samples @ W_star_solu + b_star_solu
            D_deviation = (denoiser_ideal - Dx).pow(2).mean().sqrt()
        
        if (step + 1) % 1000 == 0 or step == 0:
            print(f'Step {step + 1}, Loss {loss.item():.4f} W dev {W_deviation.item():.4f} b dev {b_deviation.item():.4f} D dev {D_deviation.item():.4f}')
        stats_traj.append({"step": step + 1, 
                           "loss": loss.item(), 
                           "W_dev": W_deviation.item(), 
                           "b_dev": b_deviation.item(), 
                           "D_dev": D_deviation.item()})
        W_dev_per_mode_col.append(W_deviation_per_mode.cpu().numpy())
    
    stats_df = pd.DataFrame(stats_traj)
    W_dev_per_mode_col = np.array(W_dev_per_mode_col)
    
    return stats_df, W_dev_per_mode_col


class OneLayerNN(nn.Module):
    def __init__(self, n_dim, W_init_scale=0.1, device='cpu'):
        super(OneLayerNN, self).__init__()
        W = th.randn(n_dim, n_dim).to(device) / math.sqrt(n_dim) * W_init_scale # TODO: Check normalization
        self.W = nn.Parameter(W)
        b = th.zeros(n_dim).to(device)
        self.b = nn.Parameter(b)
    
    def forward(self, x):
        return th.mm(x, self.W) + self.b
    
    def get_W_eff(self):
        return self.W
    
    
class TwoLayerSymNN(nn.Module):
    def __init__(self, n_dim, n_hidden, W_init_scale=0.1):
        super(TwoLayerSymNN, self).__init__()
        W = th.randn(n_dim, n_hidden) / math.sqrt(n_hidden) * W_init_scale # TODO: Check normalization?? 
        self.W = nn.Parameter(W)
        b = th.zeros(n_dim)
        self.b = nn.Parameter(b)
        #move all parameters to device
    
    def forward(self, x):
        return x @ self.W @ self.W.T + self.b
    
    def get_W_eff(self):
        return self.W @ self.W.T


class TwoLayerGenNN(nn.Module):
    def __init__(self, n_dim, n_hidden, W_init_scale=0.1):
        super(TwoLayerGenNN, self).__init__()
        W1 = th.randn(n_dim, n_hidden) / math.sqrt(n_dim) * W_init_scale # TODO: Check normalization??
        self.W1 = nn.Parameter(W1)
        W2 = th.randn(n_hidden, n_dim) / math.sqrt(n_hidden) * W_init_scale
        self.W2 = nn.Parameter(W2)
        b = th.zeros(n_dim)
        self.b = nn.Parameter(b)
    
    def forward(self, x):
        return x @ self.W1 @ self.W2 + self.b
    
    def get_W_eff(self):
        return self.W1 @ self.W2


# class MultiLayerLinearNN(nn.Module):
#     def __init__(self, n_dim, n_hidden_list, W_init_scale=0.1):
#         super(MultiLayerLinearNN, self).__init__()
#         self.layers = []
#         n_hidden_list = [n_dim] + n_hidden_list + [n_dim]
#         for i in range(len(n_hidden_list) - 1):
#             W = th.randn(n_hidden_list[i], n_hidden_list[i + 1]) / math.sqrt(n_hidden_list[i]) * W_init_scale # TODO: Check normalization
#             self.layers.append(W)
#         # reigster all parameters
#         for i, layer in enumerate(self.layers):
#             self.register_parameter(f'W{i}', nn.Parameter(layer))
#         b = th.zeros(n_dim)
#         self.b = nn.Parameter(b)
    
#     def forward(self, x):
#         for layer in self.layers:
#             x = x @ layer
#         return x + self.b
    
#     def get_W_eff(self):
#         W_eff = th.eye(n_dim)
#         for layer in self.layers:
#             W_eff = W_eff @ layer
#         return W_eff

class MultiLayerLinearNN(nn.Module):
    def __init__(self, n_dim, n_hidden_list, W_init_scale=0.1):
        super(MultiLayerLinearNN, self).__init__()
        self.layers = nn.ModuleList()
        n_hidden_list = [n_dim] + n_hidden_list + [n_dim]
        for i in range(len(n_hidden_list) - 1):
            W = th.randn(n_hidden_list[i], n_hidden_list[i + 1]) / math.sqrt(n_hidden_list[i]) * W_init_scale # TODO: Check normalization
            linear = nn.Linear(n_hidden_list[i], n_hidden_list[i + 1], bias=False)
            linear.weight.data = W.T.clone()
            self.layers.append(linear)
        b = th.zeros(n_dim)
        self.b = nn.Parameter(b)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x + self.b
    
    def get_W_eff(self):
        W_eff = th.eye(n_dim).to(self.b.device)
        for layer in self.layers:
            W_eff = layer(W_eff)
        return W_eff
#%%
from circuit_toolkit.plot_utils import saveallforms
figdir = r"/n/home12/binxuwang/Github/DiffusionLearningCurve/figures/empiric_learncurve"
#%%
# Main execution
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
n_dim = 784 
batch_size = 2048
sigma = 0.5
total_steps = 10000
lr = 1.0
optim_fun = optim.SGD
#%%
Cov, U, Lambda = get_Gaussian_Cov(n_dim)
mu = th.zeros(n_dim).to(device)
#%%
# One layer linear regression training case
expstr = "OneLayerLinear_W0_1"
denoiser = OneLayerNN(n_dim, W_init_scale=0.1, device=device)
stats_df, W_dev_per_mode_col = train_model(denoiser, U, Lambda, mu, sigma, batch_size, total_steps, lr, optim_fun)
fig1, fig2 = plot_results(stats_df, W_dev_per_mode_col)
saveallforms(figdir, f"{expstr}_loss_curve", fig1, )
saveallforms(figdir, f"{expstr}_weight_dev_per_mode", fig2, )
#%%
# One layer linear regression training case
expstr = "OneLayerLinear_W0_5"
denoiser = OneLayerNN(n_dim, W_init_scale=0.5, device=device)
stats_df, W_dev_per_mode_col = train_model(denoiser, U, Lambda, mu, sigma, batch_size, total_steps, lr, optim_fun)
fig1, fig2 = plot_results(stats_df, W_dev_per_mode_col)
saveallforms(figdir, f"{expstr}_loss_curve", fig1, )
saveallforms(figdir, f"{expstr}_weight_dev_per_mode", fig2, )
#%%
# Two layer symmetric linear case
expstr = "TwoLayerSym_D4096_W0_1"
n_hidden = 4096
denoiser = TwoLayerSymNN(n_dim, n_hidden=n_hidden, W_init_scale=0.1, ).to(device)
stats_df, W_dev_per_mode_col = train_model(denoiser, U, Lambda, mu, sigma, batch_size, total_steps, lr, optim_fun)
fig1, fig2 = plot_results(stats_df, W_dev_per_mode_col)
saveallforms(figdir, f"{expstr}_loss_curve", fig1, )
saveallforms(figdir, f"{expstr}_weight_dev_per_mode", fig2, )

#%%
# Two layer symmetric linear case
expstr = "TwoLayerSym_D4096_W0_5"
n_hidden = 4096
denoiser = TwoLayerSymNN(n_dim, n_hidden=n_hidden, W_init_scale=0.5, ).to(device)
stats_df, W_dev_per_mode_col = train_model(denoiser, U, Lambda, mu, sigma, batch_size, total_steps, lr, optim_fun)
fig1, fig2 = plot_results(stats_df, W_dev_per_mode_col)
saveallforms(figdir, f"{expstr}_loss_curve", fig1, )
saveallforms(figdir, f"{expstr}_weight_dev_per_mode", fig2, )

#%%
# Two layer symmetric linear case
expstr = "TwoLayerGen_D4096_W0_1"
n_hidden = 4096
denoiser = TwoLayerGenNN(n_dim, n_hidden=n_hidden, W_init_scale=0.1, ).to(device)
stats_df, W_dev_per_mode_col = train_model(denoiser, U, Lambda, mu, sigma, batch_size, total_steps, lr, optim_fun)
fig1, fig2 = plot_results(stats_df, W_dev_per_mode_col)
saveallforms(figdir, f"{expstr}_loss_curve", fig1, )
saveallforms(figdir, f"{expstr}_weight_dev_per_mode", fig2, )


#%%
# Two layer symmetric linear case
expstr = "TwoLayerGen_D4096_W0_5"
n_hidden = 4096
denoiser = TwoLayerGenNN(n_dim, n_hidden=n_hidden, W_init_scale=0.5, ).to(device)
stats_df, W_dev_per_mode_col = train_model(denoiser, U, Lambda, mu, sigma, batch_size, total_steps, lr, optim_fun)
fig1, fig2 = plot_results(stats_df, W_dev_per_mode_col)
saveallforms(figdir, f"{expstr}_loss_curve", fig1, )
saveallforms(figdir, f"{expstr}_weight_dev_per_mode", fig2, )


#%%
# Two layer symmetric linear case
expstr = "ThreeLayerGen_D2048_W0_1"
n_hidden = 1024
denoiser = MultiLayerLinearNN(n_dim, n_hidden_list=[n_hidden, n_hidden], W_init_scale=0.1, ).to(device)
stats_df, W_dev_per_mode_col = train_model(denoiser, U, Lambda, mu, sigma, batch_size, total_steps, lr, optim_fun)
fig1, fig2 = plot_results(stats_df, W_dev_per_mode_col)
saveallforms(figdir, f"{expstr}_loss_curve", fig1, )
saveallforms(figdir, f"{expstr}_weight_dev_per_mode", fig2, )

#%%
model = TwoLayerGenNN(n_dim, n_hidden=n_hidden, W_init_scale=1, )
model(th.randn(784)).norm()

# %%
