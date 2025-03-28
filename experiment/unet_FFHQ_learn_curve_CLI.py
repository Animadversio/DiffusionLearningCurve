# %%


# %%
#!kaggle datasets download -d denislukovnikov/ffhq256-images-only
#!cd $STORE_DIR/Datasets
#!unzip ffhq256-images-only.zip -d ffhq256

# %%
# extract features with vae 
# train a diffusion model on the features
# DiT? PixArt?

# %%

# %%
import matplotlib.pyplot as plt
from os.path import join
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image


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


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


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

from pprint import pprint
import argparse
from typing import List, Tuple
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

# %%

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
    parser.add_argument("--dataset_name", type=str, default="FFHQ", help="Dataset name")
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
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Evaluation batch size")
    parser.add_argument("--eval_sampling_steps", type=int, default=35, help="Evaluation sampling steps")
    parser.add_argument("--record_frequency", type=int, default=0, help="Evaluation sample frequency")
    parser.add_argument(
        '-r', '--record_step_range',
        metavar=('START', 'END', 'STEP'),
        type=int,
        nargs=3,
        action='append',
        # default=[(0, 10, 2), (10, 50, 4), (50, 100, 8), (100, 500, 16), (500, 2500, 32), (2500, 5000, 64), (5000, 10000, 128), (10000, 50000, 256)],#
        default=[(0, 10, 1), (10, 50, 2), (50, 100, 4), (100, 500, 8), (500, 2500, 16), (2500, 5000, 32), (5000, 10000, 128), (10000, 50000, 256)],#
        help="Define a range with start, end, and step. Can be used multiple times. Evaluation sample frequency"
    )
    return parser.parse_args()

args = parse_args()
dataset_name = args.dataset_name
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
eval_batch_size = args.eval_batch_size
eval_sampling_steps = args.eval_sampling_steps
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
sample_dir = f"{savedir}/samples"
os.makedirs(savedir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
device = get_device()

import sys
sys.path.append("/n/home12/binxuwang/Github/edm")
from training.dataset import TensorDataset, ImageFolderDataset

ffhq256dir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/ffhq256"
edm_dataset_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/EDM_datasets/datasets"
if dataset_name == "FFHQ":
    edm_ffhq64_path = join(edm_dataset_root, "ffhq-64x64.zip")
    dataset = ImageFolderDataset(edm_ffhq64_path)
    imgsize = 64
    imgchannels = 3
    Xtsr_raw = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]) / 255.0
elif dataset_name == "AFHQ":
    edm_afhq_path = join(edm_dataset_root, "afhqv2-64x64.zip")
    dataset = ImageFolderDataset(edm_afhq_path)
    imgsize = 64
    imgchannels = 3
    Xtsr_raw = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]) / 255.0
elif dataset_name == "CIFAR":
    edm_cifar_path = join(edm_dataset_root, "cifar10-32x32.zip")
    dataset = ImageFolderDataset(edm_cifar_path)
    imgsize = 32
    imgchannels = 3
    Xtsr_raw = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]) / 255.0
elif dataset_name == "words32x32_50k":
    wordimg_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset"
    image_tensor = torch.load(join(wordimg_root, "words32x32_50k.pt"))
    text_list = pkl.load(open(join(wordimg_root, "words32x32_50k_words.pkl"), "rb"))
    imgsize = 32
    imgchannels = 1
    Xtsr_raw = image_tensor
elif dataset_name == "FFHQ_fix_words":
    wordimg_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset"
    save_path = join(wordimg_root, "ffhq-64x64-fixed_text.pt")
    image_tensor = torch.load(save_path)
    imgsize = 64
    imgchannels = 3
    Xtsr_raw = image_tensor
elif dataset_name == "FFHQ_random_words_jitter":
    wordimg_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset"
    save_path = join(wordimg_root, "ffhq-64x64-random_word_jitter2-8.pt")
    image_tensor = torch.load(save_path)
    imgsize = 64
    imgchannels = 3
    Xtsr_raw = image_tensor

assert Xtsr_raw.shape[1] == imgchannels
assert Xtsr_raw.shape[2] == imgsize
assert Xtsr_raw.shape[3] == imgsize
print(f"{dataset_name} dataset: {len(Xtsr_raw)}")
print(f"value range" , (Xtsr_raw.max()), (Xtsr_raw.min()))
# %%
# layers_per_block = 1
# decoder_init_attn = True
# attn_resolutions = [8, 16,]
# model_channels = 64
# channel_mult = [1, 2, 4, 4]
# batch_size = 512
# nsteps = 10000
# lr = 1e-4
# record_frequency = 100
# exp_name = "ffhq64_edm_model_pilot"
# eval_sample_size = 5000
# eval_batch_size = 1024 # Process in batches of 1000

# ranges = [(0, 10, 1), (10, 50, 2), (50, 100, 4), (100, 500, 8), (500, 2500, 16), (2500, 5000, 32), (5000, 10000, 64)]
# record_times = generate_record_times(ranges)


# %%
# sample_store = {}
loss_store = {}

def sampling_callback_fn(epoch, loss, model):
    loss_store[epoch] = loss
    x_out_batches = []
    for i in range(0, eval_sample_size, eval_batch_size):
        batch_size_i = min(eval_batch_size, eval_sample_size - i)
        noise_init = torch.randn(batch_size_i, *imgshape).to(device)
        x_out_i, x_traj_i, x0hat_traj_i, t_steps_i = edm_sampler(model, noise_init,
                        num_steps=eval_sampling_steps, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True)
        x_out_batches.append(x_out_i)
    
    x_out = torch.cat(x_out_batches, dim=0)
    # sample_store[epoch] = x_out.cpu(), # x_traj.cpu(), x0hat_traj.cpu(), t_steps.cpu()
    torch.save(x_out, f"{sample_dir}/samples_epoch_{epoch:06d}.pt")
    mtg = to_imgrid(((x_out.cpu()[:64] + 1) / 2).clamp(0, 1), nrow=8, padding=1)
    mtg.save(f"{sample_dir}/samples_epoch_{epoch:06d}.png")


device = get_device()
Xtsr = (Xtsr_raw.to(device) - 0.5) / 0.5
pnts = Xtsr.view(Xtsr.shape[0], -1)
imgshape = Xtsr.shape[1:]
ndim = pnts.shape[1]
# cov_empirical = torch.cov(pnts.T, correction=1)
print(f"{args.dataset_name} dataset {pnts.shape[0]} samples, {ndim} features")
config = edict(
    channels=imgchannels,
    img_size=imgsize,
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
json.dump(args.__dict__, open(f"{savedir}/args.json", "w"))

unet = create_unet_model(config)
model_precd = EDMCNNPrecondWrapper(unet, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0)
edm_loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=0.5)
model_precd, loss_traj = train_score_model_custom_loss(Xtsr, model_precd, edm_loss_fn, 
                                    lr=lr, nepochs=nsteps, batch_size=batch_size, device=device, 
                                    callback=sampling_callback_fn, callback_freq=record_frequency, callback_step_list=record_times)


# pkl.dump(sample_store, open(f"{savedir}/sample_store.pkl", "wb"))
pkl.dump(loss_store, open(f"{savedir}/loss_store.pkl", "wb"))
torch.save(model_precd.model.state_dict(), f"{savedir}/model_final.pth")


noise_init = torch.randn(64, *imgshape).to(device)
x_out, x_traj, x0hat_traj, t_steps = edm_sampler(model_precd, noise_init, 
                num_steps=40, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True)
mtg = to_imgrid(((x_out.cpu()[:64]+1)/2).clamp(0, 1), nrow=8, padding=1)
mtg.save(f"{savedir}/learned_samples_final.png")
# %%



