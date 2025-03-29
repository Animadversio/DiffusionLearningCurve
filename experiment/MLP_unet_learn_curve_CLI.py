import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
sys.path.append("/n/home12/binxuwang/Github/DiffusionLearningCurve")
sys.path.append("/Users/binxuwang/Github/DiffusionLearningCurve/")
from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone
from core.toy_shape_dataset_lib import generate_random_star_shape_torch
from core.diffusion_basics_lib import *
from core.diffusion_edm_lib import *
import os
import pickle as pkl
from circuit_toolkit.plot_utils import saveallforms
from core.img_patch_stats_analysis_lib import *
saveroot = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve"

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


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
    parser.add_argument("--dataset_name", type=str, default="words32x32_50k", help="Dataset name")
    parser.add_argument("--exp_name", type=str, default="words32x32_50k_UNet_MLP_EDM_8L_1536D_lr1e-4", help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--nsteps", type=int, default=100000, help="Number of steps")
    parser.add_argument("--mlp_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--mlp_hidden_dim", type=int, default=1536, help="Hidden dimension")
    parser.add_argument("--mlp_time_embed_dim", type=int, default=128, help="Time embedding dimension")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--eval_sample_size", type=int, default=2048, help="Evaluation sample size")
    parser.add_argument("--eval_batch_size", type=int, default=2048, help="Evaluation batch size")
    parser.add_argument("--eval_sampling_steps", type=int, default=40, help="Evaluation sampling steps")
    parser.add_argument("--record_frequency", type=int, default=0, help="Evaluation sample frequency")
    parser.add_argument(
        '-r', '--record_step_range',
        metavar=('START', 'END', 'STEP'),
        type=int,
        nargs=3,
        action='append',
        # default=[(0, 10, 2), (10, 50, 4), (50, 100, 8), (100, 500, 16), (500, 2500, 32), (2500, 5000, 64), (5000, 10000, 128), (10000, 50000, 256)],#
        default=[(0, 10, 1), (10, 50, 2), (50, 100, 4), (100, 500, 8), (500, 2500, 16), (2500, 5000, 32), (5000, 10000, 128), (10000, 50000, 256), (50000, 100000, 512)],#
        help="Define a range with start, end, and step. Can be used multiple times. Evaluation sample frequency"
    )
    return parser.parse_args()

args = parse_args()
dataset_name = args.dataset_name
exp_name = args.exp_name
batch_size = args.batch_size
nsteps = args.nsteps
mlp_layers = args.mlp_layers
mlp_hidden_dim = args.mlp_hidden_dim
mlp_time_embed_dim = args.mlp_time_embed_dim
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
# dump the args to json
json.dump(args.__dict__, open(f"{savedir}/args.json", "w"))

if dataset_name == "words32x32_50k":
    image_tensor = torch.load("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset/words32x32_50k.pt")
    text_list = pkl.load(open("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset/words32x32_50k_words.pkl", "rb"))
    data_Xtsr = image_tensor
elif dataset_name == "MNIST":
    mnist_dataset = torchvision.datasets.MNIST(root='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Data', 
                                           train=True, download=True, transform=transforms.ToTensor())
    mnist_Xtsr = torch.stack([mnist_dataset[i][0] for i in range(len(mnist_dataset))])
    data_Xtsr = mnist_Xtsr
elif dataset_name == "ffhq-32x32":
    data_Xtsr = torch.load("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset/ffhq-32x32.pt")
    imgshape = data_Xtsr.shape[1:]
elif dataset_name == "ffhq-32x32-fix_words":
    data_Xtsr = torch.load("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset/ffhq-32x32-fixed_text.pt")
    imgshape = data_Xtsr.shape[1:]
elif dataset_name == "ffhq-32x32-random_word_jitter":
    data_Xtsr = torch.load("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset/ffhq-32x32-random_word_jitter1-4.pt")
    imgshape = data_Xtsr.shape[1:]
elif dataset_name == "afhq-32x32":
    data_Xtsr = torch.load("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset/afhq-32x32.pt")
    imgshape = data_Xtsr.shape[1:]
elif dataset_name == "CIFAR":
    import sys
    sys.path.append("/n/home12/binxuwang/Github/edm")
    from training.dataset import ImageFolderDataset
    edm_dataset_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/EDM_datasets/datasets"
    edm_cifar_path = join(edm_dataset_root, "cifar10-32x32.zip")
    dataset = ImageFolderDataset(edm_cifar_path)
    data_Xtsr = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]) / 255.0
    imgshape = data_Xtsr.shape[1:]
else:
    raise ValueError(f"Dataset {dataset_name} not found")

loss_store = {}
def sampling_callback_fn(epoch, loss, model):
    loss_store[epoch] = loss
    x_out_batches = []
    for i in range(0, eval_sample_size, eval_batch_size):
        batch_size_i = min(eval_batch_size, eval_sample_size - i)
        noise_init = torch.randn(batch_size_i, np.prod(imgshape)).to(device)
        x_out_i, x_traj_i, x0hat_traj_i, t_steps_i = edm_sampler(model, noise_init,
                        num_steps=eval_sampling_steps, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True)
        x_out_batches.append(x_out_i)
    
    x_out = torch.cat(x_out_batches, dim=0)
    # sample_store[epoch] = x_out.cpu(), # x_traj.cpu(), x0hat_traj.cpu(), t_steps.cpu()
    torch.save(x_out, f"{sample_dir}/samples_epoch_{epoch:06d}.pt")
    # make the shape correct
    x_out_reshaped = x_out.view(x_out.shape[0], *imgshape)
    mtg = to_imgrid(((x_out_reshaped.cpu()[:64] + 1) / 2).clamp(0, 1), nrow=8, padding=1)
    mtg.save(f"{sample_dir}/samples_epoch_{epoch:06d}.png")

device = get_device()
pnts = data_Xtsr.view(data_Xtsr.shape[0], -1).to(device)
pnts = (pnts - 0.5) / 0.5
ndim = pnts.shape[1]
cov_empirical = torch.cov(pnts.T, correction=1)
print(f"Dataset {pnts.shape[0]} samples, {ndim} features")
model = UNetBlockStyleMLP_backbone(ndim=ndim, nlayers=mlp_layers, nhidden=mlp_hidden_dim, time_embed_dim=mlp_time_embed_dim,)
model_precd = EDMPrecondWrapper(model, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0)
edm_loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=0.5)
model_precd, loss_traj = train_score_model_custom_loss(pnts, model_precd, edm_loss_fn, 
                                    lr=lr, nepochs=nsteps, batch_size=batch_size, device=device, 
                                    callback=sampling_callback_fn, callback_freq=record_frequency, 
                                    callback_step_list=record_times)

pkl.dump(loss_store, open(f"{savedir}/loss_store.pkl", "wb"))
torch.save(model_precd.model.state_dict(), f"{savedir}/model_final.pth")
# pkl.dump(sample_store, open(f"{savedir}/sample_store.pkl", "wb"))
# train_X_mean, train_X_cov, train_X_eigval, train_X_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj = \
#     process_pnts_mean_cov_statistics(pnts, sample_store, savedir, device="cuda")

noise_init = torch.randn(100, ndim).to(device)
x_out, x_traj, x0hat_traj, t_steps = edm_sampler(model_precd, noise_init, 
                num_steps=40, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True)
# make the shape correct
x_out_reshaped = x_out.view(x_out.shape[0], *imgshape)
mtg = to_imgrid(((x_out_reshaped.cpu()[:100]+1)/2).clamp(0, 1), nrow=8, padding=1)
mtg.save(f"{savedir}/learned_samples_final.png")
