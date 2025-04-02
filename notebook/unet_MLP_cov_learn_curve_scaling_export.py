# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import sys
sys.path.append("/n/home12/binxuwang/Github/DiffusionLearningCurve")
from os.path import join
import pickle as pkl
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from core.img_patch_stats_analysis_lib import compute_crossing_points, sweep_and_create_sample_store, process_img_mean_cov_statistics,\
     process_patch_mean_cov_statistics, plot_variance_trajectories, plot_mean_deviation_trajectories, \
     harmonic_mean, smooth_and_find_threshold_crossing
from core.trajectory_convergence_lib import analyze_and_plot_variance
from circuit_toolkit.plot_utils import saveallforms

# %%
# from core.edm_dataset import load_dataset
def load_dataset(dataset_name, normalize=True):
    import sys
    import torchvision
    import torchvision.transforms as transforms
    sys.path.append("/n/home12/binxuwang/Github/edm")
    from training.dataset import TensorDataset, ImageFolderDataset
    edm_dataset_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/EDM_datasets/datasets"
    word_dataset_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset"
    if dataset_name == "FFHQ":
        edm_ffhq64_path = join(edm_dataset_root, "ffhq-64x64.zip")
        dataset = ImageFolderDataset(edm_ffhq64_path)
        imgsize = 64
        Xtsr_raw = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]) / 255.0
    elif dataset_name == "AFHQ":
        edm_afhq_path = join(edm_dataset_root, "afhqv2-64x64.zip")
        dataset = ImageFolderDataset(edm_afhq_path)
        imgsize = 64
        Xtsr_raw = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]) / 255.0
    elif dataset_name == "CIFAR":
        edm_cifar_path = join(edm_dataset_root, "cifar10-32x32.zip")
        dataset = ImageFolderDataset(edm_cifar_path)
        imgsize = 32
        Xtsr_raw = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]) / 255.0
    elif dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST(root='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Data', 
                            train=True, download=True, 
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]))
        imgsize = 32
        Xtsr_raw = torch.stack([dataset[i][0] for i in range(len(dataset))])
        # mnist_Xtsr = torch.stack([mnist_dataset[i][0] for i in range(len(mnist_dataset))])
        # print(mnist_Xtsr.shape) # 60000 x 32 x 32
        # Xtsr = (mnist_Xtsr.to(device) - 0.5) / 0.5
    elif dataset_name == "afhq-32x32":
        Xtsr_raw = torch.load(join(word_dataset_root, "afhq-32x32.pt"))
        imgsize = 32
    elif dataset_name == "ffhq-32x32":
        Xtsr_raw = torch.load(join(word_dataset_root, "ffhq-32x32.pt"))
        imgsize = 32
    elif dataset_name == "ffhq-32x32-fix_words":
        Xtsr_raw = torch.load(join(word_dataset_root, "ffhq-32x32-fixed_text.pt"))
        imgsize = 32
    elif dataset_name == "ffhq-32x32-random_word_jitter":
        Xtsr_raw = torch.load(join(word_dataset_root, "ffhq-32x32-random_word_jitter1-4.pt"))
        imgsize = 32
    print(f"{dataset_name} dataset: {Xtsr_raw.shape}")
    print(f"Raw value range" , (Xtsr_raw[0].max().item()), (Xtsr_raw[0].min().item()))
    if normalize:
        print("Normalizing dataset to [-1.0, 1.0]")
        Xtsr = (Xtsr_raw - 0.5) / 0.5
    else:
        Xtsr = Xtsr_raw
    return Xtsr, imgsize

#%%
import time
def test_dataset_loading():
    """
    Test function to verify all datasets are loadable.
    This function attempts to load each dataset and prints basic information about it.
    """
    datasets_to_test = [
        "FFHQ", 
        "AFHQ", 
        "CIFAR", 
        "MNIST", 
        "afhq-32x32", 
        "ffhq-32x32", 
        "ffhq-32x32-fix_words", 
        "ffhq-32x32-random_word_jitter"
    ]
    
    results = {}
    
    print("Testing dataset loading...")
    print("-" * 50)
    
    for dataset_name in datasets_to_test:
        print(f"Loading dataset: {dataset_name}")
        try:
            start_time = time.time()
            Xtsr, imgsize = load_dataset(dataset_name, normalize=True)
            load_time = time.time() - start_time
            
            results[dataset_name] = {
                "status": "Success",
                "shape": Xtsr.shape,
                "imgsize": imgsize,
                "min_value": Xtsr.min().item(),
                "max_value": Xtsr.max().item(),
                "load_time": f"{load_time:.2f} seconds"
            }
            
            print(f"  ✓ Successfully loaded {dataset_name}")
            print(f"    Shape: {Xtsr.shape}")
            print(f"    Image size: {imgsize}x{imgsize}")
            print(f"    Value range: [{Xtsr.min().item():.2f}, {Xtsr.max().item():.2f}]")
            print(f"    Load time: {load_time:.2f} seconds")
            
        except Exception as e:
            results[dataset_name] = {
                "status": "Failed",
                "error": str(e)
            }
            print(f"  ✗ Failed to load {dataset_name}")
            print(f"    Error: {str(e)}")
        
        print("-" * 50)
    
    # Summary
    print("\nDataset Loading Summary:")
    print("-" * 50)
    success_count = sum(1 for result in results.values() if result["status"] == "Success")
    print(f"Successfully loaded: {success_count}/{len(datasets_to_test)} datasets")
    
    for dataset_name, result in results.items():
        status_symbol = "✓" if result["status"] == "Success" else "✗"
        print(f"{status_symbol} {dataset_name}: {result['status']}")
    
    return results

# Uncomment the line below to run the test
test_results = test_dataset_loading()

#%% Load in the Sample Stores FFHQ MLP 
Xtsr, imgsize = load_dataset("ffhq-32x32", normalize=True)
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/"
expname = "FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4"
savedir = join(exproot, expname)
figdir = join(savedir, "figures")
os.makedirs(figdir, exist_ok=True)
sampledir = join(savedir, "samples")
sample_store = sweep_and_create_sample_store(sampledir)
#%%
imgshape = (3, 32, 32)
patch_size, patch_stride = 32, 1
step_slice = sorted(sample_store.keys())
img_mean, img_cov, img_eigval, img_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj = \
     process_img_mean_cov_statistics(Xtsr, sample_store, savedir, device="cuda", imgshape=imgshape, save_pkl=False)
# patch_mean, patch_cov, patch_eigval, patch_eigvec, mean_x_patch_sample_traj, cov_x_patch_sample_traj, diag_cov_x_patch_sample_true_eigenbasis_traj = \
#      process_patch_mean_cov_statistics(Xtsr, sample_store, savedir, patch_size=patch_size, patch_stride=patch_stride, device="cuda", 
#                                        imgshape=imgshape)
#%%
slice2plot = slice(0, 9, 1)
plot_variance_trajectories(step_slice, diag_cov_x_sample_true_eigenbasis_traj, img_eigval.cpu(), slice2plot,
               patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name="FFHQ32 MLP")
#%% plot the convergence of the mean 
slice2plot = slice(0, 9, 1)
plot_mean_deviation_trajectories(step_slice, mean_x_sample_traj, img_mean, img_eigvec.cpu(), img_eigval.cpu(), 
               slice2plot, patch_size, patch_stride, savedir=figdir, dataset_name="FFHQ32 MLP") 
#%%
slice2plot = slice(0, 100, 10)
plot_variance_trajectories(step_slice, diag_cov_x_sample_true_eigenbasis_traj, img_eigval.cpu(), slice2plot,
               patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name="FFHQ32 MLP")
#%%
slice2plot = slice(0, 100, 10)
plot_mean_deviation_trajectories(step_slice, mean_x_sample_traj, img_mean, img_eigvec.cpu(), img_eigval.cpu(), 
               slice2plot, patch_size, patch_stride, savedir=figdir, dataset_name="FFHQ32 MLP")

 
#%%
slice2plot = slice(2, 1000, 100)
plot_variance_trajectories(step_slice, diag_cov_x_sample_true_eigenbasis_traj, img_eigval.cpu(), slice2plot,
               patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name="FFHQ32 MLP")
#%%
slice2plot = slice(2, 1000, 100)
plot_mean_deviation_trajectories(step_slice, mean_x_sample_traj, img_mean, img_eigvec.cpu(), img_eigval.cpu(), 
               slice2plot, patch_size, patch_stride, savedir=figdir, dataset_name="FFHQ32 MLP") 


#%% plot the mean and PC0 eigenmode
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_mean.reshape(3, 32, 32).permute(1, 2, 0).cpu() * 10 + 0.5)
plt.subplot(1, 2, 2)
plt.imshow(img_eigvec[:, 0].reshape(3, 32, 32).permute(1, 2, 0).cpu() * 10 + 0.5)
plt.show()
#%%%
lr = 1e-4
lr_step_slice = np.array(step_slice) * lr
df = compute_crossing_points(img_eigval, diag_cov_x_sample_true_eigenbasis_traj, lr_step_slice, smooth_sigma=0.1, threshold_type="harmonic_mean", )
df.to_csv(f"{figdir}/synopsis_image_eigenmode_emergence_harmonic_mean_vs_variance.csv", index=False)
figh = analyze_and_plot_variance(df, x_col='emergence_step', y_col='Variance', 
                          hue_col='Direction', palette={"increase": "red", "decrease": "blue"}, 
                          log_x=True, log_y=True, figsize=(6, 6), fit_label_format='{direction} fit: $x = {a:.1e}y^{{{b:.2f}}}$', 
                          reverse_equation=True, annotate=False, annotate_offset=(0, 0), title=f'Variance vs Emergence Time | patch {patch_size}x{patch_size} stride {patch_stride}', 
                          xlabel='Mode emergence step * lr | harmonic mean', ylabel='Eigenmode variance', alpha=0.5, fit_line_kwargs=None, scatter_kwargs=None, ax=None)
saveallforms(figdir, f"synopsis_image_eigenmode_emergence_harmonic_mean_vs_variance_fitline_lr_reverse")



#%% Find the eigenvalues around the initial values 
var_init_mean = diag_cov_x_sample_true_eigenbasis_traj[0,:].mean()
eigval_idx = np.where(np.abs(img_eigval.cpu().numpy() - var_init_mean.numpy()) < 0.1)[0]
slice2plot = eigval_idx[::3]
plot_variance_trajectories(step_slice, diag_cov_x_sample_true_eigenbasis_traj, img_eigval.cpu(), slice2plot,
               patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name="FFHQ32 MLP")
# plot_mean_deviation_trajectories(step_slice, mean_x_sample_traj, img_mean, img_eigvec.cpu(), img_eigval.cpu(), 
#                slice2plot, patch_size, patch_stride, savedir=figdir, dataset_name="FFHQ32 MLP") 
#%%








#%% Load in the Sample Stores CIFAR MLP 
Xtsr, imgsize = load_dataset("CIFAR", normalize=True)
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/"
expname = "CIFAR_UNet_MLP_EDM_8L_3072D_lr1e-4"
savedir = join(exproot, expname)
figdir = join(savedir, "figures")
os.makedirs(figdir, exist_ok=True)
sampledir = join(savedir, "samples")
sample_store = sweep_and_create_sample_store(sampledir)
#%%
imgshape = (3, 32, 32)
patch_size, patch_stride = 32, 1
step_slice = sorted(sample_store.keys())
patch_mean, patch_cov, patch_eigval, patch_eigvec, mean_x_patch_sample_traj, cov_x_patch_sample_traj, diag_cov_x_patch_sample_true_eigenbasis_traj = \
     process_patch_mean_cov_statistics(Xtsr, sample_store, savedir, patch_size=patch_size, patch_stride=patch_stride, device="cuda", 
                                       imgshape=imgshape, save_pkl=False)
#%%
slice2plot = slice(0, 9, 1)
plot_variance_trajectories(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj, patch_eigval.cpu(), slice2plot,
                            patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name="CIFAR MLP")
#%%
slice2plot = slice(0, 100, 10)
plot_variance_trajectories(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj, patch_eigval.cpu(), slice2plot,
                            patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name="CIFAR MLP")

slice2plot = slice(0, 1000, 100)
plot_variance_trajectories(step_slice, diag_cov_x_patch_sample_true_eigenbasis_traj, patch_eigval.cpu(), slice2plot,
                            patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name="CIFAR MLP")
#%%
lr = 1e-4
lr_step_slice = np.array(step_slice) * lr
df = compute_crossing_points(patch_eigval, diag_cov_x_patch_sample_true_eigenbasis_traj, lr_step_slice, smooth_sigma=0.1, threshold_type="harmonic_mean", )
df.to_csv(f"{figdir}/synopsis_image_eigenmode_emergence_harmonic_mean_vs_variance.csv", index=False)
figh = analyze_and_plot_variance(df, x_col='emergence_step', y_col='Variance', 
                          hue_col='Direction', palette={"increase": "red", "decrease": "blue"}, 
                          log_x=True, log_y=True, figsize=(6, 6), fit_label_format='{direction} fit: $x = {a:.1e}y^{{{b:.2f}}}$', 
                          reverse_equation=True, annotate=False, annotate_offset=(0, 0), title=f'Variance vs Emergence Time | patch {patch_size}x{patch_size} stride {patch_stride}', 
                          xlabel='Mode emergence step * lr | harmonic mean', ylabel='Eigenmode variance', alpha=0.5, fit_line_kwargs=None, scatter_kwargs=None, ax=None)
saveallforms(figdir, f"synopsis_image_eigenmode_emergence_harmonic_mean_vs_variance_fitline_lr_reverse")

#%% Mass produce the plots for CIFAR MLP
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/"
for expname, dataset_name in [("CIFAR_UNet_MLP_EDM_8L_3072D_lr1e-4", "CIFAR"), 
                              ("AFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4", "afhq-32x32",),
                              ("FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4", "ffhq-32x32",),
                              ("FFHQ32_fix_words_UNet_MLP_EDM_8L_3072D_lr1e-4", "ffhq-32x32-fix_words",),
                              ("FFHQ32_random_words_jitter_UNet_MLP_EDM_8L_3072D_lr1e-4", "ffhq-32x32-random_word_jitter",),
                              ]:
    Xtsr, imgsize = load_dataset(dataset_name, normalize=True)
    savedir = join(exproot, expname)
    figdir = join(savedir, "figures")
    os.makedirs(figdir, exist_ok=True)
    sampledir = join(savedir, "samples")
    sample_store = sweep_and_create_sample_store(sampledir)

    imgshape = (3, 32, 32)
    patch_size, patch_stride = 32, 1
    step_slice = sorted(sample_store.keys())
    img_mean, img_cov, img_eigval, img_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj = \
        process_img_mean_cov_statistics(Xtsr, sample_store, savedir, device="cuda", imgshape=imgshape, save_pkl=False)
    pkl.dump({"step_slice": step_slice, "img_mean": img_mean, "img_cov": img_cov, "img_eigval": img_eigval, "img_eigvec": img_eigvec, 
              "mean_x_sample_traj": mean_x_sample_traj, "diag_cov_x_sample_true_eigenbasis_traj": diag_cov_x_sample_true_eigenbasis_traj}, 
             open(join(savedir, f"{dataset_name}_img_mean_cov_statistics.pkl"), "wb"))
    
    for slice2plot in [slice(0, 9, 1), slice(0, 30, 3), slice(0, 100, 10), slice(5, 100, 10), slice(0, 1000, 100), slice(2, 3000, 300)]:
        plot_variance_trajectories(step_slice, diag_cov_x_sample_true_eigenbasis_traj, img_eigval.cpu(), slice2plot,
                                  patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name=dataset_name)

    for slice2plot in [slice(0, 9, 1), slice(0, 30, 3), slice(0, 100, 10), slice(5, 100, 10), slice(0, 1000, 100), slice(2, 3000, 300)]:
        plot_mean_deviation_trajectories(step_slice, mean_x_sample_traj, img_mean, img_eigvec.cpu(), img_eigval.cpu(), 
             slice2plot, patch_size, patch_stride, savedir=figdir, dataset_name=dataset_name)
        
    lr = 1e-4
    lr_step_slice = np.array(step_slice) * lr
    for threshold_type in ["harmonic_mean", "geometric_mean", ]:
        df = compute_crossing_points(img_eigval.cpu(), diag_cov_x_sample_true_eigenbasis_traj, 
                                     lr_step_slice, smooth_sigma=1, threshold_type=threshold_type, )
        df.to_csv(f"{figdir}/synopsis_image_eigenmode_emergence_{threshold_type}_vs_variance.csv", index=False)
        figh = analyze_and_plot_variance(df, x_col='emergence_step', y_col='Variance', 
                        hue_col='Direction', palette={"increase": "red", "decrease": "blue"}, 
                        log_x=True, log_y=True, figsize=(6, 6), fit_label_format='{direction} fit: $x = {a:.1e}y^{{{b:.2f}}}$', 
                        reverse_equation=True, annotate=False, annotate_offset=(0, 0), 
                        title=f'Variance vs Emergence Time | patch {patch_size}x{patch_size} stride {patch_stride} | {dataset_name}', 
                        xlabel=f'Mode emergence step * lr | {threshold_type}', ylabel='Eigenmode variance', alpha=0.5, fit_line_kwargs=None, scatter_kwargs=None, ax=None)
        saveallforms(figdir, f"synopsis_image_eigenmode_emergence_{threshold_type}_vs_variance_fitline_lr_reverse")
    
    try:
        del Xtsr, img_mean, img_cov, img_eigval, img_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj
        del sample_store
    except:
        pass

# %%
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/"
for expname, dataset_name in [("CIFAR_UNet_MLP_EDM_8L_3072D_lr1e-4", "CIFAR"), 
                              ("AFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4", "afhq-32x32",),
                              ("FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4", "ffhq-32x32",),
                              ("FFHQ32_fix_words_UNet_MLP_EDM_8L_3072D_lr1e-4", "ffhq-32x32-fix_words",),
                              ("FFHQ32_random_words_jitter_UNet_MLP_EDM_8L_3072D_lr1e-4", "ffhq-32x32-random_word_jitter",),
                              ]:
    Xtsr, imgsize = load_dataset(dataset_name, normalize=True)
    savedir = join(exproot, expname)
    figdir = join(savedir, "figures")
    os.makedirs(figdir, exist_ok=True)
    sampledir = join(savedir, "samples")
    # sample_store = sweep_and_create_sample_store(sampledir)

    imgshape = (3, 32, 32)
    patch_size, patch_stride = 32, 1
    # step_slice = sorted(sample_store.keys())
    # img_mean, img_cov, img_eigval, img_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj = \
    #      process_img_mean_cov_statistics(Xtsr, sample_store, savedir, device="cuda", imgshape=imgshape, save_pkl=False)
    # pkl.dump({"step_slice": step_slice, "img_mean": img_mean, "img_cov": img_cov, "img_eigval": img_eigval, "img_eigvec": img_eigvec, 
    #           "mean_x_sample_traj": mean_x_sample_traj, "diag_cov_x_sample_true_eigenbasis_traj": diag_cov_x_sample_true_eigenbasis_traj}, 
    #          open(join(savedir, f"{dataset_name}_img_mean_cov_statistics.pkl"), "wb"))
    
    img_mean_cov_statistics = pkl.load(open(join(savedir, f"{dataset_name}_img_mean_cov_statistics.pkl"), "rb"))
    step_slice = img_mean_cov_statistics["step_slice"]
    img_mean = img_mean_cov_statistics["img_mean"]
    img_cov = img_mean_cov_statistics["img_cov"]
    img_eigval = img_mean_cov_statistics["img_eigval"]
    img_eigvec = img_mean_cov_statistics["img_eigvec"]
    mean_x_sample_traj = img_mean_cov_statistics["mean_x_sample_traj"]
    diag_cov_x_sample_true_eigenbasis_traj = img_mean_cov_statistics["diag_cov_x_sample_true_eigenbasis_traj"]
    
    for slice2plot in [slice(0, 9, 1), slice(0, 30, 3), slice(0, 100, 10), slice(5, 100, 10), slice(2, 500, 50), slice(0, 1000, 100), slice(2, 3000, 300)]:
        plot_variance_trajectories(step_slice, diag_cov_x_sample_true_eigenbasis_traj, img_eigval.cpu(), slice2plot,
                                  patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name=dataset_name)

    for slice2plot in [slice(0, 9, 1), slice(0, 30, 3), slice(0, 100, 10), slice(5, 100, 10), slice(2, 500, 50), slice(0, 1000, 100), slice(2, 3000, 300)]:
        plot_mean_deviation_trajectories(step_slice, mean_x_sample_traj, img_mean, img_eigvec.cpu(), img_eigval.cpu(), 
             slice2plot, patch_size, patch_stride, savedir=figdir, dataset_name=dataset_name)
        
    lr = 1e-4
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
    
    try:
        del Xtsr, img_mean, img_cov, img_eigval, img_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj
        del sample_store
    except NameError:
        pass

# %% Save the samples as images for visualization
import torchvision
from torchvision.transforms import ToPILImage
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/"
MLP_experiments = [("CIFAR_UNet_MLP_EDM_8L_3072D_lr1e-4", "CIFAR"), 
                    ("AFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4", "afhq-32x32",),
                    ("FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4", "ffhq-32x32",),
                    ("FFHQ32_fix_words_UNet_MLP_EDM_8L_3072D_lr1e-4", "ffhq-32x32-fix_words",),
                    ("FFHQ32_random_words_jitter_UNet_MLP_EDM_8L_3072D_lr1e-4", "ffhq-32x32-random_word_jitter",),
                    ]
for expname, dataset_name in MLP_experiments:
    Xtsr, imgsize = load_dataset(dataset_name, normalize=True)
    savedir = join(exproot, expname)
    figdir = join(savedir, "figures")
    os.makedirs(figdir, exist_ok=True)
    sampledir = join(savedir, "samples")
    sample_store = sweep_and_create_sample_store(sampledir) 
    steps = [1, 5, 10, 50, 100, 500, 5000, 10000, 50000, 100000]
    for step in steps:
        if step not in sample_store:
            print(f"Step {step} not in sample store")
            continue
        print(sample_store[step].shape)
        # try to cat the sample as a montage 
        # samples_temp = sample_store[step][:64].view(-1, 3, 32, 32)
        # samples_temp = ((samples_temp + 1) / 2).clamp(0, 1)
        # montage = torchvision.utils.make_grid(samples_temp, nrow=8)
        # ToPILImage()(montage).save(join(figdir, f"sample_step_{step}.png"))

        samples_temp = sample_store[step][:1].view(-1, 3, 32, 32)
        samples_temp = ((samples_temp + 1) / 2).clamp(0, 1)
        montage = torchvision.utils.make_grid(samples_temp, nrow=8)
        ToPILImage()(montage).save(join(figdir, f"sample_step_{step}_single.png"))
        # plt.figure(figsize=(10, 10))
        # plt.imshow(montage.permute(1, 2, 0).cpu())
        # plt.show()

# %% CNN validations
# Define CNN experiment configurations
cnn_experiments = [
#     ("FFHQ", "FFHQ_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
#     ("FFHQ_fix_words", "FFHQ_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
#     ("FFHQ_random_words_jitter", "FFHQ_random_words_jitter_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
    ("ffhq-32x32", "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
    ("ffhq-32x32-fix_words", "FFHQ32_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
    ("ffhq-32x32-random_word_jitter", "FFHQ32_random_words_jitter_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
    ("afhq-32x32", "AFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm"),
    # ("CIFAR", "CIFAR_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm")
]

lr = 1e-4
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/"
for dataset_name, expname in cnn_experiments:
    print(f"Processing CNN experiment: {expname}")
    savedir = join(exproot, expname)
    figdir = join(savedir, "figures")
    os.makedirs(figdir, exist_ok=True)
    sampledir = join(savedir, "samples")
    
    Xtsr, imgsize = load_dataset(dataset_name, normalize=True)
    sample_store = sweep_and_create_sample_store(sampledir)
    imgshape = (3, imgsize, imgsize)
    patch_size, patch_stride = imgsize, 1
    step_slice = sorted(sample_store.keys())
    
    img_mean, img_cov, img_eigval, img_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj = \
         process_img_mean_cov_statistics(Xtsr, sample_store, savedir, device="cuda", imgshape=imgshape, save_pkl=False)
    pkl.dump({"step_slice": step_slice, "img_mean": img_mean, "img_cov": img_cov, "img_eigval": img_eigval, "img_eigvec": img_eigvec, 
              "mean_x_sample_traj": mean_x_sample_traj, "diag_cov_x_sample_true_eigenbasis_traj": diag_cov_x_sample_true_eigenbasis_traj}, 
             open(join(savedir, f"{dataset_name}_img_mean_cov_statistics.pkl"), "wb"))

    
    for slice2plot in [slice(0, 9, 1), slice(0, 30, 3), slice(0, 100, 10), slice(5, 100, 10), slice(2, 500, 50), slice(0, 1000, 100), slice(2, 3000, 300)]:
        plot_variance_trajectories(step_slice, diag_cov_x_sample_true_eigenbasis_traj, img_eigval.cpu(), slice2plot,
                                  patch_size=patch_size, patch_stride=patch_stride, savedir=figdir, dataset_name=dataset_name)

    for slice2plot in [slice(0, 9, 1), slice(0, 30, 3), slice(0, 100, 10), slice(5, 100, 10), slice(2, 500, 50), slice(0, 1000, 100), slice(2, 3000, 300)]:
        plot_mean_deviation_trajectories(step_slice, mean_x_sample_traj, img_mean, img_eigvec.cpu(), img_eigval.cpu(), 
             slice2plot, patch_size, patch_stride, savedir=figdir, dataset_name=dataset_name)
        
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
        montage = torchvision.utils.make_grid(samples_temp, nrow=8)
        ToPILImage()(montage).save(join(figdir, f"sample_step_{step}_single.png"))
        
        samples_temp = sample_store[step][:64].view(-1, 3, imgsize, imgsize)
        samples_temp = ((samples_temp + 1) / 2).clamp(0, 1)
        montage = torchvision.utils.make_grid(samples_temp, nrow=8)
        ToPILImage()(montage).save(join(figdir, f"sample_step_{step}.png"))
    # except Exception as e:
    #     print(f"Error processing {expname}: {e}")
    
    try:
        del Xtsr, img_mean, img_cov, img_eigval, img_eigvec, mean_x_sample_traj, cov_x_sample_traj, diag_cov_x_sample_true_eigenbasis_traj
        del sample_store
    except NameError:
        pass

# %%
