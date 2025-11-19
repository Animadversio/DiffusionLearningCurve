# # Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# #
# # This work is licensed under a Creative Commons
# # Attribution-NonCommercial-ShareAlike 4.0 International License.
# # You should have received a copy of the license along with this
# # work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Combine edm dataset and wordnet dataset loading functions into one."""
import sys
from os.path import join
import numpy as np
import pickle as pkl
import torch
import torchvision
import torchvision.transforms as transforms
def load_dataset(dataset_name, normalize=True, return_channels=False):
    # EDM / custom dataset code root
    sys.path.append("/n/home12/binxuwang/Github/edm")
    from training.dataset import TensorDataset, ImageFolderDataset  # noqa: F401
    # Roots
    edm_dataset_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/EDM_datasets/datasets"
    word_dataset_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/wordnet_render_dataset"
    dataset_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets"
    data_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Data"

    Xtsr_raw = None
    imgsize = None
    imgchannels = None

    # ---------- ImageFolder-style datasets ----------
    if dataset_name == "FFHQ":
        edm_ffhq64_path = join(edm_dataset_root, "ffhq-64x64.zip")
        dataset = ImageFolderDataset(edm_ffhq64_path)
        imgsize = 64
        imgchannels = 3
        Xtsr_raw = torch.stack(
            [torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]
        ) / 255.0

    elif dataset_name == "AFHQ":
        edm_afhq_path = join(edm_dataset_root, "afhqv2-64x64.zip")
        dataset = ImageFolderDataset(edm_afhq_path)
        imgsize = 64
        imgchannels = 3
        Xtsr_raw = torch.stack(
            [torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]
        ) / 255.0

    elif dataset_name == "CIFAR":
        edm_cifar_path = join(edm_dataset_root, "cifar10-32x32.zip")
        dataset = ImageFolderDataset(edm_cifar_path)
        imgsize = 32
        imgchannels = 3
        Xtsr_raw = torch.stack(
            [torch.from_numpy(dataset[i][0]) for i in range(len(dataset))]
        ) / 255.0

    # ---------- Torchvision datasets ----------
    elif dataset_name == "MNIST":
        mnist_train = torchvision.datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((32, 32))]
            ),
        )
        imgsize = 32
        imgchannels = 1
        Xtsr_raw = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))])
        labels = [mnist_train[i][1] for i in range(len(mnist_train))]

    elif dataset_name == "CIFAR100":
        cifar100_train = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        imgsize = 32
        imgchannels = 3
        Xtsr_raw = torch.stack(
            [cifar100_train[k][0] for k in range(len(cifar100_train))]
        )
        labels = [cifar100_train[k][1] for k in range(len(cifar100_train))]

    # ---------- Word-image datasets ----------
    elif dataset_name == "words32x32_50k":
        image_tensor = torch.load(join(word_dataset_root, "words32x32_50k.pt"))
        text_list = pkl.load(
            open(join(word_dataset_root, "words32x32_50k_words.pkl"), "rb")
        )
        imgsize = 32
        imgchannels = 1
        Xtsr_raw = image_tensor

    elif dataset_name == "words32x32_50k_BW":
        image_tensor = torch.load(join(word_dataset_root, "words32x32_50k_BW.pt"))
        text_list = pkl.load(
            open(join(word_dataset_root, "words32x32_50k_BW_words.pkl"), "rb")
        )
        imgsize = 32
        imgchannels = 1
        Xtsr_raw = image_tensor

    # FFHQ with associated text (64x64)
    elif dataset_name == "FFHQ_fix_words":
        save_path = join(word_dataset_root, "ffhq-64x64-fixed_text.pt")
        image_tensor = torch.load(save_path)
        imgsize = 64
        imgchannels = 3
        Xtsr_raw = image_tensor

    elif dataset_name == "FFHQ_random_words_jitter":
        save_path = join(word_dataset_root, "ffhq-64x64-random_word_jitter2-8.pt")
        image_tensor = torch.load(save_path)
        imgsize = 64
        imgchannels = 3
        Xtsr_raw = image_tensor

    # FFHQ / AFHQ 32x32 variants
    elif dataset_name == "ffhq-32x32":
        Xtsr_raw = torch.load(join(word_dataset_root, "ffhq-32x32.pt"))
        imgsize = 32
        imgchannels = 3

    elif dataset_name == "ffhq-32x32-fix_words":
        Xtsr_raw = torch.load(join(word_dataset_root, "ffhq-32x32-fixed_text.pt"))
        imgsize = 32
        imgchannels = 3

    elif dataset_name == "ffhq-32x32-random_word_jitter":
        Xtsr_raw = torch.load(
            join(word_dataset_root, "ffhq-32x32-random_word_jitter1-4.pt")
        )
        imgsize = 32
        imgchannels = 3

    elif dataset_name == "afhq-32x32":
        Xtsr_raw = torch.load(join(word_dataset_root, "afhq-32x32.pt"))
        imgsize = 32
        imgchannels = 3

    # ---------- LSUN Church ----------
    elif dataset_name == "LSUN_church-64x64":
        # numpy array: (N, 64, 64, 3) uint8
        data_tsr = np.load(join(dataset_root, "LSUN_church", "church_outdoor_train_lmdb_color_64.npy"))
        data_tsr_torch = torch.from_numpy(data_tsr).permute(0, 3, 1, 2) / 255.0
        del data_tsr
        imgsize = 64
        imgchannels = 3
        Xtsr_raw = data_tsr_torch

    elif dataset_name == "LSUN_church-32x32":
        Xtsr_raw = torch.load(join(dataset_root, "LSUN_church", "church_train_32x32.pt"))
        imgsize = 32
        imgchannels = 3
    elif dataset_name == "LSUN_bedroom-64x64":
        Xtsr_raw = torch.load(join(dataset_root, "LSUN_bedroom_20pc", "bedroom_train_64x64.pt"))
        imgsize = 64
        imgchannels = 3
    elif dataset_name == "LSUN_bedroom-32x32":
        Xtsr_raw = torch.load(join(dataset_root, "LSUN_bedroom_20pc", "bedroom_train_32x32.pt"))
        imgsize = 32
        imgchannels = 3
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    # ---------- Sanity checks ----------
    assert Xtsr_raw.ndim == 4, f"Expected 4D tensor, got shape {Xtsr_raw.shape}"
    assert Xtsr_raw.shape[1] == imgchannels, (
        f"Channel mismatch: tensor has {Xtsr_raw.shape[1]}, expected {imgchannels}"
    )
    assert Xtsr_raw.shape[2] == imgsize
    assert Xtsr_raw.shape[3] == imgsize

    print(f"{dataset_name} dataset: {Xtsr_raw.shape}")
    print(f"imgchannels: {imgchannels}, imgsize: {imgsize}")
    print("Raw value range:", Xtsr_raw.max().item(), Xtsr_raw.min().item())

    # ---------- Normalization ----------
    if normalize:
        print("Normalizing dataset to [-1.0, 1.0]")
        Xtsr = (Xtsr_raw - 0.5) / 0.5
    else:
        Xtsr = Xtsr_raw
    if return_channels:
        return Xtsr, imgsize, imgchannels
    else:
        return Xtsr, imgsize


def select_dataset_subset(Xtsr_raw, start_idx=None, end_idx=None, step_idx=None):
    """
    Select a subset of the dataset based on index range.
    
    Args:
        Xtsr_raw: Full dataset tensor
        start_idx: Start index (inclusive, default: 0)
        end_idx: End index (exclusive, default: len(dataset))
        step_idx: Step size (default: 1)
    
    Returns:
        Selected subset of the dataset tensor
    """
    if start_idx is None and end_idx is None and step_idx is None:
        return Xtsr_raw
    
    start_idx = start_idx if start_idx is not None else 0
    end_idx = end_idx if end_idx is not None else len(Xtsr_raw)
    step_idx = step_idx if step_idx is not None else 1
    
    # Validate indices
    start_idx = max(0, min(start_idx, len(Xtsr_raw)))
    end_idx = max(start_idx, min(end_idx, len(Xtsr_raw)))
    
    # Create index range
    indices = list(range(start_idx, end_idx, step_idx))
    Xtsr_subset = Xtsr_raw[indices]
    print(f"Dataset subset: indices {start_idx}:{end_idx}:{step_idx}, selected {len(Xtsr_subset)} samples")
    
    return Xtsr_subset




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

if __name__ == "__main__":
    # Uncomment the line below to run the test
    test_results = test_dataset_loading()

# """Streaming images and labels from datasets created with dataset_tool.py."""

# import os
# import numpy as np
# import zipfile
# import PIL.Image
# import json
# import torch
# import dnnlib

# try:
#     import pyspng
# except ImportError:
#     pyspng = None

# #----------------------------------------------------------------------------
# # Abstract base class for datasets.

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self,
#         name,                   # Name of the dataset.
#         raw_shape,              # Shape of the raw image data (NCHW).
#         max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
#         use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
#         xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
#         random_seed = 0,        # Random seed to use when applying max_size.
#         cache       = False,    # Cache images in CPU memory?
#     ):
#         self._name = name
#         self._raw_shape = list(raw_shape)
#         self._use_labels = use_labels
#         self._cache = cache
#         self._cached_images = dict() # {raw_idx: np.ndarray, ...}
#         self._raw_labels = None
#         self._label_shape = None

#         # Apply max_size.
#         self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
#         if (max_size is not None) and (self._raw_idx.size > max_size):
#             np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
#             self._raw_idx = np.sort(self._raw_idx[:max_size])

#         # Apply xflip.
#         self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
#         if xflip:
#             self._raw_idx = np.tile(self._raw_idx, 2)
#             self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

#     def _get_raw_labels(self):
#         if self._raw_labels is None:
#             self._raw_labels = self._load_raw_labels() if self._use_labels else None
#             if self._raw_labels is None:
#                 self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
#             assert isinstance(self._raw_labels, np.ndarray)
#             assert self._raw_labels.shape[0] == self._raw_shape[0]
#             assert self._raw_labels.dtype in [np.float32, np.int64]
#             if self._raw_labels.dtype == np.int64:
#                 assert self._raw_labels.ndim == 1
#                 assert np.all(self._raw_labels >= 0)
#         return self._raw_labels

#     def close(self): # to be overridden by subclass
#         pass

#     def _load_raw_image(self, raw_idx): # to be overridden by subclass
#         raise NotImplementedError

#     def _load_raw_labels(self): # to be overridden by subclass
#         raise NotImplementedError

#     def __getstate__(self):
#         return dict(self.__dict__, _raw_labels=None)

#     def __del__(self):
#         try:
#             self.close()
#         except:
#             pass

#     def __len__(self):
#         return self._raw_idx.size

#     def __getitem__(self, idx):
#         raw_idx = self._raw_idx[idx]
#         image = self._cached_images.get(raw_idx, None)
#         if image is None:
#             image = self._load_raw_image(raw_idx)
#             if self._cache:
#                 self._cached_images[raw_idx] = image
#         assert isinstance(image, np.ndarray)
#         assert list(image.shape) == self.image_shape
#         assert image.dtype == np.uint8
#         if self._xflip[idx]:
#             assert image.ndim == 3 # CHW
#             image = image[:, :, ::-1]
#         return image.copy(), self.get_label(idx)

#     def get_label(self, idx):
#         label = self._get_raw_labels()[self._raw_idx[idx]]
#         if label.dtype == np.int64:
#             onehot = np.zeros(self.label_shape, dtype=np.float32)
#             onehot[label] = 1
#             label = onehot
#         return label.copy()

#     def get_details(self, idx):
#         d = dnnlib.EasyDict()
#         d.raw_idx = int(self._raw_idx[idx])
#         d.xflip = (int(self._xflip[idx]) != 0)
#         d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
#         return d

#     @property
#     def name(self):
#         return self._name

#     @property
#     def image_shape(self):
#         return list(self._raw_shape[1:])

#     @property
#     def num_channels(self):
#         assert len(self.image_shape) == 3 # CHW
#         return self.image_shape[0]

#     @property
#     def resolution(self):
#         assert len(self.image_shape) == 3 # CHW
#         assert self.image_shape[1] == self.image_shape[2]
#         return self.image_shape[1]

#     @property
#     def label_shape(self):
#         if self._label_shape is None:
#             raw_labels = self._get_raw_labels()
#             if raw_labels.dtype == np.int64:
#                 self._label_shape = [int(np.max(raw_labels)) + 1]
#             else:
#                 self._label_shape = raw_labels.shape[1:]
#         return list(self._label_shape)

#     @property
#     def label_dim(self):
#         assert len(self.label_shape) == 1
#         return self.label_shape[0]

#     @property
#     def has_labels(self):
#         return any(x != 0 for x in self.label_shape)

#     @property
#     def has_onehot_labels(self):
#         return self._get_raw_labels().dtype == np.int64

# #----------------------------------------------------------------------------
# # Dataset subclass that loads images recursively from the specified directory
# # or ZIP file.

# class ImageFolderDataset(Dataset):
#     def __init__(self,
#         path,                   # Path to directory or zip.
#         resolution      = None, # Ensure specific resolution, None = highest available.
#         use_pyspng      = True, # Use pyspng if available?
#         **super_kwargs,         # Additional arguments for the Dataset base class.
#     ):
#         self._path = path
#         self._use_pyspng = use_pyspng
#         self._zipfile = None

#         if os.path.isdir(self._path):
#             self._type = 'dir'
#             self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
#         elif self._file_ext(self._path) == '.zip':
#             self._type = 'zip'
#             self._all_fnames = set(self._get_zipfile().namelist())
#         else:
#             raise IOError('Path must point to a directory or zip')

#         PIL.Image.init()
#         self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
#         if len(self._image_fnames) == 0:
#             raise IOError('No image files found in the specified path')

#         name = os.path.splitext(os.path.basename(self._path))[0]
#         raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
#         if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
#             raise IOError('Image files do not match the specified resolution')
#         super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

#     @staticmethod
#     def _file_ext(fname):
#         return os.path.splitext(fname)[1].lower()

#     def _get_zipfile(self):
#         assert self._type == 'zip'
#         if self._zipfile is None:
#             self._zipfile = zipfile.ZipFile(self._path)
#         return self._zipfile

#     def _open_file(self, fname):
#         if self._type == 'dir':
#             return open(os.path.join(self._path, fname), 'rb')
#         if self._type == 'zip':
#             return self._get_zipfile().open(fname, 'r')
#         return None

#     def close(self):
#         try:
#             if self._zipfile is not None:
#                 self._zipfile.close()
#         finally:
#             self._zipfile = None

#     def __getstate__(self):
#         return dict(super().__getstate__(), _zipfile=None)

#     def _load_raw_image(self, raw_idx):
#         fname = self._image_fnames[raw_idx]
#         with self._open_file(fname) as f:
#             if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
#                 image = pyspng.load(f.read())
#             else:
#                 image = np.array(PIL.Image.open(f))
#         if image.ndim == 2:
#             image = image[:, :, np.newaxis] # HW => HWC
#         image = image.transpose(2, 0, 1) # HWC => CHW
#         return image

#     def _load_raw_labels(self):
#         fname = 'dataset.json'
#         if fname not in self._all_fnames:
#             return None
#         with self._open_file(fname) as f:
#             labels = json.load(f)['labels']
#         if labels is None:
#             return None
#         labels = dict(labels)
#         labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
#         labels = np.array(labels)
#         labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
#         return labels

# #----------------------------------------------------------------------------

# class TensorDataset(Dataset):
#     def __init__(self,
#         path,                   # pt or npy file
#         resolution      = None, # Ensure specific resolution, None = highest available.
#         multiplier      = 127.5,    # Multiply raw data by the specified value.
#         bias            = 127.5,    # Add the specified value to raw data.
#         **super_kwargs,         # Additional arguments for the Dataset base class.
#     ):
#         self._path = path

#         if self._file_ext(self._path) == '.pt':
#             self._type = 'pt'
#             self.raw_data = torch.load(self._path)
#             self.raw_data = self.raw_data.cpu().numpy()
#         elif self._file_ext(self._path) == '.npy':
#             self._type = 'npy'
#             self.raw_data = np.load(self._path)
#         else:
#             raise IOError('Path must point to a directory or zip')
#         self.multiplier = multiplier
#         self.bias = bias
#         # overall bias and multiplier for the dataset,
#         # this is used to cancel out the training_loop transform
#         # `images = images.to(device).to(torch.float32) / 127.5 - 1`
#         self.raw_data = self.raw_data * self.multiplier + self.bias

#         name = os.path.splitext(os.path.basename(self._path))[0]
#         raw_shape = list(self.raw_data.shape)

#         if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
#             raise IOError('Image files do not match the specified resolution')
#         super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

#     @staticmethod
#     def _file_ext(fname):
#         return os.path.splitext(fname)[1].lower()

#     def __getstate__(self):
#         return dict(super().__getstate__(), _zipfile=None)

#     def __getitem__(self, idx):
#         raw_idx = self._raw_idx[idx]
#         # image = self._cached_images.get(raw_idx, None)
#         # if image is None:
#         image = self._load_raw_image(raw_idx)
#         # assert isinstance(image, np.ndarray)
#         # assert list(image.shape) == self.image_shape
#         # assert image.dtype == np.uint8
#         if self._xflip[idx]:
#             assert image.ndim == 3 # CHW
#             image = image[:, :, ::-1]
#         return image, self.get_label(idx)

#     def _load_raw_image(self, raw_idx):
#         image = self.raw_data[raw_idx]
#         if image.ndim == 2:
#             image = image[:, :, np.newaxis] # HW => HWC
#         # image = image.transpose(2, 0, 1) # HWC => CHW
#         return image

#     def _load_raw_labels(self):
#         return None
#         # fname = 'dataset.json'
#         # if fname not in self._all_fnames:
#         #     return None
#         # with self._open_file(fname) as f:
#         #     labels = json.load(f)['labels']
#         # if labels is None:
#         #     return None
#         # labels = dict(labels)
#         # labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
#         # labels = np.array(labels)
#         # labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
#         # return labels

