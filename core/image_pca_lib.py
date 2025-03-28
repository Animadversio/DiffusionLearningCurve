import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_pca_stats(images, device="cuda", svd_lowrank=False, k=200):
    """Compute PCA statistics (mean, eigenvectors, eigenvalues) for a set of images
    
    Args:
        images: torch.Tensor or numpy.ndarray of shape (N, H, W) or (N, C, H, W)
        use_cuda: bool, whether to use GPU acceleration if available
    
    Returns:
        img_mean: torch.Tensor - mean image 
        eigval: torch.Tensor - eigenvalues in descending order
        eigvec: torch.Tensor - corresponding eigenvectors as columns
    """
    # Convert numpy to torch if needed
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    # Ensure float type
    images = images.float()
    # Normalize to [0,1] if needed
    if images.max() > 1.0:
        images = images / 255.0
    # Move to GPU if requested
    if device == "cuda" and torch.cuda.is_available():
        images = images.cuda()
    # Reshape to (N, -1) for PCA
    N = images.shape[0]
    img_shape = images.shape[1:]
    X = images.view(N, -1)
    # Compute mean
    img_mean = torch.mean(X, dim=0)
    # Center the data
    X_centered = X - img_mean.unsqueeze(0)
    if svd_lowrank:
        # X_centered: (N, D), want top k PCs
        U, S, Vt = torch.svd_lowrank(X_centered, q=k, niter=2)
        # Vt is (k, D): each row is a principal component
        # If you want them as columns:
        eigvec = Vt  # (D, k)
        eigval = (S**2) / (N - 1)
    else:
        # Compute covariance matrix
        cov = torch.matmul(X_centered.T, X_centered) / (N - 1)
        # Compute eigendecomposition
        eigval, eigvec = torch.linalg.eigh(cov)
    # Sort in descending order
    sorted_indices = torch.argsort(eigval, descending=True)
    eigval = eigval[sorted_indices]
    eigvec = eigvec[:, sorted_indices]
    # Print summary statistics
    print(f"Mean shape: {img_mean.shape}")
    print(f"Mean value range: [{img_mean.min().item():.2f}, {img_mean.max().item():.2f}]")
    # print(f"Covariance matrix shape: {cov.shape}")
    print(f"Covariance eigval range: [{eigval.min().item():.2f}, {eigval.max().item():.2f}]")
    return img_mean, eigval, eigvec


def plot_eigenvectors(img_mean, eigval, eigvec, eigen_ids, img_shape=(128, 128), avg_color_channel=False):
    n_eigen = len(eigen_ids)
    ncols = 5
    nrows = (n_eigen + ncols ) // ncols
    plt.figure(figsize=(15, 3 * nrows))
    plt.subplot(nrows, ncols, 1)
    if len(img_shape) == 2:
        plt.imshow(img_mean.reshape(img_shape).cpu().numpy(), cmap='gray')
    else:
        plt.imshow(img_mean.reshape(img_shape).permute(1, 2, 0).cpu().numpy())
    plt.title('Mean Image')
    plt.axis('off')

    for i, eigen_id in enumerate(eigen_ids):
        plt.subplot(nrows, ncols, i + 2)
        # Reshape eigenvector back to image dimensions
        if len(img_shape) == 2:
            eigvec_img = eigvec[:, eigen_id].reshape(img_shape).cpu().numpy()
            plt.imshow(eigvec_img, cmap='RdBu')
        else:
            eigvec_img = eigvec[:, eigen_id].reshape(img_shape).permute(1, 2, 0).cpu().numpy()
            if avg_color_channel:
                eigvec_img = np.mean(eigvec_img, axis=-1)
                plt.imshow(eigvec_img, cmap='RdBu')
            else:
                plt.imshow(eigvec_img / eigvec_img.std())
        
        # Plot with a diverging colormap centered at 0
        plt.title(f'Eig{eigen_id}={eigval[eigen_id]:.1e}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()