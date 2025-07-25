# GMM Score Functions Summary

This document summarizes all the functions available for computing scores (gradients of log probability) for Gaussian Mixture Model (GMM) point clouds in the codebase.

## 1. Class-Based Score Functions

### GaussianMixture Class (NumPy)
**File**: `core/gaussian_mixture_lib.py`

```python
class GaussianMixture:
    def score(self, x):
        """
        Compute the score ∇_x log p(x) for the given x.
        
        Args:
            x: Input points [N_batch, N_dim]
            
        Returns:
            scores: Score vectors [N_batch, N_dim]
        """
        
    def score_decompose(self, x):
        """
        Compute the gradient to each component for the score ∇_x log p(x).
        
        Args:
            x: Input points [N_batch, N_dim]
            
        Returns:
            gradvec_list: List of gradient vectors for each component
            participance: Component participation weights [N_batch, N_components]
        """
```

### GaussianMixture_torch Class (PyTorch)
**File**: `core/gaussian_mixture_lib.py`

```python
class GaussianMixture_torch:
    def score(self, x):
        """
        Compute the score ∇_x log p(x) for the given x (PyTorch version).
        
        Args:
            x: Input points [N_batch, N_dim]
            
        Returns:
            scores: Score vectors [N_batch, N_dim]
        """
        
    def score_decompose(self, x):
        """
        Compute the gradient to each component for the score ∇_x log p(x) (PyTorch version).
        
        Args:
            x: Input points [N_batch, N_dim]
            
        Returns:
            gradvec_list: List of gradient vectors for each component
            participance: Component participation weights [N_batch, N_components]
        """
```

## 2. Standalone Score Functions

### General GMM Score Functions (NumPy)
**File**: `core/gmm_general_diffusion_lib.py`

```python
def gaussian_mixture_logprob_score(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate log probability and score of a Gaussian mixture model.
    
    Args:
        x: Input points [N_batch, N_dim]
        mus: Component means [N_comp, N_dim]
        Us: Component rotation matrices [N_comp, N_dim, N_dim]
        Lambdas: Component eigenvalues [N_comp, N_dim]
        weights: Component weights [N_comp,] or None
        
    Returns:
        logprob: Log probabilities [N_batch,]
        score_vecs: Score vectors [N_batch, N_dim]
    """

def gaussian_mixture_score(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate score of a Gaussian mixture model (score only).
    
    Args:
        x: Input points [N_batch, N_dim]
        mus: Component means [N_comp, N_dim]
        Us: Component rotation matrices [N_comp, N_dim, N_dim]
        Lambdas: Component eigenvalues [N_comp, N_dim]
        weights: Component weights [N_comp,] or None
        
    Returns:
        score_vecs: Score vectors [N_batch, N_dim]
    """
```

### General GMM Score Functions (PyTorch)
**File**: `core/gmm_general_diffusion_lib.py`

```python
def gaussian_mixture_logprob_score_torch(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate log probability and score of a Gaussian mixture model in PyTorch.
    
    Args:
        x: Input points [N_batch, N_dim]
        mus: Component means [N_comp, N_dim]
        Us: Component rotation matrices [N_comp, N_dim, N_dim]
        Lambdas: Component eigenvalues [N_comp, N_dim]
        weights: Component weights [N_comp,] or None
        
    Returns:
        logprob: Log probabilities [N_batch,]
        score_vecs: Score vectors [N_batch, N_dim]
    """

def gaussian_mixture_score_torch(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate score of a Gaussian mixture model in PyTorch (score only).
    
    Args:
        x: Input points [N_batch, N_dim]
        mus: Component means [N_comp, N_dim]
        Us: Component rotation matrices [N_comp, N_dim, N_dim]
        Lambdas: Component eigenvalues [N_comp, N_dim]
        weights: Component weights [N_comp,] or None
        
    Returns:
        score_vecs: Score vectors [N_batch, N_dim]
    """
```

### Special GMM Score Functions (Isotropic Components)
**File**: `core/gmm_special_diffusion_lib.py`

```python
def GMM_scores(mus, sigma, x):
    """
    Compute scores for isotropic GMM components.
    
    Args:
        mus: Component means [N_branch, N_dim]
        sigma: Isotropic standard deviation (scalar)
        x: Input points [N_batch, N_dim]
        
    Returns:
        scores: Score vectors [N_batch, N_dim]
    """

def GMM_scores_torch(mus, sigma, x):
    """
    Compute scores for isotropic GMM components (PyTorch version).
    
    Args:
        mus: Component means [N_branch, N_dim]
        sigma: Isotropic standard deviation (scalar)
        x: Input points [N_batch, N_dim]
        
    Returns:
        scores: Score vectors [N_batch, N_dim]
    """
```

### Batch Score Functions with Sigma
**File**: `core/gaussian_mixture_lib.py`

```python
def gaussian_mixture_score_batch_sigma_torch(x, mus, Us, Lambdas, weights=None):
    """
    Evaluate score of a Gaussian mixture model with batch sigma.
    
    Args:
        x: Input points [N_batch, N_dim]
        mus: Component means [N_comp, N_dim]
        Us: Component rotation matrices [N_comp, N_dim, N_dim]
        Lambdas: Component eigenvalues [N_batch, N_comp, N_dim] or [N_comp, N_dim]
        weights: Component weights [N_comp,] or None
        
    Returns:
        score_vecs: Score vectors [N_batch, N_dim]
    """

def gaussian_mixture_lowrank_score_batch_sigma_torch(x, mus, Us, Lambdas, sigma, weights=None):
    """
    Evaluate score of a low-rank Gaussian mixture model.
    
    Args:
        x: Input points [N_batch, N_dim]
        mus: Component means [N_comp, N_dim]
        Us: Component rotation matrices [N_comp, N_dim, N_rank]
        Lambdas: Component eigenvalues [N_comp, N_rank]
        sigma: Noise level [N_batch,] or scalar
        weights: Component weights [N_comp,] or None
        
    Returns:
        score_vecs: Score vectors [N_batch, N_dim]
    """
```

## 3. Time-Dependent Score Functions

### Diffusion Score Functions
**File**: `core/gmm_general_diffusion_lib.py`

```python
def gmm_score_t(t, x, mus, Us, Lambdas, sigma=1E-6, weights=None):
    """
    Time-dependent score function for GMM diffusion.
    
    Args:
        t: Time parameter
        x: Input points [N_dim,]
        mus: Component means [N_comp, N_dim]
        Us: Component rotation matrices [N_comp, N_dim, N_dim]
        Lambdas: Component eigenvalues [N_comp, N_dim]
        sigma: Noise parameter
        weights: Component weights [N_comp,] or None
        
    Returns:
        score: Score vector [N_dim,]
    """

def gmm_score_t_vec(t, x, mus, Us, Lambdas, sigma=1E-6, weights=None):
    """
    Vectorized time-dependent score function for GMM diffusion.
    
    Args:
        t: Time parameter
        x: Input points [N_dim, N_batch]
        mus: Component means [N_comp, N_dim]
        Us: Component rotation matrices [N_comp, N_dim, N_dim]
        Lambdas: Component eigenvalues [N_comp, N_dim]
        sigma: Noise parameter
        weights: Component weights [N_comp,] or None
        
    Returns:
        score: Score vectors [N_dim, N_batch]
    """
```

### Special Diffusion Score Functions
**File**: `core/gmm_special_diffusion_lib.py`

```python
def score_t(t, x, mus, sigma=1E-6, alpha_fun=alpha):
    """
    Score function of p(x,t) according to VP SDE probability flow.
    
    Args:
        t: Time parameter
        x: Input points [N_dim,]
        mus: Component means [N_branch, N_dim]
        sigma: Noise parameter
        alpha_fun: Alpha function for time scaling
        
    Returns:
        score: Score vector [N_dim,]
    """

def score_t_vec(t, x, mus, sigma=1E-6, alpha_fun=alpha):
    """
    Vectorized version of score_t.
    
    Args:
        t: Time parameter
        x: Input points [N_dim, N_batch]
        mus: Component means [N_branch, N_dim]
        sigma: Noise parameter
        alpha_fun: Alpha function for time scaling
        
    Returns:
        score: Score vectors [N_dim, N_batch]
    """
```

## 4. Neural Network Score Approximators

### GMM Ansatz Networks
**File**: `core/gaussian_mixture_lib.py`

```python
class GMM_ansatz_net(nn.Module):
    """
    Neural network that approximates GMM score function.
    
    Methods:
        forward(x, t): Compute score for input x at time t
    """

class GMM_ansatz_net_lowrank(nn.Module):
    """
    Low-rank neural network that approximates GMM score function.
    
    Methods:
        forward(x, t): Compute score for input x at time t
    """

class Gauss_ansatz_net(nn.Module):
    """
    Neural network that approximates single Gaussian score function.
    
    Methods:
        forward(x, t): Compute score for input x at time t
    """
```

## 5. Analytical Score Functions

### Delta GMM Score Functions
**File**: `core/analytical_score_lib.py`

```python
def delta_GMM_score(Xt, Xmat, sigma, return_weights=False):
    """
    Compute delta GMM score function.
    
    Args:
        Xt: Target points [N_batch, N_dim]
        Xmat: Reference points [N_ref, N_dim]
        sigma: Kernel bandwidth
        return_weights: Whether to return weights
        
    Returns:
        score: Score vectors [N_batch, N_dim]
        weights: (optional) Kernel weights [N_batch, N_ref]
    """

def delta_GMM_crossterm_gaussequiv_score(Xt, Xmat, Xmean, Xcov, sigma, return_weights=False):
    """
    Compute delta GMM score with cross-term Gaussian equivalence.
    
    Args:
        Xt: Target points [N_batch, N_dim]
        Xmat: Reference points [N_ref, N_dim]
        Xmean: Mean of reference points [N_dim,]
        Xcov: Covariance of reference points [N_dim, N_dim]
        sigma: Kernel bandwidth
        return_weights: Whether to return weights
        
    Returns:
        score: Score vectors [N_batch, N_dim]
        weights: (optional) Cross-term weights [N_batch, N_ref]
    """
```

## Usage Examples

### Basic GMM Score Computation
```python
import numpy as np
from core.gaussian_mixture_lib import GaussianMixture

# Create GMM
mus = [np.array([0, 0]), np.array([2, 2])]
covs = [np.eye(2), np.eye(2)]
weights = [0.5, 0.5]
gmm = GaussianMixture(mus, covs, weights)

# Compute score
x = np.array([[1, 1], [3, 3]])
scores = gmm.score(x)
print(f"Score shape: {scores.shape}")  # (2, 2)
```

### PyTorch GMM Score Computation
```python
import torch
from core.gaussian_mixture_lib import GaussianMixture_torch

# Create PyTorch GMM
mus_torch = [torch.tensor([0., 0.]), torch.tensor([2., 2.])]
covs_torch = [torch.eye(2), torch.eye(2)]
weights_torch = torch.tensor([0.5, 0.5])
gmm_torch = GaussianMixture_torch(mus_torch, covs_torch, weights_torch)

# Compute score
x = torch.tensor([[1., 1.], [3., 3.]])
scores = gmm_torch.score(x)
print(f"Score shape: {scores.shape}")  # torch.Size([2, 2])
```

### Standalone Score Function
```python
import numpy as np
from core.gmm_general_diffusion_lib import gaussian_mixture_score

# Define GMM parameters
x = np.random.randn(100, 2)  # 100 points in 2D
mus = np.array([[0, 0], [2, 2]])  # 2 components
Us = np.array([np.eye(2), np.eye(2)])  # Identity rotations
Lambdas = np.array([[1, 1], [1, 1]])  # Unit eigenvalues
weights = np.array([0.5, 0.5])

# Compute score
scores = gaussian_mixture_score(x, mus, Us, Lambdas, weights)
print(f"Score shape: {scores.shape}")  # (100, 2)
```

## Key Differences Between Functions

1. **Class vs Standalone**: Class methods are convenient for repeated use with the same GMM, while standalone functions are more flexible for different parameter combinations.

2. **NumPy vs PyTorch**: NumPy versions are faster for CPU computation, while PyTorch versions support automatic differentiation and GPU acceleration.

3. **General vs Special**: General functions support arbitrary covariance matrices, while special functions are optimized for isotropic components.

4. **Full-rank vs Low-rank**: Low-rank versions are more efficient for high-dimensional data with low intrinsic dimensionality.

5. **Time-dependent**: Some functions include time parameters for diffusion processes.

## Performance Considerations

- Use **NumPy versions** for fast CPU computation without gradients
- Use **PyTorch versions** when you need gradients or GPU acceleration
- Use **low-rank versions** for high-dimensional data (dim > 10)
- Use **special isotropic versions** when all components have the same variance
- Use **batch versions** when processing multiple samples with different noise levels 