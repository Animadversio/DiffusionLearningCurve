
    
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def smooth_and_find_threshold_crossing(trajectory, threshold, first_crossing=False, smooth_sigma=2):
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.cpu().numpy()
    smoothed_trajectory = gaussian_filter1d(trajectory, sigma=smooth_sigma)
    # determine the direction of the crossing
    direction = 1 if smoothed_trajectory[0] > threshold else -1
    if direction == 1:
        crossing_indices = np.where(smoothed_trajectory < threshold)[0]
    else:
        crossing_indices = np.where(smoothed_trajectory > threshold)[0]
    if len(crossing_indices) > 0:
        return crossing_indices[0] if first_crossing else crossing_indices[-1], direction
    else:
        return None, direction



def smooth_and_find_range_crossing(trajectory, LB, UB, smooth_sigma=2):
    """
    Smooths the trajectory and finds the first crossing into the range [LB, UB].
    
    Parameters:
        trajectory (np.ndarray or torch.Tensor): The input trajectory data.
        LB (float or np.ndarray or torch.Tensor): Lower bound of the range.
        UB (float or np.ndarray or torch.Tensor): Upper bound of the range.
        smooth_sigma (float): Standard deviation for Gaussian kernel used in smoothing.
        
    Returns:
        crossing_index (int or None): The index where the trajectory first enters the range.
        direction (str or None): Direction of crossing ('upward' or 'downward').
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    if isinstance(LB, torch.Tensor):
        LB = LB.cpu().numpy()
    if isinstance(UB, torch.Tensor):
        UB = UB.cpu().numpy()
    
    # Smooth the trajectory
    smoothed_trajectory = gaussian_filter1d(trajectory, sigma=smooth_sigma)
    
    # Ensure LB <= UB
    if np.any(LB > UB):
        raise ValueError("Lower bound LB must be less than or equal to upper bound UB.")
    
    # Initialize direction and crossing_index
    crossing_index = None
    direction = None
    
    # Iterate through the trajectory to find the first crossing into [LB, UB]
    for i in range(1, len(smoothed_trajectory)):
        prev = smoothed_trajectory[i-1]
        current = smoothed_trajectory[i]
        
        # Check if previous point was outside the range
        was_below = prev < LB
        was_above = prev > UB
        was_inside = LB <= prev <= UB
        
        # Current point is inside the range
        is_inside = LB <= current <= UB
        
        if not is_inside and was_inside:
            # Exiting the range, not entering
            continue
        if is_inside and not was_inside:
            # Entering the range
            if was_below:
                direction = -1 # 'upward'
            elif was_above:
                direction = 1 # 'downward'
            else:
                # In case previous point was not strictly above or below
                direction = 'unknown'
            crossing_index = i
            break  # Stop after finding the first crossing
    
    return crossing_index, direction



def harmonic_mean(A, B):
    return 2 / (1 / A + 1 / B)


import pandas as pd
def compute_crossing_points(patch_eigval, diag_cov_x_patch_sample_true_eigenbasis_traj, step_slice, smooth_sigma=2, threshold_type="harmonic_mean", threshold_fraction=0.2):
    num_trajectories = diag_cov_x_patch_sample_true_eigenbasis_traj.shape[1]
    crossing_steps = []
    directions = []
    for i in range(num_trajectories):
        trajectory = diag_cov_x_patch_sample_true_eigenbasis_traj[:, i]
        if threshold_type == "range":
            threshold = np.array([patch_eigval[i] * (1 - threshold_fraction), patch_eigval[i] * (1 + threshold_fraction)])
            crossing_idx, direction = smooth_and_find_range_crossing(trajectory, threshold[0], threshold[1], smooth_sigma=smooth_sigma)
        else:
            if threshold_type == "harmonic_mean":
                threshold = harmonic_mean(patch_eigval[i], trajectory[0])
            elif threshold_type == "mean":
                threshold = (patch_eigval[i] + trajectory[0]) / 2
            elif threshold_type == "geometric_mean":
                threshold = np.sqrt(patch_eigval[i] * trajectory[0])
            crossing_idx, direction = smooth_and_find_threshold_crossing(trajectory, threshold, first_crossing=True, smooth_sigma=smooth_sigma)
        if crossing_idx is not None:
            crossing_steps.append(step_slice[crossing_idx])
            directions.append(direction)
        else:
            print(f"No crossing found for mode {i}")
            crossing_steps.append(np.nan)
            directions.append(0)
    df = pd.DataFrame({"Variance": patch_eigval.cpu().numpy(), "emergence_step": crossing_steps, "direction": directions})
    # translate direction 1 -> decrease, -1 -> increase
    df["Direction"] = df["direction"].map({1: "decrease", -1: "increase"})
    return df
