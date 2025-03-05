
    
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


import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

def analyze_and_plot_variance(df, 
                              x_col='emergence_step', 
                              y_col='Variance', 
                              hue_col='Direction',
                              palette={"increase": "red", "decrease": "blue"},
                              log_x=True,
                              log_y=True,
                              figsize=(8, 6),
                              fit_label_format='{direction} fit: $y = {a:.2f}x^{{{b:.2f}}}$',
                              reverse_equation=False,
                              annotate=True,
                              annotate_offset=(0, 0),
                              title='Variance vs Emergence Step with Fitted Lines',
                              xlabel='Emergence Step',
                              ylabel='Variance',
                              alpha=0.7,
                              fit_line_kwargs=None,
                              scatter_kwargs=None,
                              ax=None):
    """
    Analyzes and plots variance against emergence steps with separate linear fits for each direction.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - x_col (str): Column name for the x-axis.
    - y_col (str): Column name for the y-axis.
    - hue_col (str): Column name for hue differentiation (e.g., "Direction").
    - palette (dict): Dictionary mapping hue categories to colors.
    - log_x (bool): Whether to apply logarithmic scale to the x-axis.
    - log_y (bool): Whether to apply logarithmic scale to the y-axis.
    - figsize (tuple): Size of the matplotlib figure.
    - fit_label_format (str): Format string for the fit labels.
    - annotate (bool): Whether to annotate the plot with fit parameters.
    - annotate_offset (tuple): (x, y) offset for annotation text.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - alpha (float): Transparency level for scatter points.
    - fit_line_kwargs (dict): Additional keyword arguments for the fitted line plot.
    - scatter_kwargs (dict): Additional keyword arguments for the scatter plot.
    
    Returns:
    - matplotlib.figure.Figure: The generated matplotlib figure.
    """
    
    # Set default keyword arguments if not provided
    if fit_line_kwargs is None:
        fit_line_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    
    # Initialize the plot
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    plt.sca(ax)
    scatter = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        alpha=alpha,
        **scatter_kwargs
    )
    
    # Apply logarithmic scales if specified
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
    
    # Prepare for regression
    directions = df[hue_col].unique()
    colors = palette  # Assumes palette keys match directions
    
    for direction in directions:
        subset = df[df[hue_col] == direction].copy()
        
        # Ensure there are enough data points to perform regression
        if subset.shape[0] < 2:
            print(f"Not enough data points to fit for direction: {direction}")
            continue
        
        # Log-transform the data if log scales are used
        if log_x and log_y:
            # Handle zero or negative values by filtering them out
            subset = subset[(subset[x_col] > 0) & (subset[y_col] > 0)]
            if subset.empty:
                print(f"No positive data points for direction: {direction}")
                continue
            
            subset['log_x'] = np.log10(subset[x_col])
            subset['log_y'] = np.log10(subset[y_col])
            
            X = subset[['log_x']].values
            y = subset['log_y'].values
        else:
            X = subset[[x_col]].values
            y = subset[y_col].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Generate values for the fitted line
        if log_x and log_y:
            x_fit = np.linspace(subset[x_col].min(), subset[x_col].max(), 100)
            y_fit = 10**(intercept) * x_fit**slope
        else:
            x_fit = np.linspace(subset[x_col].min(), subset[x_col].max(), 100)
            y_fit = intercept + slope * x_fit
        
        # Plot the fitted line
        plt.plot(x_fit, y_fit, color=colors[direction], label=fit_label_format.format(
            direction=direction.capitalize(),
            a=10**intercept if log_x and log_y else intercept,
            b=slope if log_x and log_y else slope
        ), **fit_line_kwargs)
        
        # Annotate the plot with the parameters
        if annotate:
            # Choose annotation position near the end of the fitted line
            ann_x = x_fit[0] * (1 + annotate_offset[0])
            ann_y = y_fit[0] * (1 + annotate_offset[1])
            a = 10**intercept if log_x and log_y else intercept
            b = slope if log_x and log_y else slope
            if reverse_equation:
                label = fit_label_format.format(
                    direction=direction.capitalize(),
                    a=a**(1/b),
                    b=1/b
                )
            else:
                label = fit_label_format.format(
                    direction=direction.capitalize(),
                    a=a,
                    b=b
                )
            
            plt.text(
                ann_x,
                ann_y,
                label,
                color=colors[direction],
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right'
            )
    
    # Final plot adjustments
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    # Display the plot
    # plt.show()
    
    # Optionally, return the figure object
    return fig