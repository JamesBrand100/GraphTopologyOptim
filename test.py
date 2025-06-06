import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For colormaps

# Define the sharpen_top_n function (as provided)
def sharpen_top_n(X, normLim=3, temperature=5.0):
    sorted_X, _ = torch.sort(X, dim=1)
    # Ensure sorted_X has at least normLim + 1 columns for valid indexing
    if sorted_X.shape[1] < normLim + 1:
        raise ValueError(f"Input tensor X must have at least {normLim + 1} elements per row for normLim={normLim}")

    ref_val = (sorted_X[:, -normLim] + sorted_X[:, -normLim - 1]) / 2  # avg of n and n-1
    ref_val = ref_val[:, np.newaxis]

    # Avoid division by zero
    eps = 1e-8
    x_pow = (X / (ref_val + eps)) ** temperature
    complement_pow = ((1 - X) / (1 - ref_val + eps)) ** temperature

    sharpened = x_pow / (x_pow + complement_pow + eps)
    return torch.clip(sharpened, 0, 1)

def plot_sharpened_output(input_tensor, normLim=3, min_temp=0.5, max_temp=10.0, num_lines=10):
    """
    Displays the output of sharpen_top_n for different temperatures.

    X-axis: Index in the array (original position of values).
    Y-axis: Sharpened values of the array.
    Line color: Proportional to the temperature used for that line.

    Args:
        input_tensor (torch.Tensor): The 1D or 2D input tensor to sharpen.
                                     If 2D, only the first row will be plotted.
        normLim (int): The 'n' in top-n for sharpen_top_n.
        min_temp (float): Minimum temperature value for the plot range.
        max_temp (float): Maximum temperature value for the plot range.
        num_lines (int): Number of lines (different temperatures) to plot.
    """
    if input_tensor.dim() == 2:
        # Plot only the first row if a 2D tensor is provided
        X_to_plot = input_tensor[0:1, :]
        print(f"Plotting only the first row of the input tensor with shape {input_tensor.shape}.")
    elif input_tensor.dim() == 1:
        X_to_plot = input_tensor.unsqueeze(0) # Make it 2D (1, N) for sharpen_top_n
    else:
        raise ValueError("input_tensor must be 1D or 2D.")

    temperatures = np.linspace(min_temp, max_temp, num_lines)
    num_elements = X_to_plot.shape[1]
    indices = np.arange(num_elements)

    # Choose a colormap. 'viridis' is a good perceptual choice.
    # You can explore others like 'plasma', 'inferno', 'magma', 'coolwarm'
    cmap = cm.viridis
    normalize = plt.Normalize(vmin=min_temp, vmax=max_temp) # Normalize temperatures to colormap range

    plt.figure(figsize=(10, 6))

    # Plot original input as a reference (e.g., dashed gray line)
    # plt.plot(indices, X_to_plot.squeeze().numpy(), linestyle='--', color='gray',
    #          alpha=0.7, label='Original Input (Reference)')

    for temp in temperatures:
        sharpened_X = sharpen_top_n(X_to_plot, normLim=normLim, temperature=temp)
        line_color = cmap(normalize(temp)) # Get color proportional to temperature

        plt.plot(indices, sharpened_X.squeeze().numpy(), color=line_color,
                 label=f'Temp={temp:.1f}')

    plt.xlabel('Element Index in Array', fontsize=12)
    plt.ylabel('Sharpened Value', fontsize=12)
    plt.title(f'Effect of Temperature on Sharpening (Number Binary Entries = {normLim})', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Create a colorbar to show temperature mapping
    sm = cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array(temperatures)
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Temperature Value', rotation=270, labelpad=15, fontsize=12)

    # Optional: If too many lines, legend can get crowded.
    # For ~10 lines, a legend might still be useful, or just rely on the colorbar.
    # plt.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()

# Example Usage:

# 1. Create a sample input tensor (e.g., random values)
# Let's use a 1D tensor for simplicity, reshape for sharpen_top_n
input_values = torch.tensor([0.1, 0.2, 0.95, 0.8, 0.3, 0.7, 0.6, 0.9, 0.4, 0.05], dtype=torch.float32)
input_values, _ = torch.sort(input_values, descending=True)

# Set normLim (ensure it's less than the number of elements in input_values)
my_norm_lim = 3

# Plot the output for various temperatures
plot_sharpened_output(input_values, normLim=my_norm_lim, min_temp=0.1, max_temp=15.0, num_lines=10)

# Example with a 2D tensor (only the first row will be plotted)
# input_matrix = torch.rand((5, 10)) # 5 rows, 10 columns
# plot_sharpened_output(input_matrix, normLim=my_norm_lim, min_temp=0.1, max_temp=20.0, num_lines=10)