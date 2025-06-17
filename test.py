import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random
import pdb 

def plot_spherical_network_with_colored_lines(
    num_points: int = 30,
    radius: float = 1.0,
    connection_probability: float = 0.15,
    line_width: float = 1.0, # Increased default for better visibility of color
    point_size: int = 70,    # Increased default for better visibility
    cmap_name: str = 'coolwarm', # Good for cool-to-warm gradient
    title: str = "3D Network on a Sphere (Colored Lines)"
):
    """
    Plots a set of points around a sphere in 3D, with colors ranging from cool to warm,
    and connects them sparsely to form a graph. Both points and lines are colored
    according to the specified colormap.

    Parameters
    ----------
    num_points : int, optional
        The number of points to plot on the sphere. Defaults to 30.
    radius : float, optional
        The radius of the sphere. Defaults to 1.0.
    connection_probability : float, optional
        The probability (between 0 and 1) that any two distinct points will be connected.
        A lower value results in a sparser graph. Defaults to 0.15.
    line_width : float, optional
        The linewidth of the connection lines. Defaults to 1.0.
    point_size : int, optional
        The size of the individual points (markers). Defaults to 70.
    cmap_name : str, optional
        The name of the Matplotlib colormap to use for coloring points and lines.
        'viridis', 'plasma', 'inferno', 'magma', 'coolwarm' are good choices.
    title : str, optional
        The title of the plot.
    """
    if num_points <= 0:
        raise ValueError("num_points must be greater than 0.")
    if not (0 <= connection_probability <= 1):
        raise ValueError("connection_probability must be between 0 and 1.")

    # 1. Generate Points on a Sphere
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.arccos(2 * np.random.rand(num_points) - 1)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    positions = np.stack([x, y, z], axis=-1)

    # 2. Assign Scalar Values for Coloring Points and Lines
    # Use point index as the scalar value for a smooth cool-to-warm transition
    scalar_values_for_color = np.arange(num_points)
    
    # Normalize scalar values for colormap application
    norm = plt.Normalize(vmin=scalar_values_for_color.min(), vmax=scalar_values_for_color.max())
    
    cmap = cm.coolwarm


    # Get colors for each point
    point_colors = cmap(norm(scalar_values_for_color))

    # 3. Generate Sparse Graph Connections
    connectivity_matrix = np.zeros((num_points, num_points), dtype=bool)
    connected_pairs = []

    for i in range(num_points):
        for j in range(i + 1, num_points):
            if random.random() < connection_probability:
                connectivity_matrix[i, j] = True
                connectivity_matrix[j, i] = True
                connected_pairs.append((i, j))

    # 4. Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    scatter = ax.scatter(x, y, z, c=point_colors, s=point_size, cmap=cmap_name, label='Points', zorder=2) # zorder to keep points on top

    # Plot connections with colors derived from the colormap
    for i, j in connected_pairs:
        # Determine scalar value for the line segment
        # Using the average of the two connected node's scalar values
        line_scalar_value = (scalar_values_for_color[i] + scalar_values_for_color[j]) / 2.0
    
        line_color = cmap(norm(line_scalar_value))
        print(line_color)

        pdb.set_trace()

        xline = [positions[i, 0], positions[j, 0]]
        yline = [positions[i, 1], positions[j, 1]]
        zline = [positions[i, 2], positions[j, 2]]
        
        ax.plot(xline, yline, zline, color=line_color, linewidth=3, alpha=0.8, zorder=1) # zorder to keep lines behind points

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=16)

    # Make the plot look like a sphere (equal aspect ratio)
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Optional: Remove grid lines and axis ticks for a cleaner sphere look
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Add a colorbar to indicate the cool-to-warm mapping
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10, pad=0.05)
    cbar.set_label('Point/Line Property (Cool to Warm)', rotation=270, labelpad=15)

    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Default plot with colored lines
    # plot_spherical_network_with_colored_lines()

    # Another example with 'plasma' colormap and more points
    plot_spherical_network_with_colored_lines(
        num_points=40,
        radius=3.0,
        connection_probability=0.1,
        line_width=1.5,
        point_size=100,
        cmap_name='plasma',
        title="Sphere Network with Plasma Colored Connections"
    )

    # # Example with 'magma' colormap and denser connections
    # plot_spherical_network_with_colored_lines(
    #     num_points=35,
    #     radius=1.5,
    #     connection_probability=0.3,
    #     line_width=0.8,
    #     point_size=60,
    #     cmap_name='magma',
    #     title="Denser Network with Magma Colored Connections"
    # )