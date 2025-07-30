import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import myUtils

import pdb
from matplotlib.colors import Normalize

# -----------------------------
# Generate Walker Delta constellation
# -----------------------------
def walker_delta(num_planes=18, sats_per_plane=20, inclination_deg=80, F=5):
    """
    Generates satellite constellation points in a Walker Delta pattern.

    Args:
        num_planes (int): The number of orbital planes (p).
        sats_per_plane (int): The number of satellites per plane (s).
        inclination_deg (float): The orbital inclination in degrees.
        F (int): The phase offset factor. A satellite in plane p is
                 ahead of a satellite in plane p-1 by a phase of F * 2π / (p*s).

    Returns:
        np.ndarray: An array of shape (num_planes * sats_per_plane, 3)
                    containing the [X, Y, Z] coordinates of each satellite.
    """
    inclination = np.deg2rad(inclination_deg)
    points = []
    
    # Total number of satellites
    num_sats = num_planes * sats_per_plane
    
    for p in range(num_planes):
        RAAN = 2 * np.pi * p / num_planes
        
        # Calculate the phase offset for the current plane
        plane_phase_offset = 2 * np.pi * F * p / num_sats
        
        for s in range(sats_per_plane):
            # The true anomaly is now a combination of the in-plane spacing
            # and the phase offset between planes
            true_anomaly = (2 * np.pi * s / sats_per_plane) + plane_phase_offset
            
            # position in orbital plane coordinates
            x_orb = np.cos(true_anomaly)
            y_orb = np.sin(true_anomaly)
            z_orb = 0.0
            
            # rotate by inclination about x-axis
            y_inc = y_orb * np.cos(inclination) - z_orb * np.sin(inclination)
            z_inc = y_orb * np.sin(inclination) + z_orb * np.cos(inclination)
            x_inc = x_orb
            
            # rotate by RAAN about z-axis
            X = x_inc * np.cos(RAAN) - y_inc * np.sin(RAAN)
            Y = x_inc * np.sin(RAAN) + y_inc * np.cos(RAAN)
            Z = z_inc
            
            points.append([X, Y, Z])
            
    return np.array(points)

# -----------------------------
# Tangent vector helper
# -----------------------------
def tangent_at(A, T):
    A = A / np.linalg.norm(A)
    T = T / np.linalg.norm(T)
    dot = np.dot(A, T)
    proj = T - dot * A
    norm = np.linalg.norm(proj)
    if norm < 1e-9:
        return np.zeros(3)
    return proj / norm

# -----------------------------
# Compute in-lineness
# -----------------------------
def in_lineness(origin, dest, others):
    t_dest = tangent_at(origin, dest)
    values = []
    for pt in others:
        t_other = tangent_at(origin, pt)
        values.append(np.dot(t_dest, t_other))  # cosine similarity
    return np.array(values)

# -----------------------------
# Main
# -----------------------------
points = walker_delta(num_planes=24, sats_per_plane=15)  # 24 x 15 = 360
origin_index = 0
dest_index = len(points) * 2 // 3  # pick a point ~2/3 across
origin = points[origin_index]
dest = points[dest_index]

#pdb.set_trace()
#take = myUtils.batch_similarity_metric(np.expand_dims(origin,0), np.expand_dims(dest,0), points)
take = myUtils.batch_similarity_metric_triangle_great_circle(np.expand_dims(origin,0), np.expand_dims(dest,0), points)


# compute in-lineness for all satellites
#vals = in_lineness(origin, dest, points)
# normalize from [-1,1] to [0,1] for color mapping
vals_norm = take #(take + 1) / 2.0

#pdb.set_trace()

# -----------------------------
# Visualization
# -----------------------------
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth wireframe
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.1)

# Plot satellites colored by in-lineness
scatter = ax.scatter(points[:,0], points[:,1], points[:,2], c=vals_norm, cmap='coolwarm', s=30)

# --- MODIFICATIONS START HERE ---

# 1. Add color bar to the legend
# Define the maximum value for the color bar
max_val = 1.001
min_val = 0.0 # Optional: You can also set a minimum value

# Create a Normalize object with the desired range
norm = Normalize(vmin=min_val, vmax=max_val)

# Create the color bar using the 'scatter' plot object but passing the 'norm'
# The color bar's range will now be determined by 'norm' instead of the data's max
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, 
             label=r'$\alpha_{g, i, d}$', 
             orientation='horizontal',
             norm=norm,
             ticks=[0.2, 0.4, 0.6, 0.8, 1.0,1.2]) # <-- Pass the Normalize object here

#fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label=r'$\alpha_{g, i, d}$', orientation='horizontal')

ax.axis('off')

# 2. No red dot at origin & destination, change to a different color (I chose black)
# 5. Make dots for origin and destination larger
ax.scatter(origin[0], origin[1], origin[2], c='green', s=200, edgecolor='green', label='g (given satellite)')
ax.scatter(dest[0], dest[1], dest[2], c='purple', s=200, edgecolor='purple', label='d (destination satellite)')

# 3. Remove axes/numbers along them
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 4. Remove title
# ax.set_title("Walker Delta Constellation (360 sats)\nColored by tangent in-lineness with dest path")
ax.legend()
plt.show()