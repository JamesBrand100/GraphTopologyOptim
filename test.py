import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import myUtils

import pdb

# -----------------------------
# Generate Walker Delta constellation
# -----------------------------
def walker_delta(num_planes=24, sats_per_plane=15, inclination_deg=55):
    inclination = np.deg2rad(inclination_deg)
    points = []
    for p in range(num_planes):
        RAAN = 2*np.pi * p / num_planes
        for s in range(sats_per_plane):
            true_anomaly = 2*np.pi * s / sats_per_plane
            # position in orbital plane coordinates
            x_orb = np.cos(true_anomaly)
            y_orb = np.sin(true_anomaly)
            z_orb = 0.0
            # rotate by inclination about x-axis
            y_inc = y_orb*np.cos(inclination) - z_orb*np.sin(inclination)
            z_inc = y_orb*np.sin(inclination) + z_orb*np.cos(inclination)
            x_inc = x_orb
            # rotate by RAAN about z-axis
            X = x_inc*np.cos(RAAN) - y_inc*np.sin(RAAN)
            Y = x_inc*np.sin(RAAN) + y_inc*np.cos(RAAN)
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
vals = in_lineness(origin, dest, points)
# normalize from [-1,1] to [0,1] for color mapping
vals_norm = (take + 1) / 2.0

pdb.set_trace()

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
ax.scatter(points[:,0], points[:,1], points[:,2], c=vals_norm, cmap='coolwarm', s=30)

# Highlight origin and destination
ax.scatter(origin[0], origin[1], origin[2], c='green', s=120, edgecolor='black', label='Origin')
ax.scatter(dest[0], dest[1], dest[2], c='red', s=120, edgecolor='black', label='Destination')

ax.set_title("Walker Delta Constellation (360 sats)\nColored by tangent in-lineness with dest path")
ax.legend()
plt.show()
