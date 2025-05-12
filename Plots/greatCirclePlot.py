# Re-run the full code after kernel reset

import numpy as np
import matplotlib.pyplot as plt

# Define the similarity metric: cosine of angle between great circle tangents at A toward B and C
def tangent_vector(base, target):
    proj = target - np.dot(target, base) * base  # projection to tangent plane
    return proj / np.linalg.norm(proj)

def similarity_metric(A, B, C):
    T_AB = tangent_vector(A, B)
    T_AC = tangent_vector(A, C)
    return np.dot(T_AB, T_AC)

# Arc generation helper
def great_circle_arc(A, B, num_points=100):
    # Rotate A toward B over the shortest arc
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    axis = np.cross(A, B)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-8:
        return np.tile(A.reshape(3,1), (1, num_points))  # Degenerate case: A and B are colinear
    axis /= axis_len
    angle = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))
    ts = np.linspace(0, angle, num_points)
    arcs = []
    for t in ts:
        rot = (A * np.cos(t) +
               np.cross(axis, A) * np.sin(t) +
               axis * np.dot(axis, A) * (1 - np.cos(t)))
        arcs.append(rot)
    return np.array(arcs).T

# Define points on the unit sphere
A = np.array([1, 0, 0])
B = np.array([0, 1, 0])
C1 = np.array([0, 0, 1])   # Perpendicular to both A and B
C2 = np.array([1, 1, 0]) / np.linalg.norm([1, 1, 0])  # In plane of A and B

# Compute similarity
sim1 = similarity_metric(A, B, C1)
sim2 = similarity_metric(A, B, C2)

print("Similarity between A→B and A→C1 (orthogonal):", sim1)
print("Similarity between A→B and A→C2 (coplanar):", sim2)

# Generate sphere
phi, theta = np.mgrid[0:np.pi:100j, 0:2*np.pi:100j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Arcs
arc_AB = great_circle_arc(A, B)
arc_AC1 = great_circle_arc(A, C1)
arc_AC2 = great_circle_arc(A, C2)

# Plot
fig = plt.figure(figsize=(12, 6))

for i, (C, arc_AC, sim, title) in enumerate(zip(
    [C1, C2],
    [arc_AC1, arc_AC2],
    [sim1, sim2],
    ["C1 (orthogonal)", "C2 (coplanar)"])):
    
    ax = fig.add_subplot(1, 2, i + 1, projection='3d')
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='none')
    
    ax.plot(*arc_AB, color='red', label='Arc A → B')
    ax.plot(*arc_AC, color='green', label='Arc A → C')
    
    ax.quiver(0, 0, 0, *A, color='blue', label='A')
    ax.quiver(0, 0, 0, *B, color='red', label='B')
    ax.quiver(0, 0, 0, *C, color='green', label='C')
    
    ax.set_title(f"{title}\nSimilarity: {sim:.3f}")
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

plt.tight_layout()
plt.show()
