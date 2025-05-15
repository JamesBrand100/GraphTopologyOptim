import numpy as np
import matplotlib.pyplot as plt

# Define the similarity metric: cosine of angle between great circle tangents at g toward d and i
def tangent_vector(base, target):
    proj = target - np.dot(target, base) * base  # projection to tangent plane
    return proj / np.linalg.norm(proj)

def similarity_metric(g, d, i):
    T_gd = tangent_vector(g, d)
    T_gi = tangent_vector(g, i)
    return np.dot(T_gd, T_gi)

# Arc generation helper
def great_circle_arc(start, end, num_points=100):
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    axis = np.cross(start, end)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-8:
        return np.tile(start.reshape(3,1), (1, num_points))  # Degenerate case
    axis /= axis_len
    angle = np.arccos(np.clip(np.dot(start, end), -1.0, 1.0))
    ts = np.linspace(0, angle, num_points)
    arcs = []
    for t in ts:
        rot = (start * np.cos(t) +
               np.cross(axis, start) * np.sin(t) +
               axis * np.dot(axis, start) * (1 - np.cos(t)))
        arcs.append(rot)
    return np.array(arcs).T

# Define points on the unit sphere
g = np.array([1, 0, 0])  # satellite
d = np.array([0, 1, 0])  # destination
i1 = np.array([0, 0, 1])   # neighbor 1
i2 = np.array([1, 1, 0]) / np.linalg.norm([1, 1, 0])  # neighbor 2

# Compute similarity
sim1 = similarity_metric(g, d, i1)
sim2 = similarity_metric(g, d, i2)

print("Similarity between g→d and g→i1 (orthogonal):", sim1)
print("Similarity between g→d and g→i2 (coplanar):", sim2)

# Generate sphere
phi, theta = np.mgrid[0:np.pi:100j, 0:2*np.pi:100j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Arcs
arc_gd = great_circle_arc(g, d)
arc_gi1 = great_circle_arc(g, i1)
arc_gi2 = great_circle_arc(g, i2)

# Plot
fig = plt.figure(figsize=(12, 6))

for i, (ei, arc_gi, sim, title) in enumerate(zip(
    [i1, i2],
    [arc_gi1, arc_gi2],
    [sim1, sim2],
    ["i1 (orthogonal)", "i2 (coplanar)"])):

    ax = fig.add_subplot(1, 2, i + 1, projection='3d')
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='none')
    
    ax.plot(*arc_gd, color='red', label='Arc g → d')
    ax.plot(*arc_gi, color='green', label='Arc g → i')
    
    ax.quiver(0, 0, 0, *g, color='blue', label='g (satellite)')
    ax.quiver(0, 0, 0, *d, color='red', label='d (destination)')
    ax.quiver(0, 0, 0, *ei, color='green', label='i (neighbor)')
    
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
