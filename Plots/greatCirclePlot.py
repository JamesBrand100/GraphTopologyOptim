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
    return np.array(arcs).T, (np.array(arcs).T)[:,0], (np.array(arcs).T)[:,5]

# Define points on the unit sphere
g = np.array([1, 0, 0])  # satellite
d = np.array([0, -1, 0]) / np.linalg.norm([0, -1, 0])  # destination
i1 = np.array([0, 0, 1])   # neighbor 1
i2 = np.array([1, -1, 0]) / np.linalg.norm([1, -1, 0])  # neighbor 2

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
arc_gd, arc_start1, arc_start2 = great_circle_arc(g, d)
arc_gi1, arc_startgi11, arc_startgi12 = great_circle_arc(g, i1)
arc_gi2, arc_startgi21, arc_startgi22 = great_circle_arc(g, i2)

# Plot
fig = plt.figure(figsize=(12, 6))

for i, (ei, arc_gi, sim, pathQuiver, title) in enumerate(zip(
    [i1, i2],
    [arc_gi1, arc_gi2],
    [sim1, sim2],
    [[arc_startgi11, arc_startgi12 ], [arc_startgi21, arc_startgi22]],
    ["i1 (orthogonal)", "i2 (coplanar)"])):

    ax = fig.add_subplot(1, 2, i + 1, projection='3d')
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='none')
    
    ax.plot(*arc_gd, color='red')
    ax.plot(*arc_gi, color='green')
    
    #point quivers
    ax.quiver(0, 0, 0, *g, color='blue', label='g (satellite)')
    ax.quiver(0, 0, 0, *d, color='red', label='d (destination)')
    ax.quiver(0, 0, 0, *ei, color='green', label='i (neighbor)')
    
    #path quivers
    #ax.quiver(*[pathQuiver[0]],  *[pathQuiver[1]])
    #ax.quiver(*arc_start1, *arc_start2)

    #ax.set_title(f"{title}\nSimilarity: {sim:.3f}")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.legend()

    # Turn off the X-axis tick labels
    ax.set_xticklabels([])

    # Turn off the Y-axis tick labels
    ax.set_yticklabels([])

    # Turn off the Z-axis tick labels
    ax.set_zticklabels([])

plt.tight_layout()
plt.show()
