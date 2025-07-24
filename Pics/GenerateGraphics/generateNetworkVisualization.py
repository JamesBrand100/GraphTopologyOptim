import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np

# --- 1. Create the Network ---
G = nx.Graph()
# Add nodes
nodes = [i for i in range(10)]
G.add_nodes_from(nodes)

# Add edges (example: a simple path and some connections)
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    (5, 6), (6, 7), (7, 8), (8, 9),
    (0, 6), (1, 7), (2, 8), (3, 9)
]
G.add_edges_from(edges)

# Define node positions (fixed for the animation)
pos = nx.spring_layout(G, seed=42) # For consistent layout

# --- 2. Simulate Propagation Data ---
num_frames = 10
active_edges_per_frame = [] # This will store the active edges for each display frame

# Simple propagation logic: signal spreads outwards from node 0
active_nodes = {0}
visited_nodes = {0}
edge_activation_history = [] # This stores edges activated at each *propagation step*

propagation_steps = 7

for step in range(propagation_steps):
    current_active_edges_in_step = set()
    next_active_nodes = set()

    for node in active_nodes:
        for neighbor in G.neighbors(node):
            edge = tuple(sorted((node, neighbor)))

            if neighbor not in visited_nodes:
                current_active_edges_in_step.add(edge)
                next_active_nodes.add(neighbor)
    
    # Store the set of edges activated in this *propagation step*
    edge_activation_history.append(current_active_edges_in_step)
    
    active_nodes = next_active_nodes.copy()
    visited_nodes.update(next_active_nodes)

# --- MODIFICATION START ---

# Ensure the very first frame is empty
active_edges_per_frame.append(set())

# Distribute the edge activations across more frames to make the animation smoother
frames_per_step = num_frames // propagation_steps if propagation_steps > 0 else num_frames
if frames_per_step == 0 and propagation_steps > 0: frames_per_step = 1

# Start cumulative_active_edges from an empty set
cumulative_active_edges = set()

# Iterate for `num_frames - 1` to account for the initial empty frame
for frame_idx in range(num_frames - 1): # Adjusted loop range
    step_idx = frame_idx // frames_per_step # This now correctly maps to edge_activation_history
    
    if step_idx < len(edge_activation_history):
        cumulative_active_edges.update(edge_activation_history[step_idx])
    
    active_edges_per_frame.append(cumulative_active_edges.copy())

# --- MODIFICATION END ---


# --- 3. Initialize the Plot ---
fig, ax = plt.subplots(figsize=(8, 6))

# Draw all nodes (static)
node_collection = nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=700)
node_collection.set_zorder(2) 

# Draw all edges initially (inactive state)
edge_collection = nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1.0)
edge_collection.set_zorder(1) 

# Create a mapping from edge tuple to its index in the LineCollection
edge_to_idx = {tuple(sorted((u, v))): i for i, (u, v) in enumerate(G.edges())}

# Add labels and set their zorder
text_labels = nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color='black')
for text in text_labels.values():
    text.set_zorder(4) 

ax.set_title("Node-Based Network Propagation")
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()

# --- 4. Animation Functions ---

def init():
    # This function is called to draw a clear frame before animation starts.
    # It sets the initial state of the animated artists.
    edge_collection.set_colors(['gray'] * len(G.edges()))
    edge_collection.set_linewidths([1.0] * len(G.edges()))
    return [edge_collection]

def update(frame):
    # The 'frame' argument directly corresponds to the index in active_edges_per_frame
    current_active_edges = active_edges_per_frame[frame]

    edge_colors = ['gray'] * len(G.edges())
    edge_widths = [1.0] * len(G.edges())

    for active_edge_tuple in current_active_edges:
        if active_edge_tuple in edge_to_idx:
            idx = edge_to_idx[active_edge_tuple]
            edge_colors[idx] = 'red'
            edge_widths[idx] = 3.0

    edge_collection.set_colors(edge_colors)
    edge_collection.set_linewidths(edge_widths)

    return [edge_collection]

# --- 5. Create FuncAnimation Object ---
ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames, # Total number of frames, including the initial empty one
    init_func=init,
    blit=False,
    interval=800
)

# --- 6. Show/Save the Animation ---
plt.show()

ani.save('network_propagation_initial.gif', writer='pillow', fps=1)
# ani.save('network_propagation_initial_empty.mp4', writer='ffmpeg', fps=10)