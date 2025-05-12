import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random

# Parameters
GRID_SIZE = (5, 5)  # 5x5 grid
NUM_PACKETS = 100
EDGE_CAPACITY = 2
TIME_STEPS = 10

#Node init 
# Create grid graph
def create_grid_graph(grid_size, capMat = None):
    G = nx.grid_2d_graph(*grid_size)
    G = nx.DiGraph(G)  # Make it directed for routing clarity

    #init capacity accordingly 
    for u, v in G.edges:
        if(capMat == None): 
            G[u][v]['capacity'] = EDGE_CAPACITY
        else: 
            G[u][v]['capacity'] = capMat[u][v]

    return G

#Routing init 
# Manhattan distance-based greedy routing
def greedy_next_hop(src, dst, neighbors):
    min_dist = float('inf')
    candidates = []
    for nbr in neighbors:
        dist = abs(nbr[0] - dst[0]) + abs(nbr[1] - dst[1])
        if dist < min_dist:
            candidates = [nbr]
            min_dist = dist
        elif dist == min_dist:
            candidates.append(nbr)
    return random.choice(candidates) if candidates else None

#Time Scale Init 
# Simulate one time step of packet delivery
def simulate_step(G, packets, delivery_count):
    edge_usage = defaultdict(int)
    next_packets = []

    for src, dst, path in packets:
        if src == dst:
            delivery_count += 1
            continue

        neighbors = list(G.neighbors(src))
        next_hop = greedy_next_hop(src, dst, neighbors)
        if next_hop and edge_usage[(src, next_hop)] < G[src][next_hop]['capacity']:
            edge_usage[(src, next_hop)] += 1
            next_packets.append((next_hop, dst, path + [next_hop]))
        else:
            # Drop or requeue packet - for simplicity, we drop
            continue

    return next_packets, delivery_count

#Traffic Init 
# Generate random packets (src, dst)
def generate_packets(num_packets, grid_size, dist = "random"):
    packets = []

    for _ in range(num_packets):
        if(dist == "random"): 
            src = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            dst = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
        if(dist == "corners"):
            src = (0,0)
            dst = (grid_size[0],  grid_size[1] )

        while dst == src:
            dst = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
        packets.append((src, dst, [src]))
    return packets

# Run full simulation
def run_simulation():
    G = create_grid_graph(GRID_SIZE)
    nx.draw(G)
    plt.show()
    packets = generate_packets(NUM_PACKETS, GRID_SIZE)
    delivered = 0

    for _ in range(TIME_STEPS):
        packets, delivered = simulate_step(G, packets, delivered)

    total = NUM_PACKETS
    delivery_rate = delivered / total
    return delivery_rate

# Run and show result
delivery_rate = run_simulation()
delivery_rate

print(delivery_rate)