import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
# Import the specific flow algorithm module
import networkx.algorithms.flow as flow

def plot_edge_disjoint_paths_histogram(adj_matrix):
    """
    Calculates the number of edge-disjoint paths for all pairs in a graph
    defined by an adjacency matrix and plots a histogram of the results.

    This version uses the max-flow min-cut theorem and specifies the
    Edmonds-Karp algorithm for performance.

    Args:
        adj_matrix (list of lists or numpy array): The adjacency matrix of the graph.
    """
    # Create a graph from the adjacency matrix. For unweighted graphs,
    # networkx automatically assumes a capacity of 1 for each edge.
    G = nx.from_numpy_array(np.array(adj_matrix), edge_attr='capacity')
    
    # Get all unique pairs of nodes
    nodes = list(G.nodes())
    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]

    # Calculate the number of edge-disjoint paths for each pair using max flow
    path_counts = []
    print("Calculating edge-disjoint paths for all pairs...")
    for s, t in pairs:
        # The number of edge-disjoint paths equals the max flow in a unit-capacity network.
        # We specify Edmonds-Karp as requested.
        num_paths = nx.maximum_flow_value(G, s, t, flow_func=flow.edmonds_karp)
        path_counts.append(num_paths)
    print("Calculation complete.")

    # Count the frequency of each number of paths
    histogram_data = Counter(path_counts)

    # Prepare data for plotting
    paths = list(histogram_data.keys())
    counts = list(histogram_data.values())
    
    if not paths:
        print("No node pairs found or graph is empty.")
        return

    # --- Plotting the Histogram ---
    plt.figure(figsize=(10, 6))
    bar_container = plt.bar(paths, counts, color='teal', width=0.8)

    plt.xlabel("Number of Edge-Disjoint Paths", fontsize=12)
    plt.ylabel("Number of Satellite Pairs", fontsize=12)
    plt.title("Histogram of Edge-Disjoint Paths per Pair (using Edmonds-Karp)", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.bar_label(bar_container, fmt='{:,.0f}') # Add labels on top of bars
    
    # Set x-axis ticks to be integers for clarity
    max_paths = max(paths)
    plt.xticks(range(max_paths + 2))
    plt.xlim(-0.5, max_paths + 1.5)

    print("\n--- Histogram Data ---")
    print("(Number of Paths: Number of Pairs)")
    for path, count in sorted(histogram_data.items()):
        print(f"({path}, {count})")

    plt.show()


# --- Example Usage ---
if __name__ == "__main__":

    #Create parameters
    
    #training params
    gamma       = 3.0      # sharpness for alignment

    #simulation params 
    trafficScaling = 100000
    max_hops    = 20       # how many hops to unroll
    maxDemand   = 1.0
    #numFlows = 30
    beam_budget = 4      # sum of beam allocations per node

    #constellation params 
    orbitRadius = 6.946e6   
    #numSatellites = 100
    #orbitalPlanes = 10
    inclination = 80 
    phasingParameter = 5

    EARTH_MEAN_RADIUS = 6371.0e3

    #create positions
    positions, vecs = myUtils.generateWalkerStarConstellationPoints(numSatellites,
                                            inclination,
                                            orbitalPlanes,
                                            phasingParameter,
                                            orbitRadius)

    # Define a sample adjacency matrix for a satellite network
    # This represents a hypothetical 8-satellite constellation
    adjacency_matrix = [
        [0, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 0]
    ]

    # adjacency_matrix = [
    #     [0, 1],
    #     [1, 0]
    # ]

    # Run the analysis and plot the histogram
    plot_edge_disjoint_paths_histogram(adjacency_matrix)