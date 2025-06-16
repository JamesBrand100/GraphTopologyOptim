import numpy as np
import networkx as nx
import pdb
# --- Original Floyd-Warshall implementation (for reference, no change needed) ---
def size_weighted_latency_matrix_original(connectivity_matrix, traffic_matrix):
    num_nodes = connectivity_matrix.shape[0]
    distance_matrix = np.copy(connectivity_matrix)

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, k] != np.inf and distance_matrix[k, j] != np.inf:
                    distance_matrix[i, j] = np.minimum(distance_matrix[i, j], distance_matrix[i, k] + distance_matrix[k, j])

    # Element-wise multiplication of latency and traffic matrices
    # np.multiply handles 0 * inf as nan by default. The original method implicitly allows this.
   
    print("FW")
    print(distance_matrix)
    size_weighted_latency = np.multiply(distance_matrix, traffic_matrix)


    return size_weighted_latency[traffic_matrix > 0]*traffic_matrix[traffic_matrix > 0]

# --- NetworkX Dijkstra-based implementation (with nan handling) ---
def size_weighted_latency_matrix_networkx(connectivity_matrix: np.ndarray, traffic_matrix: np.ndarray):
    num_nodes = connectivity_matrix.shape[0]

    distance_matrix = np.full((num_nodes, num_nodes), np.inf)

    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if connectivity_matrix[i, j] != np.inf and i != j:
                G.add_edge(i, j, weight=connectivity_matrix[i, j])

    active_sources = np.where(np.any(traffic_matrix != 0, axis=1))[0]

    for source_node in active_sources:
        shortest_paths_from_source = nx.single_source_dijkstra_path_length(G, source_node, weight='weight')
        for dest_node, dist in shortest_paths_from_source.items():
            distance_matrix[source_node, dest_node] = dist
            
    np.fill_diagonal(distance_matrix, 0) # Distance from node to itself is 0

    size_weighted_latency = np.multiply(distance_matrix, traffic_matrix)

    size_weighted_latency[np.isnan(size_weighted_latency)] = 0 # Replace NaNs (from 0*inf) with 0

    return size_weighted_latency[traffic_matrix > 0]*traffic_matrix[traffic_matrix > 0]

# --- Test Method (same as before) ---
def test_latency_matrix_implementations():
    print("Running comparison test for latency matrix implementations...")

    # --- Test Case 1: Small, dense graph ---
    num_nodes_1 = 4
    conn_matrix_1 = np.array([
        [0, 1, np.inf, 4],
        [1, 0, 2, np.inf],
        [np.inf, 2, 0, 1],
        [4, np.inf, 1, 0]
    ])
    traffic_matrix_1 = np.array([
        [0, 10, 5, 0],
        [1, 0, 0, 15],
        [0, 2, 0, 3],
        [10, 0, 5, 0]
    ])

    print("\n--- Test Case 1: Small, dense graph ---")
    output_original_1 = size_weighted_latency_matrix_original(conn_matrix_1, traffic_matrix_1)
    output_networkx_1 = size_weighted_latency_matrix_networkx(conn_matrix_1, traffic_matrix_1)

    print("\nOriginal (Floyd-Warshall) Output:")
    print(output_original_1)
    print("\nNetworkX (Dijkstra) Output:")
    print(output_networkx_1)

    try:
        np.testing.assert_allclose(output_original_1, output_networkx_1, rtol=1e-7, atol=1e-8, equal_nan=False)
        print("\nTest Case 1 PASSED: Outputs match!")
    except AssertionError as e:
        print(f"\nTest Case 1 FAILED: Outputs do NOT match!\n{e}")

    # --- Test Case 2: Larger, sparse graph with some unreachable nodes ---
    num_nodes_2 = 7
    conn_matrix_2 = np.full((num_nodes_2, num_nodes_2), np.inf)
    np.fill_diagonal(conn_matrix_2, 0)

    conn_matrix_2[0, 1] = 1
    conn_matrix_2[1, 2] = 2
    conn_matrix_2[2, 3] = 1
    conn_matrix_2[0, 4] = 10 
    conn_matrix_2[3, 5] = 5
    conn_matrix_2[5, 6] = 2
    conn_matrix_2[6, 0] = 3 

    traffic_matrix_2 = np.zeros((num_nodes_2, num_nodes_2))
    traffic_matrix_2[0, 3] = 100 
    traffic_matrix_2[1, 5] = 50  
    traffic_matrix_2[4, 6] = 20  
    traffic_matrix_2[2, 0] = 5   

    print("\n--- Test Case 2: Larger, sparse graph ---")
    output_original_2 = size_weighted_latency_matrix_original(conn_matrix_2, traffic_matrix_2)
    output_networkx_2 = size_weighted_latency_matrix_networkx(conn_matrix_2, traffic_matrix_2)

    print("\nOriginal (Floyd-Warshall) Output:")
    print(output_original_2)
    print("\nNetworkX (Dijkstra) Output:")
    print(output_networkx_2)

    try:
        np.testing.assert_allclose(output_original_2, output_networkx_2, rtol=1e-7, atol=1e-8, equal_nan=False)
        print("\nTest Case 2 PASSED: Outputs match!")
    except AssertionError as e:
        print(f"\nTest Case 2 FAILED: Outputs do NOT match!\n{e}")

# --- Run the test ---
if __name__ == "__main__":
    test_latency_matrix_implementations()