import torch
import math

def differentiable_floyd_warshall(weights_matrix: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Computes an approximate shortest path distance matrix using a differentiable
    Floyd-Warshall algorithm with softmin (LogSumExp) approximation.

    Args:
        weights_matrix (torch.Tensor): A [N, N] tensor representing direct edge weights.
                                        Assumes large values (e.g., float('inf')) for no direct edge.
                                        This matrix is the initial distance matrix (D_0).
        beta (float): Temperature parameter for the softmin approximation.
                      Smaller beta -> closer to true min, but potentially less smooth gradients
                      and numerical instability. Larger beta -> smoother gradients, but less
                      accurate approximation of the true shortest path.

    Returns:
        torch.Tensor: A [N, N] tensor of approximate shortest path distances.
    """
    num_nodes = weights_matrix.shape[0]
    
    # Ensure weights_matrix is float for calculations
    if weights_matrix.dtype != torch.float32 and weights_matrix.dtype != torch.float64:
        weights_matrix = weights_matrix.float()

    # 1. Initialization of the Distance Matrix
    # We clone the input weights_matrix to serve as our initial distances (D_0).
    # At this stage, distances[i, j] is either the direct edge weight from i to j,
    # or float('inf') if there's no direct edge.
    distances = weights_matrix.clone() 
    
    # 2. Handling Self-Loops
    # The distance from any node to itself should always be zero.
    # This explicitly sets the diagonal elements to 0.0, ensuring correctness
    # even if the initial weights_matrix had non-zero values on the diagonal.
    distances.fill_diagonal_(0.0) 

    # 3. Outer Loop: Iterating Through Intermediate Nodes (k)
    # This loop is identical to the standard Floyd-Warshall algorithm.
    # In each iteration 'k', we consider paths that use 'k' as an intermediate node
    # to potentially find shorter paths between all other pairs of nodes (i, j).
    for k in range(num_nodes):
        # 4. Preparing the Terms for Softmin Approximation
        # The standard Floyd-Warshall update rule is:
        # D[i, j] = min(D[i, j], D[i, k] + D[k, j])

        # We're approximating min(a, b) using -beta * log(exp(-a/beta) + exp(-b/beta))
        # Or more stably: -beta * logsumexp([-a/beta, -b/beta])

        # term1 represents -D[i, j] (the first argument to the 'min' function)
        # We negate it because logsumexp inherently approximates a 'max' operation.
        # min(x, y) is equivalent to -max(-x, -y).
        term1 = -distances # This is an [N, N] tensor

        # term2 represents -(D[i, k] + D[k, j]) (the second argument to the 'min' function)
        # We perform this calculation using PyTorch's broadcasting:
        # distances[:, k] extracts the k-th column, shape [N]. .unsqueeze(1) makes it [N, 1]. (D[i, k] for all i)
        # distances[k, :] extracts the k-th row, shape [N]. .unsqueeze(0) makes it [1, N]. (D[k, j] for all j)
        # Adding a [N, 1] tensor and a [1, N] tensor results in an [N, N] tensor due to broadcasting.
        # The result [i, j] will be distances[i, k] + distances[k, j].
        # We then negate this sum.
        term2 = -(distances[:, k].unsqueeze(1) + distances[k, :].unsqueeze(0)) # This is an [N, N] tensor
        
        # 5. Scaling and Stacking Terms for LogSumExp
        # We divide both terms by 'beta'. This scaling is intrinsic to the softmin formula.
        # Then, we stack these two [N, N] tensors along a new dimension (dim=0)
        # to create a [2, N, N] tensor. This prepares the data for torch.logsumexp.
        combined_terms = torch.stack([term1 / beta, term2 / beta], dim=0)
        
        # 6. Applying LogSumExp (The Differentiable Approximation of Max)
        # torch.logsumexp(combined_terms, dim=0) performs the logsumexp operation
        # across the stacked dimension (dim=0). For each [i, j] position, it computes:
        # log(exp(term1[i,j]/beta) + exp(term2[i,j]/beta))
        # This function is numerically stable and differentiable.
        # Conceptually, it's approximating max(term1[i,j]/beta, term2[i,j]/beta).
        min_approximated_neg_dist_div_beta = torch.logsumexp(combined_terms, dim=0)
        
        # 7. Converting Back to Approximate Distances
        # We negate the result and multiply by 'beta' to undo the scaling and transformation
        # applied in steps 4 and 5, thereby converting the logsumexp output back into
        # the approximate shortest path distance for D[i,j].
        # Since logsumexp approximates max(-dist/beta), negating it approximates min(dist/beta).
        # Multiplying by beta gives the approximate min(dist).
        distances = -min_approximated_neg_dist_div_beta * beta
        
        # 8. Re-ensuring Zero Diagonals
        # Due to numerical approximation, it's possible that the diagonal elements (distance from node to itself)
        # might drift slightly from 0.0. This line explicitly resets them to 0.0 in each iteration,
        # which is crucial for correctness and stability.
        distances.fill_diagonal_(0.0)

    # 9. Return the Final Approximate Distance Matrix
    # After all 'k' iterations, the 'distances' tensor contains the approximate shortest path
    # distances between all pairs of nodes, computed in a differentiable manner.
    return distances

# --- Test Case Implementation ---
if __name__ == "__main__":
    print("--- Running Differentiable Floyd-Warshall Stability Test ---")

    # Define a simple graph with 4 nodes
    # Edge weights:
    # 0 --10--> 1
    # 0 --30--> 2
    # 1 -- 5--> 3
    # 2 --10--> 3
    # Other connections are considered 'infinity' (very large number)
    
    num_nodes = 4
    INF = 1e9 # Represents effectively infinite distance for unconnected nodes

    # Initial weights matrix (adjacency matrix with costs)
    # Row i, Column j = direct cost from i to j
    test_weights_matrix = torch.tensor([
        [0.0, 10.0, 30.0, INF],  # 0 to 0,1,2,3
        [INF, 0.0, INF, 5.0],   # 1 to 0,1,2,3
        [INF, INF, 0.0, 10.0],  # 2 to 0,1,2,3
        [INF, INF, INF, 0.0]    # 3 to 0,1,2,3
    ], dtype=torch.float32) # Using float32 for typical ML scenarios

    print("\nInput Weights Matrix:")
    print(test_weights_matrix)

    # Run the differentiable Floyd-Warshall
    # Use a moderate beta for a good balance of accuracy and differentiability
    beta_val = 1.0
    computed_distance_matrix = differentiable_floyd_warshall(test_weights_matrix, beta=beta_val)

    print(f"\nComputed Distance Matrix (beta={beta_val}):")
    print(computed_distance_matrix)

    # --- Assertions for Correctness and Stability ---

    # 1. Check for non-negativity
    assert torch.all(computed_distance_matrix >= -1e-6), "Distances should be non-negative (allowing for small floating point errors)."
    print("Assertion Passed: All computed distances are non-negative.")

    # 2. Check diagonals are zero
    assert torch.all(torch.diag(computed_distance_matrix) < 1e-6), "Diagonal elements (self-distances) should be close to zero."
    print("Assertion Passed: Diagonal elements are zero.")

    # 3. Check specific expected values (approximate due to softmin)
    # Expected shortest paths (true min):
    # D[0,0]=0, D[0,1]=10, D[0,2]=30, D[0,3]=15 (0->1->3)
    # D[1,0]=INF, D[1,1]=0, D[1,2]=INF, D[1,3]=5
    # D[2,0]=INF, D[2,1]=INF, D[2,2]=0, D[2,3]=10
    # D[3,0]=INF, D[3,1]=INF, D[3,2]=INF, D[3,3]=0

    # We expect the differentiable version to be close to these values.
    # The tolerance depends on beta. For beta=1, it should be reasonably close.
    tolerance = 1.0 # Allow for some deviation due to softmin approximation

    assert torch.isclose(computed_distance_matrix[0, 1], torch.tensor(10.0), atol=tolerance), "D[0,1] is incorrect."
    assert torch.isclose(computed_distance_matrix[0, 2], torch.tensor(30.0), atol=tolerance), "D[0,2] is incorrect."
    assert torch.isclose(computed_distance_matrix[0, 3], torch.tensor(15.0), atol=tolerance), "D[0,3] is incorrect (0->1->3)."
    assert torch.isclose(computed_distance_matrix[1, 3], torch.tensor(5.0), atol=tolerance), "D[1,3] is incorrect."
    assert torch.isclose(computed_distance_matrix[2, 3], torch.tensor(10.0), atol=tolerance), "D[2,3] is incorrect."
    
    # Check for large values where INF was expected (i.e., still unreachable)
    assert computed_distance_matrix[1, 0] > 100, "D[1,0] should be large (unreachable)."
    assert computed_distance_matrix[2, 0] > 100, "D[2,0] should be large (unreachable)."

    print("Assertion Passed: Specific path distances are approximately correct.")

    # 4. Check if gradients can be computed (basic differentiability test)
    test_weights_matrix.requires_grad_(True)
    computed_distance_matrix_grad_test = differentiable_floyd_warshall(test_weights_matrix, beta=beta_val)
    
    # Create a dummy loss that depends on the output distance matrix
    dummy_loss = computed_distance_matrix_grad_test.sum()
    dummy_loss.backward()

    assert test_weights_matrix.grad is not None, "Gradients should be computed for input weights."
    assert torch.all(test_weights_matrix.grad == test_weights_matrix.grad), "Gradients should not be NaN." # Checks for NaN
    print("Assertion Passed: Gradients can be computed and are not NaN.")

    print("\n--- Differentiable Floyd-Warshall Test Completed Successfully! ---")
