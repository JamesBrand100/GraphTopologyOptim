import torch

def symmetric_sinkhorn(logits, n_iters=10, eps=1e-8):
    X = logits.clone()
    for _ in range(n_iters):
        X = 0.5 * (X + X.T)  # Enforce symmetry
        X = X / (X.sum(dim=1, keepdim=True) + eps)  # Row normalization
        X = X / (X.sum(dim=0, keepdim=True) + eps)  # Column normalization
    return X

#torch.softmax(raw_masked, dim=0)

def symmetric_softmax(logits, n_iters=10, eps=1e-8):
    X = logits.clone()
    for _ in range(n_iters):
        X = torch.softmax(X, dim=0) # Row normalization
        X = torch.softmax(X, dim=1) # Column normalization
        X = 0.5 * (X + X.T)  # Enforce symmetry
    return X

# Test case
torch.manual_seed(42)
logits = torch.randn(6, 6, requires_grad=True)

alloc = symmetric_softmax(logits, n_iters=2)

# Symmetry error
symmetry_error = (alloc - alloc.T).abs().mean().item()

# Row/col sum deviation from 1
row_sum_error = (alloc.sum(dim=1) - 1).abs().mean().item()
col_sum_error = (alloc.sum(dim=0) - 1).abs().mean().item()

print("=== Sinkhorn Symmetry + Normalization Test ===")
print(f"Symmetry error (mean |X - X^T|): {symmetry_error:.6f}")
print(f"Row sum error (mean |row_sum - 1|): {row_sum_error:.6f}")
print(f"Col sum error (mean |col_sum - 1|): {col_sum_error:.6f}")
