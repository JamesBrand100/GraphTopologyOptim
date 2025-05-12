import torch
import math
import matplotlib.pyplot as plt

# 1) Build the geometry: 7 points on the unit circle
n = 7
angles = torch.linspace(0, 2*math.pi, n)
pts = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

# Indices
src, dst = 0, 3
inters = [1,2,4,5,6]

# 2) Utility: tangent‐plane projection at A of B
def tangent_vec(A, B):
    proj = B - (A.dot(B))*A
    return proj / proj.norm()

# Precompute α_i = |cos angle| between (src→dst) and (src→i)
A = pts[src]
B = pts[dst]
dir_sd = tangent_vec(A, B)
alphas = torch.stack([
    dir_sd.dot(tangent_vec(A, pts[i])).abs()
    for i in inters
])

# Precompute delays d_gi and d_id (Euclidean for simplicity)
d_gi = torch.tensor([torch.dist(pts[src], pts[i]) for i in inters])
d_id = torch.tensor([torch.dist(pts[i], pts[dst]) for i in inters])
latencies = d_gi + d_id  # shape (5,)

# 3) Learnable beam‐allocation logits (softmax → sum to 1 ⇒ 1 beam budget)
logits = torch.zeros(len(inters), requires_grad=True)
opt = torch.optim.SGD([logits], lr=0.2)

loss_history = []
for epoch in range(200):
    opt.zero_grad()
    c = torch.softmax(logits, dim=0)            # allocations ≥0, sum=1
    weights = c * (alphas**1.0)                 # γ=1
    r = weights / weights.sum()                 # routing probabilities
    loss = torch.dot(r, latencies)              # weighted latency
    loss.backward()
    opt.step()
    loss_history.append(loss.item())

# 4) Visualize results
best_idx = torch.softmax(logits,0).argmax().item()
print(f"→ Optimal intermediary: {inters[best_idx]}")
print("Final allocations (c):")
for i,cval in zip(inters, torch.softmax(logits,0).tolist()):
    print(f"  via {i}:  {cval:.3f}, total path len = {latencies[inters.index(i)]:.3f}")

plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Weighted latency")
plt.title("Gradient‐based beam allocation")
plt.show()
