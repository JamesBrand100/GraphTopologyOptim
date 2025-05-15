import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import pdb

# 1) Configurable parameters
torch.manual_seed(0)
num_nodes = 50
max_dist = 0.1     # maximum direct-hop distance; larger needs intermediary
beam_budget = 3.0  # sum of allocations per source
lr = 0.1
epochs = 200
numSrcs = 5
numDsts = 5 
maxDemand = 1

# 2) Random node positions (in unit square for visualization)
positions = torch.rand(num_nodes, 2)

# 3) Define sources, destinations, intermediaries, and demands
# 1) pick your sources
src_indices = np.random.choice(np.arange(num_nodes),size=numSrcs,replace=False)

# 2) form the pool of remaining nodes
available = np.setdiff1d(np.arange(num_nodes), src_indices)

# 3) pick your destinations from that pool
dst_indices = np.random.choice(available,size=numDsts,replace=False)

# & create intermediaries 
inter_indices = [i for i in range(num_nodes) if i not in src_indices + dst_indices]

#create demands 
demands = {} 
demandVals = np.random.rand(numSrcs)*maxDemand
for flowInd in range(numSrcs):
    demands[(src_indices[flowInd],dst_indices[flowInd])] = demandVals[flowInd]

# 4) Compute distance matrix
dmat = torch.cdist(positions, positions)  # (num_nodes, num_nodes)

# 5) Build feasible adjacency: only edges <= max_dist
adj_mask = (dmat <= max_dist) & (~torch.eye(num_nodes, dtype=bool))
dmat = dmat * adj_mask

# 6) Initialize raw logits for each (src -> any node) pair
# We'll optimize per src independently but package together
logits = torch.zeros(len(src_indices), num_nodes, requires_grad=True)

opt = torch.optim.SGD([logits], lr=lr)

loss_history = []

for epoch in range(epochs):
    opt.zero_grad()
    total_loss = 0.0
    # For each source-dest pair
    for sx, dx in zip(src_indices, dst_indices):
        # feasible neighbors of sx
        mask = adj_mask[sx]
        raw = logits[src_indices.index(sx)]
        # mask out infeasible by setting logit to large negative
        raw_masked = torch.where(mask, raw, torch.tensor(-1e9))
        c = torch.softmax(raw_masked, dim=0)  # allocations over neighbors
        # compute two-hop latencies via each neighbor k: d(sx,k) + d(k,dx)
        lat_k = dmat[sx] + dmat[:, dx]
        # mask only keep k in mask
        lat_k = lat_k * mask.float() + (~mask).float() * 1e6
        # routing weights
        r = c  # gamma=1, and c already sums to 1
        loss_sd = torch.dot(r, lat_k)
        total_loss += demands[(sx, dx)] * loss_sd
    total_loss.backward()
    opt.step()
    loss_history.append(total_loss.item())

# 7) Plot results
plt.figure(figsize=(6,6))
pts = positions.numpy()
plt.scatter(pts[:,0], pts[:,1], c='black')
# label nodes
for i,(x,y) in enumerate(pts):
    plt.text(x+0.02, y+0.02, str(i), fontsize=10)

# draw colored edges for beam allocations
cmap = plt.cm.viridis
for idx, sx in enumerate(src_indices):
    raw = logits[idx].detach()
    mask = adj_mask[sx]
    raw_masked = torch.where(mask, raw, torch.tensor(-1e9))
    c = torch.softmax(raw_masked, dim=0).numpy()
    for j, weight in enumerate(c):
        if mask[j] and weight>1e-2:
            x1,y1 = pts[sx]
            x2,y2 = pts[j]
            plt.plot([x1,x2],[y1,y2], lw=3*weight, color=cmap(weight))

plt.title("Beam allocations (edge width+color ∝ weight)")
plt.axis('off')
plt.show()

# 8) Plot loss curve
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Weighted latency loss")
plt.title("Optimization progress")
plt.show()
