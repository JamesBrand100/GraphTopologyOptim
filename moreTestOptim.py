import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import pdb

# ─── 1) Configurable parameters ────────────────────────────────────────────────
torch.manual_seed(0)
num_nodes   = 75
max_dist    = 0.3      # maximum direct-hop distance
beam_budget = 3.0      # sum of beam allocations per node
lr          = 0.1
epochs      = 30
numSrcs     = 30
numDsts     = 30
maxDemand   = 1.0

gamma       = 2.0      # sharpness for alignment
max_hops    = 5        # how many hops to unroll

# ─── 2) Random node positions & demands ────────────────────────────────────────
positions = torch.rand(num_nodes, 2)
src_indices = np.random.choice(np.arange(num_nodes), size=numSrcs, replace=False)
available   = np.setdiff1d(np.arange(num_nodes), src_indices)
dst_indices = np.random.choice(available, size=numDsts, replace=False)

# per (s,d) demand
demandVals = torch.rand(numSrcs)
demands = { (int(s),int(d)): demandVals[i]
            for i,(s,d) in enumerate(zip(src_indices, dst_indices)) }

# ─── 3) Precompute one-hop distances & adjacencies ─────────────────────────────
dmat = torch.cdist(positions, positions)                              # [N,N]
adj_mask = (dmat <= max_dist) & (~torch.eye(num_nodes, dtype=bool))   # feasible edges
dmat = dmat * adj_mask.float()                                        # zero out infeasible

# ─── 4) Compute “in-line-ness” α for every (g,d,i) ────────────────────────────
vec_gi = positions.unsqueeze(1) - positions.unsqueeze(0)    # [g,i,2]
norm_gi = vec_gi.norm(dim=2, keepdim=True).clamp(min=1e-6)
unit_gi = vec_gi / norm_gi  # [g,i,2]

norm_gd  = norm_gi.transpose(0,1)  # same magnitudes, switched axes
unit_gd  = unit_gi.clone().transpose(0,1)  # [g,d,2]

# ─── 5) Learnable beam logits per node ───────────────────────────────────────
logits = torch.zeros(num_nodes, num_nodes, requires_grad=True)
opt    = torch.optim.SGD([logits], lr=lr)

loss_history = []

for epoch in range(epochs):
    opt.zero_grad()

    logits_sym = 0.5 * (logits + logits.transpose(0, 1))

    # Convert logits to valid beam allocations (softmax + masking)
    logits_masked = logits_sym.masked_fill(~adj_mask, -1e9)
    c_unnorm      = torch.softmax(logits_masked, dim=1) * beam_budget
    c = c_unnorm.clamp(0, 1)

    # ─── Routing matrix R[g,d,i] ──────────────────────────────────────────────
    alpha = (unit_gi.unsqueeze(1) * unit_gd.unsqueeze(2)).sum(-1).clamp(min=0)  # [g,d,i]
    pdb.set_trace() 
    alpha_sharp = alpha ** gamma

    numer = c.unsqueeze(1) * alpha_sharp      # [g,d,i]
    denom = numer.sum(dim=2, keepdim=True)    # [g,d,1]
    R = numer / (denom + 1e-9)                # [g,d,i]

    # ─── Latency unrolling ────────────────────────────────────────────────────
    L = dmat.clone()  # initial one-hop latency
    for _ in range(max_hops):
        L_expand = L.transpose(0,1).unsqueeze(0).expand(num_nodes, num_nodes, num_nodes)
        L_new = (R * (dmat.unsqueeze(1) + L_expand)).sum(dim=2)  # [g,d]
        L = L_new

    # ─── Compute total demand-weighted latency ────────────────────────────────
    total_loss = 0.0
    for (s,d), val in demands.items():
        total_loss += L[s,d] * val

    loss_history.append(total_loss.item())
    total_loss.backward()
    opt.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")

# ─── Optional: Plot loss curve ────────────────────────────────────────────────
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Demand-Weighted Latency")
plt.title("Training Loss")
plt.grid(True)
plt.show()
