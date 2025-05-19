import torch
import math
import matplotlib.pyplot as plt
import numpy as np

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
γ           = 2.0      # sharpness for alignment
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
#TODO: not sure if infeasible should be "zerod" out 
dmat = dmat * adj_mask.float()                                        # zero out infeasible. not sure if this should be zerod out...

# ─── 4) Compute “in-line-ness” α for every (g,d,i) ────────────────────────────
#    α_{g,d,i} = cos( angle between vectors g→i and g→d )
vec_gi = positions.unsqueeze(1) - positions.unsqueeze(0)    # [g,i,2]
vec_gd = positions.unsqueeze(1) - positions.unsqueeze(0)    # [g,d,2], reuse dims later
# we will broadcast vec_gd[:,d,:] against vec_gi[g,i,:]
# pre-normalize
norm_gi = vec_gi.norm(dim=2, keepdim=True).clamp(min=1e-6)
unit_gi = vec_gi / norm_gi  # [g,i,2]

# ─── 5) Learnable beam logits per node ───────────────────────────────────────
# one logit vector per source, but you can generalize to every g→i pair
#TODO: non diagonal symmetry needs to be fixed 
logits = torch.zeros(num_nodes, num_nodes, requires_grad=True)
opt    = torch.optim.SGD([logits], lr=lr)

loss_history = []

for epoch in range(epochs):
    opt.zero_grad()

    #first, apply symmetric operations for good normalize....
    logits_sym = 0.5 * (logits + logits.transpose(0, 1))

    # 5a) From logits → soft allocations c_{g,i}, with beam_budget enforced via scaling
    #    mask infeasible edges by -1e9
    logits_masked = logits.masked_fill(~adj_mask, -1e9)
    #convert logits to probability distribution
    c_unnorm      = torch.softmax(logits_masked, dim=1) * beam_budget
    # optionally clamp c in [0,1]
    c = c_unnorm.clamp(0, 1)

    # ─── 6) Build per-destination routing matrices R_d [g,i] ───────────────────
    #First, we compute "in line ness"
    # We'll stack them into R tensor of shape [num_nodes, num_nodes, num_nodes]:
    #   R[g, d, i] = routing probability from g to i when final dest is d
    # First compute α[g,d,i]:
    #   we need unit vectors from g→i and g→d
    # Extract unit_gi = [g,i,2], and build unit_gd = [g,d,2]
    norm_gd  = norm_gi.transpose(0,1)  # hack: same magnitudes but different axes
    unit_gd  = unit_gi.clone()         # reuse shape [d,g,2]
    unit_gd  = unit_gd.transpose(0,1)  # now [g,d,2]
    α        = (unit_gi.unsqueeze(1) * unit_gd.unsqueeze(2)).sum(-1).clamp(min=0)  # [g,d,i]

    # apply sharpness γ
    αgdi_sharp = α ** γ  # [g,d,i]

    # numerator: c[g,i] * α[g,d,i]^γ
    numer = c.unsqueeze(1) * αgdi_sharp     # [g,d,i]
    denom = numer.sum(dim=2, keepdim=True)  # [g,d,1]
    R     = numer / (denom + 1e-9)          # [g,d,i]

    # ─── 7) Unroll latency L^{(t)} ─────────────────────────────────────────────
    # Initialize L⁽⁰⁾ = one-hop delay:
    #   we will build a D tensor [g,d,i] = d(g,i) + L⁽ᵗ⁾(i→d)
    L = dmat.clone()  # [g,i] but interpreted as initial latency to itself

    # We want L^{(t)}[g,d] so reshape to [g,d] by picking diagonal
    # Actually: for t+1: L_new[g,d] = Σ_i R[g,d,i] * ( dmat[g,i] + L[i,d] )
    for _ in range(max_hops):
        # expand L for indexing: L_prev[i,d] → shape [g,d,i]
        L_expand = L.transpose(0,1).unsqueeze(0).expand(num_nodes, num_nodes, num_nodes)
        # dmat[g,i] expand to [g,d,i]
        D_expand = dmat.unsqueeze(1).expand(num_nodes, num_nodes, num_nodes)
        # compute next
        L_next = ( R * (D_expand + L_expand) ).sum(dim=2)  # [g,d]
        L = L_next  # feed into next iteration

    # ─── 8) Compute weighted-latency loss ────────────────────────────────────────
    total_loss = torch.tensor(0., requires_grad=True)
    for (g,d), w in demands.items():
        total_loss = total_loss + w * L[g,d]
    total_loss.backward()
    opt.step()

    loss_history.append(total_loss.item())

# ─── 9) Visualize final beam allocations ──────────────────────────────────────
plt.figure(figsize=(6,6))
pts = positions.numpy()
plt.scatter(pts[:,0], pts[:,1], c='black')
for i,(x,y) in enumerate(pts):
    plt.text(x+0.01, y+0.01, str(i), fontsize=8)

c_fin = c.detach().numpy()
for g in range(num_nodes):
    for i in range(num_nodes):
        if adj_mask[g,i] and c_fin[g,i]>1e-2:
            x1,y1 = pts[g]; x2,y2 = pts[i]
            plt.plot([x1,x2],[y1,y2], lw=3*c_fin[g,i], color='C0')
plt.title("Learned beam allocations")
plt.axis('off')

plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Weighted latency")
plt.title("Training curve")
plt.show()
