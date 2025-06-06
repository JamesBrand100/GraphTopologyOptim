import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import myUtils
import pdb

# ─── 1) Configurable parameters ────────────────────────────────────────────────
torch.manual_seed(0)
beam_budget = 3.0      # sum of beam allocations per node
lr          = 1 #0.1
epochs      = 500
numFlows = 100
maxDemand   = 1.0

gamma       = 3.0      # sharpness for alignment
max_hops    = 10        # how many hops to unroll
trafficScaling = 100000

#constellation params 
orbitRadius = 6.946e6   
numSatellites = 360
orbitalPlanes = 10
inclination = 80 
phasingParameter = 5

# ─── 2) Create node positions and demands────────────────────────────────────────
positions, vecs = myUtils.generateWalkerStarConstellationPoints(numSatellites,
                                              inclination,
                                              orbitalPlanes,
                                              phasingParameter,
                                              orbitRadius)
positions = np.reshape(positions, [np.shape(positions)[0]*np.shape(positions)[1], np.shape(positions)[2]])

src_indices = np.random.choice(np.arange(numSatellites), size=numFlows, replace=False)
available   = np.setdiff1d(np.arange(numSatellites), src_indices)
dst_indices = np.random.choice(available, size=numFlows, replace=False)

# per (s,d) demand
demandVals = torch.rand(numFlows)*trafficScaling
demands = { (int(s),int(d)): demandVals[i]
            for i,(s,d) in enumerate(zip(src_indices, dst_indices)) }

# ─── 3) Precompute one-hop distances & adjacencies ─────────────────────────────
dmat = torch.cdist(torch.from_numpy(positions).to(dtype=torch.float32), torch.from_numpy(positions).to(dtype=torch.float32))  / 3e8                            # [N,N]

feasibleMask = myUtils.check_los(positions, positions)  # (dmat <= max_dist) & (~torch.eye(numSatellites, dtype=bool))   # feasible edges
#dmat #= dmat * adj_mask.float()                                        # zero out infeasible

# ─── 4) Compute “in-line-ness” α for every (g,d,i) ────────────────────────────
#need to look at all components for g, as others may be "g" within the path
#need to also insert: if d(g,d) < d(g,i), alpha = 0
similarityMetric = myUtils.batch_similarity_metric(positions, positions[dst_indices], positions)

#print(np.sum(np.abs(simiMetric) != 1))
#pdb.set_trace()

# ─── 5) Learnable beam logits per node, LOS restriction enabled ───────────────────────────────────────
#logits = torch.zeros(numSatellites, numSatellites, requires_grad=True)
#logits = logits.masked_fill(~feasibleMask, -1e9)
#opt    = torch.optim.SGD([logits[feasibleMask]], lr=lr)

# ─── 5) Learnable beam logits per node, LOS restriction enabled ───────────────────────────────────────
#Make restriction on logits based on feasibility...
feasible_indices = feasibleMask.nonzero()  # Shape: [2, num_feasible]
logits_feasible = torch.nn.Parameter(torch.zeros(len(feasible_indices[0]), dtype=torch.float))
opt = torch.optim.SGD([logits_feasible], lr=lr)

#create storage for loss 
loss_history = []

# ─── 6) Create differentiable process for learning beam allocations───────────────────────────────────────
for epoch in range(epochs):
    #Remove gradients 
    opt.zero_grad()

    logits = myUtils.build_full_logits(logits_feasible, feasible_indices, feasibleMask.shape)
    logits = torch.nn.functional.softplus(logits)

    # Implement logit normalization 
    # This converts our logits to valid values in the context of beam allocations 
    # it might possibly remove....the beam restrictions
    c = myUtils.symmetric_sinkhorn(logits, num_iters=1)

    # ─── Compute Routing matrix, R[i,d,i] ──────────────────────────────────────────────
    alpha_sharp = similarityMetric ** gamma

    numer = c.unsqueeze(1) * alpha_sharp * beam_budget     # [i,d,i]
    denom = numer.sum(dim=2, keepdim=True)    # [i,d,1]
    Rsub = numer / (denom + 1e-9)                # [i,d,i]
    R = torch.zeros([numSatellites,numSatellites,numSatellites], dtype=torch.float)
    R[:,dst_indices,:] = Rsub

    #within routing matrix, zero out outgoing edges for when we are at our destination 
    R[dst_indices, dst_indices, :] = 0

    #also for routing matrix, need to set proper conditional routing scheme
    #that is, if the final destination is in view, that routing # is 1 

    #how do we establish if its in view? from feasible indices...

    # ─── Latency unrolling ────────────────────────────────────────────────────
    total_latency = 0.0

    #indexing T, means [dst, satNum]
    T_current = torch.zeros((numSatellites, numSatellites))
    #initialize it the src_indices as the actual demand vals 
    T_current[dst_indices, src_indices] = demandVals

    for _ in range(max_hops):
        # Compute traffic sent: traffic[d, i] * R[i, d, j] → output: [d, i, j]
        # Need to broadcast T_current to [d, i, 1] to match R[i, d, j]
        traffic_sent = T_current[:, :, None] * R.permute(1, 0, 2)  # [d, i, j]

        # Compute latency for all traffic
        latency = torch.einsum('dij,ij->', traffic_sent, dmat)  # Scalar
        total_latency += latency

        # Propagate: sum over i (source), traffic now at j for each destination d
        # T_next[d, j] = sum_i T_current[d, i] * R[i, d, j]
        T_next = torch.einsum('di,dij->dj', T_current, R.permute(1, 0, 2))  # [d, j]

        T_current = T_next

        #print("Current Traffic")
        #print(torch.sum(T_current))
        #pdb.set_trace()

    # print("Negative terms? ")
    # print(torch.sum(T_current < 0))
    # print(torch.sum(R < 0))
    # print(torch.sum(traffic_sent < 0 ))
    # print(torch.sum(logits < 0 ))
    # print(torch.sum(logits_feasible < 0 ))

    #Penalize traffic that hasnt reached its end destination. they are on relatively same scale (good)
    #pdb.set_trace()

    #total_latency+=torch.sum(T_current)

    loss_history.append(total_latency.item())
    total_latency.backward(retain_graph=True)

    holdLogits = logits.clone()
    foldFLogits = logits_feasible.clone()

    torch.nn.utils.clip_grad_norm_(logits_feasible, max_norm=10.0)
    opt.step()
    print("Logit magnitude diff, all: " + str( torch.sum(torch.abs(holdLogits - logits))))
    print("Logit magnitude diff, feasible: " + str( torch.sum(torch.abs(foldFLogits - logits_feasible))))

    #pdb.set_trace()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_latency.item():.4f}")
