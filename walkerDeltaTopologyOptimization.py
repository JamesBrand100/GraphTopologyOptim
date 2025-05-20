import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import myUtils
import pdb
from torch.utils.tensorboard import SummaryWriter   # <-- import
import copy 

# ─── 1) Configurable parameters ────────────────────────────────────────────────
torch.manual_seed(0)
beam_budget = 4      # sum of beam allocations per node
lr          = .01
epochs      = 20
numFlows = 20
maxDemand   = 1.0

gamma       = 3.0      # sharpness for alignment
max_hops    = 10        # how many hops to unroll
trafficScaling = 100000

#constellation params 
orbitRadius = 6.946e6   
numSatellites = 90
orbitalPlanes = 10
inclination = 80 
phasingParameter = 5

# ─── Prepare TensorBoard ───────────────────────────────────────────────────────
# every run goes under runs/<timestamp> by default
writer = SummaryWriter(comment=f"_flows={numFlows}_gamma={gamma}")

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

#create storage for loss and latency 
loss_history = []
total_latency = None
# ─── 6) Create differentiable process for learning beam allocations───────────────────────────────────────
for epoch in range(epochs):
    #Remove gradients 
    opt.zero_grad()

    logits = myUtils.build_full_logits(logits_feasible, feasible_indices, feasibleMask.shape)
    logits = torch.nn.functional.softplus(logits)

    # Implement logit normalization 
    # This converts our logits to valid values in the context of beam allocations 
    # it might possibly remove....the beam restrictions

    hyperBase = 20000

    #calculate temperature for mapping based on current levels of loss 
    #if we have no latency right now, use temperature of 1 
    if(not total_latency):
        c = myUtils.symmetric_sinkhorn(logits, num_iters=3, normLim = beam_budget, temperature = 1)
        gamma = 1
    #if we can use latency for temperature calculation 
    else:
        c = myUtils.plus_sinkhorn(logits, num_iters=int(3*20000/total_latency.item()), normLim = beam_budget, temperature = int(7*hyperBase/total_latency.item()))
        gamma = 3 * hyperBase * total_latency.item() 

    # ─── Compute Routing matrix, R[i,d,i] ──────────────────────────────────────────────
    alpha_sharp = similarityMetric ** gamma

    numer = c.unsqueeze(1) * alpha_sharp     # [i,d,i]
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

    T_store = copy.deepcopy(T_current)

    for _ in range(max_hops):
        # Compute traffic sent: traffic[d, i] * R[i, d, j] → output: [d, i, j]
        # Need to broadcast T_current to [d, i, 1] to match R[i, d, j]
        traffic_sent = T_current[:, :, None] * R.permute(1, 0, 2)  # [d, i, j]

        # Compute latency for all traffic
        scaledDist = dmat / c
        #compute one hop 
        latency = torch.einsum('dij,ij->', traffic_sent, dmat)  # Scalar
        #then, affected by allocations 
        total_latency += latency

        # Propagate: sum over i (source), traffic now at j for each destination d
        # T_next[d, j] = sum_i T_current[d, i] * R[i, d, j]
        T_next = torch.einsum('di,dij->dj', T_current, R.permute(1, 0, 2))  # [d, j]

        T_current = T_next

    loss_history.append(total_latency.item())
    total_latency.backward(retain_graph=True)

    holdLogits = logits.clone()
    foldFLogits = logits_feasible.clone()

    torch.nn.utils.clip_grad_norm_(logits_feasible, max_norm=10.0)
    opt.step()
    print("Logit magnitude diff, all: " + str( torch.sum(torch.abs(holdLogits - logits))))
    print("Logit magnitude diff, feasible: " + str( torch.sum(torch.abs(foldFLogits - logits_feasible))))

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_latency.item():.4f}")

    # ——— log to TensorBoard —————————————————————————————————————————————
    writer.add_scalar('Loss/TotalLatency', total_latency.item(), epoch)
    
# ─── Finish up ─────────────────────────────────────────────────────────────────

originalC = torch.clone(c)

print("Diff")
print( torch.sum(torch.abs(c - originalC)))

#pdb.set_trace() 

print("Diagonal Symmetry Score")
print(myUtils.diagonal_symmetry_score(c))
print("Row/Col Normalization Score")
print(myUtils.normalization_score(c))
print("Actual Connections Made in Rounded")
print(torch.sum(c))
print("Actual Connections Made in Non-Rounded")
print(torch.sum(originalC))

#take only reasonable entries 
print(torch.sum(c >0.9))
#pdb.set_trace() 

originalC = originalC[ originalC > 0.04]

plt.hist(originalC.detach().numpy(), bins=20, range=(0, 1), edgecolor='black')
plt.show()

#hardcode / process connections 
c[c > 0.5] = 1
c[c < 0.5] = 0

#process based on numpy components
c = c.detach().numpy()
dmat = dmat.detach().numpy()
T_store = T_store.detach().numpy()

#my method stuff 
diffMethodProp = np.multiply(c, dmat )
diffMethodProp[diffMethodProp == 0] = 1000
diffMethodLatencyMat = myUtils.size_weighted_latency_matrix(diffMethodProp, T_store)
print("My method size weighted latency")
print(np.sum(diffMethodLatencyMat))

#get baseline stuff next 
#first build out connectivity 
gridPlusConn = myUtils.build_plus_grid_connectivity(positions,
                                                    orbitalPlanes,
                                                    int(numSatellites/orbitalPlanes)) 
#then... get then ext 
gridPlusProp = np.multiply(gridPlusConn, dmat )
gridPlusProp[gridPlusProp == 0] = 1000
gridPlusLatencyMat = myUtils.size_weighted_latency_matrix(gridPlusProp, T_store)

#my method stuff 
print("Grid plus size weighted latency")
print(np.sum(gridPlusLatencyMat))

writer.close()