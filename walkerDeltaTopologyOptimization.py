import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import myUtils
import pdb
from torch.utils.tensorboard import SummaryWriter   # <-- import
import copy 
import random
# ─── 1) Configurable parameters ────────────────────────────────────────────────
# Python built-in
random.seed(0)
# NumPy
np.random.seed(0)
# Torch 
torch.manual_seed(0)

beam_budget = 4      # sum of beam allocations per node
lr          = .01
epochs      = 400 #40 epochs for very small constellation. more like ~100-150 epochs maybe for bigger constellation (~360 sats or somethin) 
numFlows = 150
maxDemand   = 1.0

gamma       = 3.0      # sharpness for alignment
max_hops    = 35       # how many hops to unroll
trafficScaling = 100000

#constellation params 
orbitRadius = 6.946e6   
numSatellites = 400
orbitalPlanes = 20
inclination = 80 
phasingParameter = 5

#init routing: {"LOSweight","FWPropDiff","FWPropBig", "LearnedLogit"}
routingMethod = "LearnedLogit"

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

great_circle_prop = myUtils.great_circle_distance_matrix_cartesian(positions, 6.946e6) / (3e8)

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

#new computation for sim metric 
positions = torch.from_numpy(positions)
similarityMetric = myUtils.batch_similarity_metric_new(positions, positions[dst_indices], positions).float()

#old computation for sim metric 
#similarityMetric = myUtils.batch_similarity_metric(positions, positions[dst_indices], positions)

#pdb.set_trace()

#print(np.sum(np.abs(simiMetric) != 1))
#pdb.set_trace()

# ─── 5) Learnable beam logits per node, LOS restriction enabled ───────────────────────────────────────
#logits = torch.zeros(numSatellites, numSatellites, requires_grad=True)
#logits = logits.masked_fill(~feasibleMask, -1e9)
#opt    = torch.optim.SGD([logits[feasibleMask]], lr=lr)

# ─── 5) Learnable beam logits per node, LOS restriction enabled ───────────────────────────────────────
#Make restriction on logits based on feasibility...
 
feasible_indices = feasibleMask.nonzero()  # Shape: [2, num_feasible]
connectivity_logits = torch.nn.Parameter(torch.zeros(len(feasible_indices[0]), dtype=torch.float))
connOpt = torch.optim.SGD([connectivity_logits], lr=lr)

if(routingMethod == "LearnedLogit"):
    routing_logits = torch.nn.Parameter(torch.zeros(len(feasible_indices[0])*numFlows, dtype=torch.float))
    routeOpt = torch.optim.SGD([routing_logits], lr=lr)   

#create storage for loss and latency 
loss_history = []
total_latency = None
# ─── 6) Create differentiable process for learning beam allocations───────────────────────────────────────
for epoch in range(epochs):

    if(routingMethod == "LearnedLogit"):
        routeOpt.zero_grad()
    
    #Remove gradients 
    connOpt.zero_grad()
    
    connLogits = myUtils.build_full_logits(connectivity_logits, feasible_indices, feasibleMask.shape)
    connLogits = torch.nn.functional.softplus(connLogits)

    # Implement logit normalization 
    # This converts our logits to valid values in the context of beam allocations 
    # it might possibly remove....the beam restrictions

    hyperBase = None

    #calculate temperature for mapping based on current levels of loss 
    #if we have no latency right now, use temperature of 1 
    if(not total_latency):
        c = myUtils.plus_sinkhorn(connLogits, num_iters=3, normLim = beam_budget, temperature = 1)
        gamma = 1
    #if we can use latency for temperature calculation 
    else:
        if(not hyperBase):
            hyperBase = total_latency.item()
        c = myUtils.plus_sinkhorn(connLogits, num_iters=int(6*epoch/epochs + 1), normLim = beam_budget, temperature = int(3*epoch/epochs + 1))
        gamma = int(epoch/epochs + 1)

    # ─── Compute Routing matrix, R[i,d,i] ──────────────────────────────────────────────
    #work with routing logits 
    if(routingMethod == "LearnedLogit"):
        #build logits similarly for routing
        routeLogits = myUtils.build_full_logits_route(routing_logits, feasible_indices, feasibleMask.shape, dst_indices)
        #create soft plus for routing as well, can only have positive values 
        #routeLogits = torch.nn.functional.softplus(routeLogits)

        #set the indces to 0 so that sinks are properly initialized 
        routeLogits[dst_indices, dst_indices, :] = -1e12

        #create soft max across 2nd dimension to put into valid prob distribution for routing 
        # @ epoch = 0 -> temp = 1
        # @ epoch = max -> temp = 0 
        routeLogits = routeLogits / (1 - epoch/epochs) # This is the temperature scaling
        routeLogits = torch.softmax(routeLogits, dim = 2)

        #normalize across 3rd dimension ("outflow" for each)
        #routeLogits = routeLogits / (torch.sum(routeLogits, dim = 2) + 1e-15)

        R = routeLogits

        #print("Num Sharp: " +str(torch.sum(R > 0.8)))

    if routingMethod == "FWPropBig":
        #Compute routing matrix using floyd warshall first with beam weighted distance 
        distance_matrix = myUtils.FW( dmat / (c.detach().numpy()+1e-15))
        #After we have distance matrix, convert it to proper routing table 
        R = myUtils.proportional_routing_table(distance_matrix)

    if routingMethod == "FWPropDiff":
        #format our dmat properly 
        FWdmat = dmat
        FWdmat[~feasibleMask] = float('inf')

        #Compute routing matrix using floyd warshall first with beam weighted distance 
        #Have arbitrary scaling to ensure logsumexp function works correctly 
        distance_matrix = myUtils.differentiable_floyd_warshall(1000*FWdmat/ (c+1e-8))
        
        #After we have distance matrix, convert it to proper routing table 
        R = myUtils.proportional_routing_table(distance_matrix)

    if routingMethod == "LOSweight":
        alpha_sharp = similarityMetric ** gamma
        numer = c.unsqueeze(1) * alpha_sharp     # [i,d,i] 
        denom = numer.sum(dim=2, keepdim=True)    # [i,d,1]
        Rsub = numer / (denom + 1e-15)                # [i,d,i]

        R = torch.zeros([numSatellites,numSatellites,numSatellites], dtype=torch.float)
        R[:,dst_indices,:] = Rsub

        #within routing matrix, zero out outgoing edges for when we are at our destination 
        R[dst_indices, dst_indices, :] = 0


    # #sharpen routing components:
    #R = R ** (gamma)
    #R = R / (R.sum(dim=-1,keepdim = True) + 1e-15)

    # ─── Latency unrolling ────────────────────────────────────────────────────
    total_latency = 0.0

    #indexing T, means [dst, satNum]
    T_current = torch.zeros((numSatellites, numSatellites))

    #initialize it the src_indices as the actual demand vals 
    T_current[dst_indices, src_indices] = demandVals

    T_store = copy.deepcopy(T_current)

    #assert torch.allclose(R.sum(dim=2)[src_indices, dst_indices], torch.ones_like(torch.from_numpy(dst_indices), dtype=torch.float), atol=1e-3)

    for _ in range(max_hops):

        # Compute traffic sent: traffic[d, i] * R[i, d, j] → output: [d, i, j]
        # Need to broadcast T_current to [d, i, 1] to match R[i, d, j]
        traffic_sent = T_current[:, :, None] * R.permute(1, 0, 2)  # [d, i, j]

        # Compute latency for all traffic
        scaledDist = dmat / (c + 1e-6)

        #compute one hop 
        latency = torch.einsum('dij,ij->', traffic_sent, scaledDist)  # Scalar

        #then, affected by allocations 
        total_latency += latency

        if routingMethod == "LOSweight":
            # ── Step: Apply stuck traffic penalty ───────────────────────
            penalty_per_unit = 3 # or whatever penalty makes sense

            # Step: Compute outgoing routing per destination and node
            outflow = R.permute(1, 0, 2).sum(dim=2)  # [d, i]

            # # Step: Construct destination mask (True where i == d)
            # # Shape: [d, i]
            # dest_mask = torch.eye(numSatellites, dtype=torch.bool)[None, :, :].expand(numSatellites, -1, -1)  # [d, i, i]
            # at_dest_mask = dest_mask.any(dim=2)  # [d, i]

            # Step: Construct destination mask (True where i == d)
            # Shape: [d, i]
            at_dest_mask = torch.eye(numSatellites, dtype=torch.bool) # Shape [numSatellites, numSatellites]

            # Step: Identify where routing gives zero outgoing flow AND we are NOT at destination
            stuck_mask = (outflow <= 1e-6) & (~at_dest_mask)  # [d, i]

            # Step: Extract stuck traffic
            stuck_traffic = T_current * stuck_mask  # [d, i]

            # Apply penalty for stuck traffic, scaled on ending distance 
            stuck_penalty = (stuck_traffic * great_circle_prop ).sum() * penalty_per_unit

            #stuck_penalty = stuck_traffic.sum() * penalty_per_unit
            total_latency += stuck_penalty

        # Propagate: sum over i (source), traffic now at j for each destination d
        # T_next[d, j] = sum_i T_current[d, i] * R[i, d, j]
        T_next = torch.einsum('di,dij->dj', T_current, R.permute(1, 0, 2))  # [d, j]

        #
        T_current = T_next

    # if(hyperBase):
    #     β = hyperBase / total_latency.item()
    #     entropy = -torch.sum(c * torch.log(c + 1e-8))
    #     total_latency = total_latency - β * entropy

    loss_history.append(total_latency.item())
    total_latency.backward(retain_graph=True)

    holdLogits = connLogits.clone()
    foldFLogits = connectivity_logits.clone()

    torch.nn.utils.clip_grad_norm_(connectivity_logits, max_norm=10.0)
    connOpt.step()

    if(routingMethod == "LearnedLogit"):
        torch.nn.utils.clip_grad_norm_(routing_logits, max_norm=10.0)
        routeOpt.step()

    print("Logit magnitude diff, all: " + str( torch.sum(torch.abs(holdLogits - connLogits))))
    print("Logit magnitude diff, feasible: " + str( torch.sum(torch.abs(foldFLogits - connectivity_logits))))

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_latency.item():.4f}")

    # ——— log to TensorBoard —————————————————————————————————————————————
    writer.add_scalar('Loss/TotalLatency', total_latency.item(), epoch)
    
# ─── Finish up ─────────────────────────────────────────────────────────────────
R = R[R>0.01]
plt.hist(R.detach().numpy(), bins=100, range=(0, 1), edgecolor='black')
plt.show()

originalC = torch.clone(c)
pdb.set_trace()


print("Diff")
print( torch.sum(torch.abs(c - originalC)))

#pdb.set_trace() 

print("Diagonal Symmetry Score")
print(myUtils.diagonal_symmetry_score(c))
print("Row/Col Normalization Score")
print(myUtils.normalization_score(c, ref=4.0, epsilon=1e-8))
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

#myUtils.plot_connectivity(positions, c, figsize=(8,8))

#my method stuff 
diffMethodProp = np.multiply(c, dmat )
diffMethodProp[diffMethodProp == 0] = 1000
np.fill_diagonal(diffMethodProp, 0)
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
np.fill_diagonal(gridPlusProp, 0)
gridPlusLatencyMat = myUtils.size_weighted_latency_matrix(gridPlusProp, T_store)

#my method stuff 
print("Grid plus size weighted latency")
print(np.sum(gridPlusLatencyMat))

writer.close()