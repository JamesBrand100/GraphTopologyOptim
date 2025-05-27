import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import myUtils
import pdb
from torch.utils.tensorboard import SummaryWriter   # <-- import
import copy 
import random
import json
import argparse


def run_simulation(numFlows, 
                   numSatellites, 
                   orbitalPlanes, 
                   routingMethod, 
                   epochs, 
                   lr, 
                   fileToSaveTo):

    # --- 0) Define Device (NEW) ---------------------------------------------------
    # This is the first and most important change: define the device (GPU or CPU)   
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--- Using GPU: {torch.cuda.get_device_name(0)} ---")
    else:
        device = torch.device("cpu")
        print("--- Using CPU ---")

    # ─── 1) Configurable parameters ────────────────────────────────────────────────
    # Python built-in
    random.seed(0)
    # NumPy
    np.random.seed(0)
    # Torch 
    torch.manual_seed(0)

    #training params
    #we commented out the ones that are derived from user inputs 

    #lr          = .01
    #epochs      = 200 #40 epochs for very small constellation. more like ~100-150 epochs maybe for bigger constellation (~360 sats or somethin) 
    gamma       = 3.0      # sharpness for alignment

    #simulation params 
    trafficScaling = 100000
    max_hops    = 20       # how many hops to unroll
    maxDemand   = 1.0
    #numFlows = 30
    beam_budget = 4      # sum of beam allocations per node

    #constellation params 
    orbitRadius = 6.946e6   
    #numSatellites = 100
    #orbitalPlanes = 10
    inclination = 80 
    phasingParameter = 5

    #init routing: {"LOSweight","FWPropDiff","FWPropBig", "LearnedLogit"}
    #routingMethod = "LearnedLogit"

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
    #great_circle_prop = torch.from_numpy(great_circle_prop).to(dtype=torch.float32).to(device) # NEW

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
    connOpt = torch.optim.AdamW([connectivity_logits], lr=lr)

    if(routingMethod == "LearnedLogit"):
        routing_logits = torch.nn.Parameter(torch.zeros(len(feasible_indices[0])*numFlows, dtype=torch.float))
        routeOpt = torch.optim.AdamW([routing_logits], lr=lr)   

    #create storage for loss and latency 
    loss_history = []
    total_latency = None
    # ─── 6) Create differentiable process for learning beam allocations───────────────────────────────────────
    for epoch in range(epochs):

        if(routingMethod == "LearnedLogit"):
            routeOpt.zero_grad()

        #Remove gradients 
        connOpt.zero_grad()
        

        # Implement logit normalization 
        # This converts our logits to valid values in the context of beam allocations 
        # it might possibly remove....the beam restrictions

        #hyperBase = None

        #calculate temperature for mapping based on current levels of loss 
        #if we have no latency right now, use temperature of 1 
        # if(not total_latency):
        #     c = myUtils.plus_sinkhorn(connLogits, num_iters=3, normLim = beam_budget, temperature = 1)
        #     gamma = 1
        # #if we can use latency for temperature calculation 
        # else:
        #     if(not hyperBase):
        #         hyperBase = total_latency.item()

        #logit conversion  
        connLogits = myUtils.build_full_logits(connectivity_logits, feasible_indices, feasibleMask.shape)
        connLogits = torch.nn.functional.softplus(connLogits)
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

        #initialize traffic components 
        #indexing T, means [dst, satNum]
        T_current = torch.zeros((numSatellites, numSatellites))
        #initialize it the src_indices as the actual demand vals 
        T_current[dst_indices, src_indices] = demandVals
        T_store = copy.deepcopy(T_current)

        #hoppity hop 
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

        torch.nn.utils.clip_grad_norm_(connectivity_logits, max_norm=1.0)
        connOpt.step()

        if(routingMethod == "LearnedLogit"):
            torch.nn.utils.clip_grad_norm_(routing_logits, max_norm=1.0)
            routeOpt.step()

        print("Logit magnitude diff, all: " + str( torch.sum(torch.abs(holdLogits - connLogits))))
        print("Logit magnitude diff, feasible: " + str( torch.sum(torch.abs(foldFLogits - connectivity_logits))))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_latency.item():.4f}")

        # ——— log to TensorBoard —————————————————————————————————————————————
        writer.add_scalar('Loss/TotalLatency', total_latency.item(), epoch)
        
    # ─── Finish up ─────────────────────────────────────────────────────────────────

    #Create Metrics
    diagSym = myUtils.diagonal_symmetry_score(c).detach().numpy()
    normScore = myUtils.normalization_score(c, ref=4.0, epsilon=1e-8).detach().numpy()

    #hardcode / process connections 
    c[c > 0.5] = 1
    c[c < 0.5] = 0

    #process based on numpy components
    c = c.detach().numpy()
    dmat = dmat.detach().numpy()
    T_store = T_store.detach().numpy()

    #compute weighted latency for my method 
    #create distance based on combined conn. & distance 
    diffMethodProp = np.multiply(c, dmat )
    #simulate lack of conn. with high # 
    diffMethodProp[diffMethodProp == 0] = 1000
    #replace diagonal with 0 to prevent self loops
    np.fill_diagonal(diffMethodProp, 0)
    #compute latency 
    diffMethodLatencyMat = myUtils.size_weighted_latency_matrix(diffMethodProp, T_store)
    diffSumLatency = np.sum(diffMethodLatencyMat)

    #get baseline stuff next 
    #same process, but we generate c in one step here 
    gridPlusConn = myUtils.build_plus_grid_connectivity(positions,
                                                        orbitalPlanes,
                                                        int(numSatellites/orbitalPlanes)) 
    gridPlusProp = np.multiply(gridPlusConn, dmat )
    gridPlusProp[gridPlusProp == 0] = 1000
    np.fill_diagonal(gridPlusProp, 0)
    gridPlusLatencyMat = myUtils.size_weighted_latency_matrix(gridPlusProp, T_store)
    gridPlusSumLatency = np.sum(gridPlusLatencyMat)

    #save all metrics to dictionary
    metrics = {}
    metrics["Diagonal Symmetry Score"] = diagSym
    metrics["Row/Col Normalization Score"] = normScore
    metrics["My method size weighted latency"] = diffSumLatency
    metrics["Grid plus size weighted latency"] = gridPlusSumLatency

    writer.close()

    # Save the dictionary to a file, if we give a proper name 
    if(fileToSaveTo != "None"):
        with open(fileToSaveTo+str(".json"), 'w') as f:
            json.dump(metrics, f)
    else:
        print("No file name given, metrics not saved, here they are:")
        print("Diff Method: "+str(diffSumLatency))
        print("Grid plus: "+str(gridPlusSumLatency))

    #Plots 
    #myUtils.plot_connectivity(positions, c, figsize=(8,8))
    # plt.hist(originalC.detach().numpy(), bins=20, range=(0, 1), edgecolor='black')
    # plt.show()
    # R = R[R>0.01]
    # plt.hist(R.detach().numpy(), bins=100, range=(0, 1), edgecolor='black')
    # plt.show()


    # print("Diagonal Symmetry Score")
    # print(myUtils.diagonal_symmetry_score(c))
    # print("Row/Col Normalization Score")
    # print(myUtils.normalization_score(c, ref=4.0, epsilon=1e-8))
    # print("Actual Connections Made in Rounded")
    # print(torch.sum(c))
    # print("Actual Connections Made in Non-Rounded")
    # print(torch.sum(originalC))

    # print("Diff")
    # print( torch.sum(torch.abs(c - originalC)))

    # #take only reasonable entries 
    # print(torch.sum(c >0.9))
    # #pdb.set_trace() 

    # print("My method size weighted latency")
    # print(np.sum(diffMethodLatencyMat))

# def run_main():
#     parser = argparse.ArgumentParser(description='Run the simulation')
#     parser.add_argument('--numFlows', type=int, default=30, help='Number of flows')
#     parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
#     parser.add_argument('--numSatellites', type=int, default=100, help='Number of satellites')
#     parser.add_argument('--orbitalPlanes', type=int, default=10, help='Number of orbital planes')
#     parser.add_argument('--routingMethod', type=str, default='LOSweight', choices=['LOSweight', 'FWPropDiff', 'FWPropBig', 'LearnedLogit'], help='Routing method')
#     parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
#     parser.add_argument('--fileName', type=str, default="None", help='File to save to <3, without json tag')

#     args = parser.parse_args()

#     run_simulation(args.numFlows, 
#                    args.epochs, 
#                    args.numSatellites, 
#                    args.orbitalPlanes, 
#                    args.routingMethod, 
#                    args.lr, 
#                    args.fileName)
    
def run_main():
    parser = argparse.ArgumentParser(description='Run the simulation')
    parser.add_argument('--numFlows', type=int, default=30, help='Number of flows')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--numSatellites', type=int, default=100, help='Number of satellites')
    parser.add_argument('--orbitalPlanes', type=int, default=10, help='Number of orbital planes')
    parser.add_argument('--routingMethod', type=str, default='LOSweight', choices=['LOSweight', 'FWPropDiff', 'FWPropBig', 'LearnedLogit'], help='Routing method')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--fileName', type=str, default="None", help='File to save to <3, without json tag')

    args = parser.parse_args()

    # Reordered arguments to match the run_simulation function signature
    run_simulation(args.numFlows,
                   args.numSatellites,       
                   args.orbitalPlanes,       
                   args.routingMethod,      
                   args.epochs,              
                   args.lr,
                   args.fileName)

if __name__ == '__main__':
    run_main()