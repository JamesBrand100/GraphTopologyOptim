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
from Baselines.funcedBaseline import *
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad



def run_simulation(numFlows, 
                   numSatellites, 
                   orbitalPlanes, 
                   routingMethod, 
                   epochs, 
                   lr, 
                   fileToSaveTo,
                   metricToOptimize = "latency",
                   demandDist = "popBased",
                   inLineness = "GCD"):
    #demand dist can be random or popBased
 
    """Hardware, randomization, environmental parameters initialization"""    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--- Using GPU: {torch.cuda.get_device_name(0)} ---")
    else:
        device = torch.device("cpu")
        print("--- Using CPU ---")

    # Python built-in
    random.seed(0)
    # NumPy
    np.random.seed(0)
    # Torch 
    torch.manual_seed(0)

    #training params
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
    inclination = 90 #used to be 80  
    phasingParameter = 5

    EARTH_MEAN_RADIUS = 6371.0e3

    #init routing: {"LOSweight","FWPropDiff","FWPropBig", "LearnedLogit"}
    #routingMethod = "LearnedLogit"

    # every run goes under runs/<timestamp> by default
    writer = SummaryWriter(comment=f"_flows={numFlows}_gamma={gamma}")

    """Create node positions and demands, so environmental configuration"""
    positions, vecs = myUtils.generateWalkerStarConstellationPoints(numSatellites,
                                                inclination,
                                                orbitalPlanes,
                                                phasingParameter,
                                                orbitRadius)
    positions = np.reshape(positions, [np.shape(positions)[0]*np.shape(positions)[1], np.shape(positions)[2]])

    #if demand dist is random, treat it as such
    if(demandDist == "random"):
        src_indices = np.random.choice(np.arange(numSatellites), size=numFlows, replace=False)
        available   = np.setdiff1d(np.arange(numSatellites), src_indices)
        dst_indices = np.random.choice(available, size=numFlows, replace=False)

        # per (s,d) demand
        demandVals = torch.rand(numFlows)*trafficScaling
        demands = { (int(s),int(d)): demandVals[i]
                    for i,(s,d) in enumerate(zip(src_indices, dst_indices)) }
        
    #otherwise, generate based on populationa
    elif(demandDist == "popBased"):

        #first, load in population and associated locations
        res3GridLocations = np.load("Data/InitData/Res3Grid.npy") 
        res3GridPops = np.load("Data/InitData/Res3Pops.npy") 

        #then, assign population demands to each position 
        #so, first get closest element of each res3GridLocations to satellite positions
        res3Distances = np.linalg.norm(res3GridLocations[:, np.newaxis] - positions, axis=2)
        closest_idx = np.argmin(res3Distances, axis=1)

        """Flow Generation"""
        #get argsum / closest satellite linked populations 
        satellitePopulations = np.bincount(closest_idx, weights=res3GridPops)

        #then, create corresponding flows 
        distances = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
        population_products = np.outer(np.log(satellitePopulations + 1), np.log(satellitePopulations+ 1))
        flows_matrix = np.zeros_like(distances, dtype=float)
        flows_matrix = \
            100 * population_products / \
            (distances**0.001 + 10e-7)
        flows_matrix[np.diag_indices(len(flows_matrix))] = 0 

        #remove symmetry so one can be "sources," one can be "destination" 
        flows_matrix = np.triu(flows_matrix)

        #get top 500
        num_top_flows = 500
        flat_flows = flows_matrix[flows_matrix > 0]
        
        if len(flat_flows) > num_top_flows:
            threshold_value = np.partition(flat_flows, -num_top_flows)[-num_top_flows]
            flows_matrix[flows_matrix < threshold_value] = 0
        else:
            pass

        src_indices, dst_indices = np.nonzero(flows_matrix)

        #scale it accordingly
        scaling_factor = 1e6 / np.sum(flows_matrix)
        flows_matrix *=scaling_factor

        # Use these indices to get the corresponding values from the matrix
        demandVals = torch.from_numpy(flows_matrix[src_indices, dst_indices]).float()

        #uplink / downlink latency generation 
        #get argsum / closest satellite linked populations
        row_indices = np.arange(res3Distances.shape[0])
        closest_dist = res3Distances[row_indices, closest_idx] 

        #then get the amount of latency traffic at each satellite 
        weightedSatelliteLatencies = np.bincount(closest_idx, weights=res3GridPops*closest_dist/(3e8))

        #normalize by satellite populations to get weighted latency to satellite 
        satellitePopulations = np.bincount(closest_idx, weights=res3GridPops)
        weightedSatelliteLatencies /= satellitePopulations + 1e-7

        #then use outer product to expand with itself, modeling uplink and downlink together
        weightedSatelliteLatencies = weightedSatelliteLatencies[:, np.newaxis] + weightedSatelliteLatencies[np.newaxis, :]

        #then, get final uplink/downlink latency
        uplinkDownlinkLatency = np.sum(weightedSatelliteLatencies * flows_matrix)

    """Differential parameter generation"""
    great_circle_prop = myUtils.great_circle_distance_matrix_cartesian(positions, 6.946e6) / (3e8)
    #great_circle_prop = torch.from_numpy(great_circle_prop).to(dtype=torch.float32).to(device) # NEW

    # ─── 3) Precompute one-hop distances & adjacencies ─────────────────────────────
    dmat = torch.cdist(torch.from_numpy(positions).to(dtype=torch.float32), torch.from_numpy(positions).to(dtype=torch.float32))  / 3e8                            # [N,N]

    feasibleMask = myUtils.check_los(positions, positions)  # (dmat <= max_dist) & (~torch.eye(numSatellites, dtype=bool))   # feasible edges
    #dmat #= dmat * adj_mask.float()                                        # zero out infeasible

    # ─── 4) Compute “in-line-ness” α for every (g,d,i) ────────────────────────────
    #need to look at all components for g, as others may be "g" within the path
    #need to also insert: if d(g,d) < d(g,i), alpha = 0

    #new computation for sim metric 
    #positions = torch.from_numpy(positions)
    
    if(inLineness == "GCD"):
        #newer similarity metric computation
        similarityMetric = myUtils.batch_similarity_metric_triangle_great_circle(positions, positions[dst_indices], positions).float()
    if(inLineness == "angle"):
        #old computation for sim metric 
        similarityMetric = myUtils.batch_similarity_metric(positions, positions[dst_indices], positions)
    
    feasible_indices = feasibleMask.nonzero()  # Shape: [2, num_feasible]
    connectivity_logits = torch.nn.Parameter(torch.zeros(len(feasible_indices[0]), dtype=torch.float))
    connOpt = torch.optim.AdamW([connectivity_logits], lr=lr)

    """Inter-epoch variables"""
    #create storage for loss and latency 
    loss_history = []
    total_latency = None

    # Create the reset functions
    update_state, reset_state, reset_to_best, find_closest_logit_history = myUtils.create_individual_param_reset_method(
        [(connectivity_logits, connOpt)],
        epochs_to_track=2
    )
    reset = False
    lossArr = []

    #set up multi objective loss 
    storeLatencyLoss = myUtils.FixedSizeArray(5)
    storeResilienceLoss = myUtils.FixedSizeArray(5)

    """Gradient Descent Loop"""
    #Create differentiable process for learning beam allocations
    for epoch in range(epochs):

        #if we are at the last epoch, load the most recent best one. 
        if(epoch == epochs - 1):
            reset_to_best()

        #Remove gradients 
        connOpt.zero_grad()
        
        #Differentiable feasibility mapping for connectivity 
        connLogits = myUtils.build_full_logits(connectivity_logits, feasible_indices, feasibleMask.shape)
        connLogits = torch.nn.functional.softplus(connLogits)
        c = myUtils.plus_sinkhorn(connLogits, num_iters=int(8*epoch/epochs + 1), normLim = beam_budget, temperature = int(4*epoch/epochs + 1))
        gamma = 3 * int(epoch/epochs + 1)

        # ─── Compute Routing matrix, R[i,d,i] ──────────────────────────────────────────────
        if routingMethod == "LOSweight":
            #alpha_sharp = similarityMetric ** gamma
            #numer = c.unsqueeze(1) * alpha_sharp     # [i,d,i] 

            #alpha_sharp = similarityMetric ** gamma
            numer = (c.unsqueeze(1) * similarityMetric ) ** gamma     # [i,d,i] 

            denom = numer.sum(dim=2, keepdim=True)    # [i,d,1] 
            Rsub = numer / (denom + 1e-15)                # [i,d,i] 

            #index placement is: currr x dst x next 
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

        totalDemand = torch.sum(demandVals).detach().numpy()
        T_store = copy.deepcopy(T_current)

        #compute penalty scaling
        if(metricToOptimize == "hops"):
            #compute sphere surface area 
            surfaceArea = 4*math.pi*orbitRadius ** 2
            avgDistBetweenPoints = surfaceArea / numSatellites

        #hoppity hop 
        """Latency/hop unrolling for loss"""
        for _ in range(max_hops):

            # Compute traffic sent: traffic[d, i] * R[i, d, j] → output: [d, i, j]
            # Need to broadcast T_current to [d, i, 1] to match R[i, d, j]

            #this is done as every routing portion for "next hop"
            #uses the same "traffic state" with regards to a specific destination
            #broadcasting : repeat element entry for multiple multiplications 
            traffic_sent = T_current[:, :, None] * R.permute(1, 0, 2)  # [d, i, j]

            # Compute latency for all traffic
            if(metricToOptimize == "latency"):
                scaledDist = dmat / (c + 1e-6)
            if(metricToOptimize == "hops"):
                scaledDist = dmat / (dmat + 1e-6)

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
                if(metricToOptimize == "latency"):
                    stuck_penalty = (stuck_traffic * great_circle_prop ).sum() * penalty_per_unit
                # If we are optimizing hops, then normalize with respect to that 
                if(metricToOptimize == "hops"):
                    stuck_penalty = (stuck_traffic * great_circle_prop / avgDistBetweenPoints ).sum() * penalty_per_unit

                #stuck_penalty = stuck_traffic.sum() * penalty_per_unit
                total_latency += stuck_penalty

            # Propagate: sum over i (source), traffic now at j for each destination d
            # T_next[d, j] = sum_i T_current[d, i] * R[i, d, j]
            T_next = torch.einsum('di,dij->dj', T_current, R.permute(1, 0, 2))  # [d, j]
            T_current = T_next

            #Generalized algebraic connectivity for additional loss component 
            if routingMethod == "LOSweight":
                #generate node importance storage 
                nodeImportance = torch.ones([numSatellites,numSatellites])
                #store node importance based on satellite populations 
                nodeImportance[torch.eye(numSatellites, dtype = torch.bool)] = torch.log(torch.from_numpy(satellitePopulations+1)).to(torch.float32)
                #compute algebraic connectivity in a differentiable manner 
                algebraic_connectivity = myUtils.calculate_generalized_algebraic_connectivity_torch(c,nodeImportance)
                
        #first store values         
        storeLatencyLoss.add_value(total_latency.detach())
        storeResilienceLoss.add_value(algebraic_connectivity.detach())

        #then compute respective normalized joint loss
        totalLoss = 100000*(0.4*total_latency/storeLatencyLoss.get_average() - 0.6*algebraic_connectivity/storeResilienceLoss.get_average())
        loss_history.append(totalLoss.item())
        
        #mtl_backward(losses=[total_latency, -algebraic_connectivity], features=c, aggregator=aggregator)
        #maybe this should be freed graph ...
        # retaining graph keeps the intermediate tensors in your graph in memory
        # these tensors are needed for the backward pass, so useful in RNNs where you use them multiple times 
        totalLoss.backward(retain_graph=False)

        #create info to double check we are not using infeasible connections 
        holdLogits = connLogits.clone()
        foldFLogits = connectivity_logits.clone()

        torch.nn.utils.clip_grad_norm_(connectivity_logits, max_norm=1.0)
        connOpt.step()

        update_state(epoch, totalLoss.item())
        lossArr.append(totalLoss.item())

        print("Logit magnitude diff, all: " + str( torch.sum(torch.abs(holdLogits - connLogits))))
        print("Logit magnitude diff, feasible: " + str( torch.sum(torch.abs(foldFLogits - connectivity_logits))))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {totalLoss.item():.4f}")

        # ——— log to TensorBoard —————————————————————————————————————————————
        writer.add_scalar('Loss/TotalLatency', totalLoss.item(), epoch)
        
    #Plot loss...
    #myUtils.plot_loss(np.arange(len(lossArr))+1, lossArr)

    #Save it as well
    # Define file names
    #file_name = "exampleLoss360Hops.npy"

    # ─── Finish up ─────────────────────────────────────────────────────────────────

    """Evalute and compare end solution"""
    #if we are optimizing hop count, then normalize distance matrix 
    if(metricToOptimize == "hops"):
        dmat = dmat/dmat 

    #Create Metrics
    diagSym = myUtils.diagonal_symmetry_score(c).detach().numpy()
    normScore = myUtils.normalization_score(c, ref=4.0, epsilon=1e-8).detach().numpy()

    #hardcode connections 
    c[c > 0.85] = 1
    c[c < 0.85] = 0

    #process based on numpy components
    c = c.detach().numpy()
    dmat = dmat.detach().numpy()
    T_store = T_store.detach().numpy()

    #get violations 
    print("Num violations:")
    print(np.sum(np.sum(c,axis = 1) > 4))

    #post processing for diff method 
    #compute weighted latency for my method 
    #create distance based on combined conn. & distance 
    diffMethodProp = np.multiply(c, dmat )
    #simulate lack of conn. with high # 
    diffMethodProp[diffMethodProp == 0] = 1000
    diffMethodProp[np.isnan(diffMethodProp)] = 1000
    np.fill_diagonal(diffMethodProp, 0)
    #compute latency 
    diffMethodLatencyMat = myUtils.size_weighted_latency_matrix_networkx(diffMethodProp, T_store)
    diffSumLatency = np.sum(diffMethodLatencyMat)

    #get baseline stuff next 
    #same process, but we generate c in one step here 
    gridPlusConn = myUtils.build_plus_grid_connectivity(positions,
                                                        orbitalPlanes,
                                                        int(numSatellites/orbitalPlanes))

    #post processing for gridPlusConn
    gridPlusConn = gridPlusConn*feasibleMask
    gridPlusProp = np.multiply(gridPlusConn, dmat)
    gridPlusProp[gridPlusProp == 0] = 1000
    gridPlusProp[np.isnan(gridPlusProp)] = 1000
    np.fill_diagonal(gridPlusProp, 0)
    gridPlusLatencyMat = myUtils.size_weighted_latency_matrix_networkx(gridPlusProp, T_store)
    gridPlusSumLatency = np.sum(gridPlusLatencyMat)

    #get motif data     
    motifSumLatency, graph = calculate_min_metric(
        1,
        numSatellites,
        orbitalPlanes,
        inclination,
        phasingParameter,
        orbitRadius,
        EARTH_MEAN_RADIUS,
        src_indices,
        dst_indices,
        demandVals,
        False,
        metricToOptimize,
        feasibleMask
    ) 

    #convert graph to valid conn components 
    node_positions, motifConn = myUtils.graph_to_matrix_representation(graph)
    #plot motif baseline 
    #myUtils.plot_connectivity(node_positions, conn, figsize=(8,8))

    # _, link_utilization, _= myUtils.calculate_network_metrics(conn, T_store)
    # myUtils.plot_connectivity(node_positions, conn, link_utilization, figsize=(8,8))

    #plot components for each of the approaches 

    #look at link utilization / network metrics for each 
    if(False):
        _, link_utilization_gplus, _= myUtils.calculate_network_metrics(gridPlusConn, T_store)
        myUtils.plot_connectivity(positions, gridPlusConn, link_utilization_gplus, figsize=(8,8), maxUtil = np.max(link_utilization_gplus))
        #myUtils.plot_edge_disjoint_paths_histogram(gridPlusConn)

        _, link_utilization_motif, _= myUtils.calculate_network_metrics(motifConn.astype(bool), T_store)
        myUtils.plot_connectivity(positions, motifConn.astype(bool), link_utilization_motif, figsize=(8,8), maxUtil = np.max(link_utilization_gplus))

        #compare data from motif to grid plus 
        pdb.set_trace()
        #myUtils.plot_edge_disjoint_paths_histogram(gridPlusConn)

        _, link_utilization_diff, _= myUtils.calculate_network_metrics(c.astype(bool), T_store)
        myUtils.plot_connectivity(positions, c.astype(bool), link_utilization_diff, figsize=(8,8), maxUtil = np.max(link_utilization_gplus))
        #myUtils.plot_edge_disjoint_paths_histogram(c.astype(bool))

        # Save the arrays
        np.savez("linkUtilData.npz",
                positions=positions,
                gridPlusConn=gridPlusConn,
                c=c,
                link_utilization_gplus=link_utilization_gplus,
                link_utilization_diff=link_utilization_diff)
    
    #save or print a bunch of metrics 
    pathsGridPlus, countsGridPlus, weightedEdgeDisjointGridPlus = myUtils.plot_edge_disjoint_paths_histogram(gridPlusConn, T_store, False)
    pathsMotif, countsMotif, weightedEdgeDisjointMotif  = myUtils.plot_edge_disjoint_paths_histogram(motifConn.astype(bool), T_store, False)
    pathsC, countsC, weightedEdgeDisjointC  = myUtils.plot_edge_disjoint_paths_histogram(c.astype(bool), T_store, False)
    
    #save all metrics to dictionary
    #save metrics for latency 
    metrics = {}
    metrics["Diagonal Symmetry Score"] = float(diagSym) 
    metrics["Row/Col Normalization Score"] = float(normScore)
    metrics["My method size weighted latency"] = float(diffSumLatency)# / totalDemand)
    metrics["Grid plus size weighted latency"] = float(gridPlusSumLatency)# / totalDemand)
    metrics["Motif size weighted latency"] = float(motifSumLatency)# / totalDemand)
    metrics["Total Demand"] = float(totalDemand)
    metrics["Objective"] = str(metricToOptimize)
    metrics["Latency Addendum"] = float(uplinkDownlinkLatency)

    #save metrics for resilience 
    metrics["Paths Grid Plus"] = pathsGridPlus
    metrics["Counts Grid Plus"] = countsGridPlus
    metrics["Counts Motif"] = countsMotif
    metrics["Counts C"] = countsC

    writer.close()

    print("Latency/Hops Metrics")
    print("Diff Method: "+str(diffSumLatency))
    print("Grid plus: "+str(gridPlusSumLatency))
    print("Motif: "+str(motifSumLatency))

    print(str(diffSumLatency/ totalDemand))
    print(str(gridPlusSumLatency/ totalDemand))
    print(str(motifSumLatency/ totalDemand))

    print("Edge disjoint paths")
    print(countsGridPlus)
    print(countsMotif)
    print(countsC)

    print("Weighted Edge Disjoint Paths")
    print(weightedEdgeDisjointGridPlus)
    print(weightedEdgeDisjointMotif)
    print(weightedEdgeDisjointC)

    # Save the dictionary to a file, if we give a proper name 
    if(fileToSaveTo != "None"):
        with open(fileToSaveTo+str(".json"), 'w') as f:
            json.dump(metrics, f)
    else:
        print("No file name given, metrics not saved, here they are:")
        print("Diff Method: "+str(diffSumLatency/ totalDemand))
        print("Grid plus: "+str(gridPlusSumLatency/ totalDemand))
        print("Motif: "+str(motifSumLatency/ totalDemand))

    #Plots for diff method 
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
    parser.add_argument('--numFlows', type=int, default=20, help='Number of flows')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--numSatellites', type=int, default=160, help='Number of satellites')
    parser.add_argument('--orbitalPlanes', type=int, default=10, help='Number of orbital planes')
    parser.add_argument('--routingMethod', type=str, default='LOSweight', choices=['LOSweight', 'FWPropDiff', 'FWPropBig', 'LearnedLogit'], help='Routing method')
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--fileName', type=str, default="None", help='File to save to <3, without json tag')
    parser.add_argument('--metricToOptimize', type=str, default="hops", choices=['latency','hops'])

    args = parser.parse_args()

    # Reordered arguments to match the run_simulation function signature
    run_simulation(args.numFlows,
                   args.numSatellites,       
                   args.orbitalPlanes,       
                   args.routingMethod,      
                   args.epochs,              
                   args.lr,
                   args.fileName,
                   args.metricToOptimize)

if __name__ == '__main__':
    run_main()