


def build_plus_grid_connectivity(positions: np.ndarray,
                                 num_planes: int,
                                 sats_per_plane: int) -> np.ndarray:
    """
    Build the '+-grid' inter‐satellite link (ISL) connectivity matrix
    for a Walker‐delta constellation.

    Parameters
    ----------
    positions : np.ndarray, shape (N,3)
        ECEF or unit‐sphere positions of all N=num_planes*sats_per_plane satellites,
        ordered so that indices [p * sats_per_plane + s] correspond to
        plane p ∈ {0…num_planes−1}, satellite s ∈ {0…sats_per_plane−1}.
    num_planes : int
        Number of orbital planes in the Walker constellation.
    sats_per_plane : int
        Number of satellites per plane.

    Returns
    -------
    C : np.ndarray, shape (N,N), dtype=bool
        Connectivity matrix: C[i,j] = True if satellite i and j are joined
        by an ISL under the '+-grid' topology (2 in‐plane + 2 cross‐plane links).
    """

    #idk if 360 is sat dependent....


    #so, the reset # after a full rotation is:
    #num_planes * phasing parameter, so how much degree difference for full thing
    #divided by (360 / sats_per_plane), so how many satellite differences in orbital
    resetNum = int(num_planes * 5 / ( 360 ) * sats_per_plane) + 2

    N = num_planes * sats_per_plane
    assert positions.shape[0] == N, "positions must have num_planes * sats_per_plane rows"
    
    # initialize no‐links
    C = np.zeros((N, N), dtype=bool)
    
    # for p in range(num_planes):
    #     for s in range(sats_per_plane):
            
    #         idx = p * sats_per_plane + s
            
    #         # 1) In‐plane neighbors (left/right on the ring)
    #         s_prev = (s - 1) % sats_per_plane
    #         s_next = (s + 1) % sats_per_plane
    #         idx_prev = p * sats_per_plane + s_prev
    #         idx_next = p * sats_per_plane + s_next
            
    #         C[idx, idx_prev] = True
    #         C[idx_prev, idx] = True
    #         C[idx, idx_next] = True
    #         C[idx_next, idx] = True
            
    #         # 2) Cross‐plane neighbors (same slot in adjacent planes)
    #         p_prev = (p - 1) % num_planes
    #         p_next = (p + 1) % num_planes
    #         idx_prev_plane = p_prev * sats_per_plane + s
    #         idx_next_plane = p_next * sats_per_plane + s
            
    #         C[idx, idx_prev_plane] = True
    #         C[idx_prev_plane, idx] = True
    #         C[idx, idx_next_plane] = True
    #         C[idx_next_plane, idx] = True

    #store results 
    planeInd1 = 0
    satInd1 = 0

    planeInd2 = 1
    satInd2 = 0

    #iterate through
    for ind in range(num_planes*sats_per_plane):
        
        print("Hello")
        print(satInd2)
        print(planeInd2)

        #store connectivity 
        C[int(planeInd1*sats_per_plane + satInd1), int(planeInd2*sats_per_plane + satInd2)] = True
        C[int(planeInd2*sats_per_plane + satInd2), int(planeInd1*sats_per_plane + satInd1)] = True

        C[int(planeInd1*sats_per_plane + satInd1), int(planeInd1*sats_per_plane + satInd1 + 1)] = True
        C[ int(planeInd1*sats_per_plane + satInd1 + 1), int(planeInd1*sats_per_plane + satInd1)] = True


        planeInd1+=1 

        #if we reach end of plane index 
        if(planeInd1 >= num_planes): 
            #change sat index appropriately 
            planeInd1 = planeInd1 - num_planes
            #(see comments above topology section to explain this number "5")
            satInd1+=resetNum

        planeInd2+=1

        #if we reach end of plane index 
        if(planeInd2 >= num_planes): 
            #change sat index appropriately (proper reset in other direction)
            planeInd2 = planeInd2 - num_planes
            satInd2+=resetNum    

        satInd1 = satInd1%sats_per_plane
        satInd2 = satInd2%sats_per_plane

    row_sums = np.sum(C, axis=1)
    rows_with_sum_one_mask = (row_sums == 8)
    entries_from_rows_with_sum_one = C[rows_with_sum_one_mask]
    row_indices_with_sum_one = np.where(rows_with_sum_one_mask)[0]
    for row_idx in row_indices_with_sum_one:
        print(f"Row {row_idx}: {C[row_idx]}")

    pdb.set_trace()

    return C
# def great_circle_alignment(A, B, C, eps=1e-9):
#     """
#     Compute great circle alignment α(A, B, C), where:
#     A: [N, 3] - origin points (e.g., ground stations)
#     B: [N, 3] - destination points (e.g., satellite destination)
#     C: [N, 3] - relay points (e.g., neighbor satellites)
    
#     Returns:
#     alpha: [N] - cosine similarity between great-circle tangent from A→B and A→C
#     """

#     # Project B onto the tangent plane at A
#     dot_AB = (A * B).sum(dim=1, keepdim=True)                # [N,1]
#     proj_B = B - dot_AB * A                                  # [N,3]
#     v1 = proj_B / (proj_B.norm(dim=1, keepdim=True) + eps)   # [N,3]

#     # Project C onto the tangent plane at A
#     dot_AC = (A * C).sum(dim=1, keepdim=True)                # [N,1]
#     proj_C = C - dot_AC * A                                  # [N,3]
#     v2 = proj_C / (proj_C.norm(dim=1, keepdim=True) + eps)   # [N,3]

#     # Cosine similarity (dot product of unit vectors)
#     alpha = (v1 * v2).sum(dim=1)                             # [N]

#     return alpha.clamp(min=0)  # optional clamp if negative alignment isn't useful



    #if we are at the last epoch 
    if(epoch == epochs - 1):
        #pdb.set_trace()
        #then, get the actual loss
        HardenedR = myUtils.harden_routing(R)
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
            traffic_sent = T_current[:, :, None] * HardenedR.permute(1, 0, 2)  # [d, i, j]

            # Compute latency for all traffic
            #maybe c here isnt...good....
            scaledDist = dmat / (c + 1e-6)
            #compute one hop 
            latency = torch.einsum('dij,ij->', traffic_sent, scaledDist)  # Scalar
            #then, affected by allocations 
            total_latency += latency

            # Propagate: sum over i (source), traffic now at j for each destination d
            # T_next[d, j] = sum_i T_current[d, i] * R[i, d, j]
            T_next = torch.einsum('di,dij->dj', T_current, HardenedR.permute(1, 0, 2))  # [d, j]

            T_current = T_next

        print("Hardened Loss: " + str(total_latency.item()))

    alpha_sharp = similarityMetric ** gamma  # [i,d,i]
    numer = c.unsqueeze(1) * alpha_sharp     # [i,d,i]
    denom = numer.sum(dim=2, keepdim=True)   # [i,d,1]

    #then, after setting the Rsub, take entries that have all 0s properly....
    #so if  a node is not the end dest, and has no alpha > 0, set all alpha to be uniform (uniformly spread out then among valid links)
    # Identify where denom is effectively zero (i.e., all similarity scores are 0)
    fallback_mask = (denom <= 1e-15)  # [i,d,1]

    # Compute normalized c across valid ISLs for fallback
    # c: [i, j] → normalize over j (outgoing links for i)
    c_norm = c / (c.sum(dim=1, keepdim=True) + 1e-15)  # [i, j]

    # Broadcast to [i, d, j] to match shape
    fallback_numer = c_norm.unsqueeze(1).expand_as(numer)  # [i, d, j]

    # Replace numer and denom where fallback is needed
    numer = torch.where(fallback_mask.expand_as(numer), fallback_numer, numer)
    denom = numer.sum(dim=2, keepdim=True) + 1e-15  # Recompute denom after numer substitution

    # Final routing probabilities
    Rsub = numer / denom  # [i,d,i]

    R = torch.zeros([numSatellites, numSatellites, numSatellites], dtype=torch.float)
    R[:, dst_indices, :] = Rsub