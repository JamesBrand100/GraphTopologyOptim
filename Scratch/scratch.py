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