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