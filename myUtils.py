# Numerical and scientific computing
import numpy as np
import math
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.integrate import quad

# Geospatial tools
import healpy as hp
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from pyproj import CRS
import pylink

# Linear and nonlinear programming
import pulp
from pyscipopt import Model, quicksum
from scipy.special import xlogy
import cvxpy as cp

# General utilities
import copy
import random
from itertools import product
import pdb
import warnings
import subprocess
import warnings
import cProfile
import pstats
import io
import time
import pdb
import subprocess
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from math import radians, sin, cos, sqrt, atan2

def generateWalkerStarConstellationPoints(
        numSatellites,
        inclination,
        numPlanes,
        phaseParameter,
        orbitRadius):
    """
    This method generates as many points as needed for the walker star constellation.

    It will use the rotate_3d_points function as well.

    numSatellites: number of satellites we are working with
    inclination: it really can be with reference to any axis since we are equally
    distributing the planes, but we will use the connectivity_matrix axis
    numPlanes: number of circles over which we distribute the satellites
    phaseParameter: parameter for calculating the phase difference for
    planes. You get it from pP*360/t, where t = num satellites
    altitude: how far above the earth our satellite is

    Outputs:
    walkerPoints: constellation points for walker delta
    normVecs: normal vectors, used for updating the position

    """

    #alright, now that we have intro done, lets work with the calculation
    numSatellitesPerPlane = numSatellites/numPlanes
    if(int(numSatellitesPerPlane) - numSatellitesPerPlane != 0):
        raise Exception("Should have int # satellites per plane")

    #convert to integer for future use
    numSatellitesPerPlane = int(numSatellitesPerPlane)

    #after we have the number of satellties per plane we are working with,
    #first get the base set of points, to do this, we use the
    #numSatellitesPerPlane number with a linearly spaced angle, and spherical
    #coordinates while the z is kept constant

    #vary the angle from 0 to 2*Pi for full circle
    #get the number of points + 1 without the end so that start != end
    phi = np.linspace(0, 2 * np.pi, numSatellitesPerPlane+1)[:-1]

    #have the distance from core for radius of circle, added with the altitude of the satellite
    distFromCore = orbitRadius

    #calculate the basic set of points using spherical coordinates
    basePoints = [distFromCore*np.cos(phi), distFromCore*np.sin(phi), 0*phi]

    #create storage for all points
    # storage will be: numPlanes, numPointsPerPlane ^ 3
    walkerPoints = np.ones([numPlanes, numSatellitesPerPlane, 3])

    #create storage for normal vectors, one for each plane
    normVecs = [0]*numPlanes

    #in this loop, for each plane we are going to do 3 rotations.
    # 1. rotate about the z axis for the phasing parameter angle result
    # 2. rotate about the y axis for the inclination angle
    # 3. rotate obout z axis again for "general rotation angle" (difference of planes)
    for planeInd in range(numPlanes):

        #first, get our deep copy of the basePoints set
        basePointsCopy = copy.deepcopy(basePoints)

        #after we have a deep copy, then follow through with rotations
        #please note, these rotations do not directly translate to spherical
        #coordinate system angles

        #first, rotate for phasing parameter
        zRotateForPhasing = planeInd*phaseParameter*360/numSatellites
        basePointsCopy = rotate_3d_points(basePointsCopy, [0,0,zRotateForPhasing])

        #then, rotate for inclination and general plane rotation
        zRotateAngle = planeInd*360/numPlanes
        yRotateAngle = inclination
        xRotateAngle = 0
        basePointsCopy = rotate_3d_points(basePointsCopy, [xRotateAngle,yRotateAngle,zRotateAngle])

        walkerPoints[planeInd] = basePointsCopy.T

        #also, calculate the normal vector for each respective circular path
        #then, you can use the same normal vector for each path
        normVecs[planeInd] = calculate_normal_vector(walkerPoints[planeInd,0], walkerPoints[planeInd,1])

    return walkerPoints, normVecs

def rotate_3d_points(points, angles):
    """
    This function takes a set of points in 3d space
    (row for the point number) and rotates in 3 dimensions
    by 3 respective angles

    points: the points we are rotating (still in xyz format)
    angles: the angles about each axis we are rotating. When we say "about"
    we really just mean rotating around...range for each is 0 to 2Pi

    """
    # Convert angles to radians
    angles_rad = np.radians(angles)

    # Define rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
                   [0, np.sin(angles_rad[0]), np.cos(angles_rad[0])]])

    Ry = np.array([[np.cos(angles_rad[1]), 0, np.sin(angles_rad[1])],
                   [0, 1, 0],
                   [-np.sin(angles_rad[1]), 0, np.cos(angles_rad[1])]])

    Rz = np.array([[np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
                   [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
                   [0, 0, 1]])

    # Apply rotations

    rotated_points = np.dot(Rz, np.dot(Ry, np.dot(Rx, points)))

    return rotated_points

def calculate_normal_vector(point1, point2):

    # Calculate normal vector
    normal_vector = np.cross(np.array(point1), np.array(point2))

    # Normalize the normal vector
    normal_vector /= np.linalg.norm(normal_vector)

    return normal_vector

# Define the similarity metric: cosine of angle between great circle tangents at A toward B and C
def tangent_vector(base, target):
    proj = target - np.dot(target, base) * base  # projection to tangent plane
    return proj / np.linalg.norm(proj)

def similarity_metric(A, B, C):
    T_AB = tangent_vector(A, B)
    T_AC = tangent_vector(A, C)
    return np.dot(T_AB, T_AC)

def batch_tangent_vectors(base, target):
    """
    Computes tangent vectors from `base` to `target` projected onto the tangent plane at `base`.
    
    base:   (..., 3)
    target: (..., 3)
    
    Returns:
        tangent vectors of shape (..., 3)
    """
    dot = np.sum(base * target, axis=-1, keepdims=True)          # (..., 1)
    proj = target - dot * base                                    # (..., 3)
    norm = np.linalg.norm(proj, axis=-1, keepdims=True) + 1e-9    # avoid division by 0
    return proj / norm

def batch_similarity_metric(A, B, C):
    """
    Computes similarity metric α(A,B,C) over all combinations:
    If A.shape=(n,3), B.shape=(m,3), C.shape=(k,3), returns shape (n, m, k)

    α(A,B,C) = cosine between the tangent vector at A toward B and toward C
    """
    A = A[:, None, None, :]     # shape (n,1,1,3)
    B = B[None, :, None, :]     # shape (1,m,1,3)
    C = C[None, None, :, :]     # shape (1,1,k,3)

    T_AB = batch_tangent_vectors(A, B)    # shape (n, m, k, 3)
    T_AC = batch_tangent_vectors(A, C)    # shape (n, m, k, 3)

    sim = np.sum(T_AB * T_AC, axis=-1)    # shape (n, m, k)
    
    #set component equal to 0 if negative in line ness ... 
    sim[sim < 0] = 0

    # now compute distances A→B and A→C
    # these will broadcast to (n,m,k)
    d_AB = np.linalg.norm(B - A, axis=-1)   # (n,m,1)
    d_AC = np.linalg.norm(C - A, axis=-1)   # (n,1,k)

    # mask = True where we should *keep* α, i.e. only if dist(A→B) >= dist(A→C)
    keep = (d_AB >= d_AC)

    # zero out α where destination is closer to A than the intermediate C
    sim[~keep] = 0

    return torch.tensor(sim, dtype=torch.float32)

def _get_tangent_vector(P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
    P1_norm = torch.nn.functional.normalize(P1, p=2, dim=-1)
    P2_norm = torch.nn.functional.normalize(P2, p=2, dim=-1)
    dot_product = torch.sum(P1_norm * P2_norm, dim=-1, keepdim=True)
    raw_tangent_vector = P2_norm - dot_product * P1_norm
    norm_val = torch.linalg.norm(raw_tangent_vector, dim=-1, keepdim=True)
    norm_val[norm_val == 0] = 1.0
    tangent_vector = raw_tangent_vector / norm_val
    return tangent_vector

def batch_similarity_metric_new(
    origin_points: torch.Tensor, # Shape (N, 3)
    to_points_1: torch.Tensor,   # Shape (M, 3)
    to_points_2: torch.Tensor    # Shape (K, 3)
) -> torch.Tensor:
    """
    Computes the great-arc similarity (cosine of the angle between initial tangent vectors)
    for all combinations of arcs defined by (origin_points[i] -> to_points_1[j])
    and (origin_points[i] -> to_points_2[k]).

    Output shape will be (N, M, K).

    Assumes all input points are 3D Cartesian coordinates and will be normalized to unit vectors.
    """
    N = origin_points.shape[0]
    M = to_points_1.shape[0] # Correctly gets M from the actual input tensor B
    K = to_points_2.shape[0] # Correctly gets K from the actual input tensor C

    # Normalize all points to unit vectors
    origin_points_norm = torch.nn.functional.normalize(origin_points, p=2, dim=-1)
    to_points_1_norm = torch.nn.functional.normalize(to_points_1, p=2, dim=-1)
    to_points_2_norm = torch.nn.functional.normalize(to_points_2, p=2, dim=-1)

    # --- Calculate tangent vectors for (Origin_i -> ToPoint1_j) arcs ---
    # We need to broadcast origin_points_norm (N, 3) and to_points_1_norm (M, 3)
    # to result in (N, M, 3) combinations.
    # origin_points_norm becomes (N, 1, 3) then expands to (N, M, 3)
    origin_points_expanded_for_T1 = origin_points_norm.unsqueeze(1).expand(-1, M, -1)
    # to_points_1_norm becomes (1, M, 3) then expands to (N, M, 3)
    to_points_1_expanded = to_points_1_norm.unsqueeze(0).expand(N, -1, -1)

    # Tangent vectors from each A_i to each B_j. Shape: (N, M, 3)
    tangent_vectors_1 = _get_tangent_vector(origin_points_expanded_for_T1, to_points_1_expanded)


    # --- Calculate tangent vectors for (Origin_i -> ToPoint2_k) arcs ---
    # We need to broadcast origin_points_norm (N, 3) and to_points_2_norm (K, 3)
    # to result in (N, K, 3) combinations.
    # origin_points_norm becomes (N, 1, 3) then expands to (N, K, 3)
    origin_points_expanded_for_T2 = origin_points_norm.unsqueeze(1).expand(-1, K, -1)
    # to_points_2_norm becomes (1, K, 3) then expands to (N, K, 3)
    to_points_2_expanded = to_points_2_norm.unsqueeze(0).expand(N, -1, -1)

    # Tangent vectors from each A_i to each C_k. Shape: (N, K, 3)
    tangent_vectors_2 = _get_tangent_vector(origin_points_expanded_for_T2, to_points_2_expanded)

    # --- Compute the final similarity (dot product of tangent vectors) ---
    # We need to compare tangent_vectors_1[i, j, :] with tangent_vectors_2[i, k, :]
    # To do this using broadcasting, we expand them to (N, M, 1, 3) and (N, 1, K, 3)
    tangent_vectors_1_expanded_for_sim = tangent_vectors_1.unsqueeze(2) # Shape: (N, M, 1, 3)
    tangent_vectors_2_expanded_for_sim = tangent_vectors_2.unsqueeze(1) # Shape: (N, 1, K, 3)

    # Element-wise multiplication will broadcast to (N, M, K, 3)
    # Summing over the last dimension (3) gives the dot product.
    similarity_matrix = torch.sum(tangent_vectors_1_expanded_for_sim * tangent_vectors_2_expanded_for_sim, dim=-1)

    # Clamp results to [0, 1] for valid cosine range
    similarity_matrix = torch.clamp(similarity_matrix, 0, 1.0)

    # --- Apply condition: if distance from origin to to_point_1 < distance from origin to to_point_2, set sim_metric to 0 ---

    # Calculate Euclidean distance from origin_points to to_points_1
    # origin_points_expanded_for_T1 has shape (N, M, 3)
    # to_points_1_expanded has shape (N, M, 3)
    # The distance will be (N, M)
    distances_0_to_1 = torch.norm(to_points_1_expanded - origin_points_expanded_for_T1, dim=-1)

    # Calculate Euclidean distance from origin_points to to_points_2
    # origin_points_expanded_for_T2 has shape (N, K, 3)
    # to_points_2_expanded has shape (N, K, 3)
    # The distance will be (N, K)
    distances_0_to_2 = torch.norm(to_points_2_expanded - origin_points_expanded_for_T2, dim=-1)

    # Expand dimensions for broadcasting to (N, M, K) for comparison
    # distances_0_to_1 becomes (N, M, 1)
    distances_0_to_1_expanded = distances_0_to_1.unsqueeze(2)
    # distances_0_to_2 becomes (N, 1, K)
    distances_0_to_2_expanded = distances_0_to_2.unsqueeze(1)

    # Create a boolean mask where the condition is met
    # condition_mask will have shape (N, M, K)
    # True where distance from origin to destination (to_points_1) is shorter than
    # distance from origin to intermediate (to_points_2)
    condition_mask = distances_0_to_1_expanded < distances_0_to_2_expanded

    # Set the similarity metric to 0.0 where the condition is True
    similarity_matrix[condition_mask] = 0.0

    return similarity_matrix
    
def symmetric_sinkhorn(logits, num_iters=10, epsilon=1e-9, normLim = 1, temperature = 1):
    """
    Performs symmetric Sinkhorn normalization on a square matrix.

    Args:
        logits (torch.Tensor): A square matrix of shape (N, N) representing unnormalized scores.
        num_iters (int): Number of Sinkhorn iterations to perform.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: A symmetric doubly-stochastic matrix (approximate).
    """
    X = logits.clone()

    for _ in range(num_iters):

        # Row normalization
        row_sums = X.sum(dim=1, keepdim=True) + epsilon
        X = normLim * X / row_sums

        # Column normalization
        col_sums = X.sum(dim=0, keepdim=True) + epsilon
        X = normLim * X / col_sums

    #Clamp to ensure each value less than 1 
    X = torch.clamp(X, max = 1)

    X = sharpen_top_n(X, normLim, temperature * 2)

    # Enforce symmetry
    X = 0.5 * (X + X.T)

    return X

import torch

def sharpen_top_n(X, normLim=3, temperature=5.0):
    sorted_X, _ = torch.sort(X, dim=1)
    ref_val = (sorted_X[:, -normLim] + sorted_X[:, -normLim - 1]) / 2  # avg of n and n-1
    ref_val = ref_val[:, np.newaxis]

    # Avoid division by zero
    eps = 1e-8
    x_pow = (X / (ref_val + eps)) ** temperature
    complement_pow = ((1 - X) / (1 - ref_val + eps)) ** temperature

    sharpened = x_pow / (x_pow + complement_pow + eps)
    return torch.clip(sharpened, 0, 1)


def plus_sinkhorn(logits, num_iters=10, epsilon=1e-9, normLim=1, temperature=1):
    """
    Performs 'plus' Sinkhorn normalization on a square matrix.

    For each index i, the i-th row and i-th column are normalized together
    to have a combined sum of normLim * 2 (since it's row + col).

    Args:
        logits (torch.Tensor): A square matrix (N x N).
        num_iters (int): Number of iterations.
        epsilon (float): To avoid divide-by-zero.
        normLim (float): Target sum per plus (row + column).
        temperature (float): Push entries toward 0 or 1 with soft binarization.

    Returns:
        torch.Tensor: Soft doubly-normalized matrix with 'plus' symmetry.
    """
    X = logits.clone()

    N = X.shape[0]

    for _ in range(num_iters):
        for i in range(N):
            row = X[i, :]         # i-th row
            col = X[:, i]         # i-th column

            total = row.sum() + col.sum()
            scale = (2 * normLim) / (total + epsilon)

            X[i, :] *= scale
            X[:, i] *= scale

        # Clamp and push toward 0/1
        # X = torch.clamp(X, max=1.0)
        # x_pow = X ** temperature
        # complement_pow = (1 - X) ** temperature
        # X = x_pow / (x_pow + complement_pow + epsilon)
        X = sharpen_top_n(X, normLim, temperature*2)

        # Enforce symmetry
        X = 0.5 * (X + X.T)

        X.fill_diagonal_(0)

    return X


def check_los_old(locSet1, locSet2, earth_radius=6371000):
    """
    Determines if there is a clear Line-of-Sight (LOS) between each combination

    Returns:
    - los_matrix: Boolean matrix where True means LOS exists, False means blocked.
    """
    # Compute differences
    diff = locSet2[np.newaxis, :, :] - locSet1[:, np.newaxis, :]
    
    # Compute quadratic equation coefficients
    a = np.sum(diff**2, axis=2)  # Magnitude squared of (B - S)
    b = 2 * np.sum(locSet1[:, np.newaxis, :] * diff, axis=2)  # Dot product term
    c = np.sum(locSet1**2, axis=1)[:, np.newaxis] - earth_radius**2  # Satellite altitude squared - R^2

    # Compute discriminant
    discriminant = b**2 - 4 * a * c

    # Compute valid t values
    t1 = (-b - np.sqrt(discriminant + 1e-10)) / (2 * a)  # Avoid numerical issues
    t2 = (-b + np.sqrt(discriminant + 1e-10)) / (2 * a)

    # LOS condition: No valid t in (0,1)
    los_matrix = (discriminant < 0) | ((t1 < 0) & (t2 < 0)) | ((t1 > 1) & (t2 > 1))

    return los_matrix

def check_los(locSet1, locSet2, earth_radius=6371000):
    """
    Determines Line-of-Sight (LOS) between points in locSet1 and locSet2.
    Handles cases where locSet1 and locSet2 may contain identical points.

    Args:
        locSet1: Array of positions (N x 3) in ECEF coordinates (origin at Earth center)
        locSet2: Array of positions (M x 3) in ECEF coordinates
        earth_radius: Radius of Earth in meters

    Returns:
        los_matrix: Boolean matrix (N x M) where True means LOS exists
    """
    # Vector from locSet1 to locSet2
    diff = locSet2[np.newaxis, :, :] - locSet1[:, np.newaxis, :]  # [N, M, 3]
    
    # Check for zero vectors (identical points)
    identical_mask = np.all(diff == 0, axis=2)  # [N, M]
    
    # Quadratic equation coefficients
    a = np.sum(diff**2, axis=2)  # [N, M]
    b = 2 * np.sum(locSet1[:, np.newaxis, :] * diff, axis=2)  # [N, M]
    c = np.sum(locSet1**2, axis=1)[:, np.newaxis] - earth_radius**2  # [N, 1]
    
    # Discriminant
    discriminant = b**2 - 4 * a * c
    
    # Initialize LOS matrix
    los_matrix = np.zeros_like(a, dtype=bool)
    
    # Case 1: Different points (a != 0)
    mask = ~identical_mask
    if np.any(mask):
        # Calculate roots only where needed
        sqrt_disc = np.sqrt(np.maximum(discriminant[mask], 0))
        denominator = 2 * a[mask]
        
        # Avoid division by zero (though mask should prevent this)
        denominator[denominator == 0] = np.inf
        
        t1 = (-b[mask] - sqrt_disc) / denominator
        t2 = (-b[mask] + sqrt_disc) / denominator
        
        # LOS exists if:
        # 1. No real roots (discriminant < 0), OR
        # 2. Both roots outside [0,1] segment
        los_matrix[mask] = (discriminant[mask] < 0) | ((t1 < 0) & (t2 < 0)) | ((t1 > 1) & (t2 > 1))
    
    # Case 2: Identical points (a = 0)
    if np.any(identical_mask):
        # For identical points:
        # - If point is above Earth: LOS = True (can see itself)
        # - If point is below Earth: LOS = False
        #r_sq = np.sum(locSet1**2, axis=1)[:, np.newaxis]  # [N, 1]
        los_matrix[identical_mask] = True
    
    return los_matrix

def build_full_logits(logits_feasible, feasible_indices, shape):
    full_logits = torch.full(shape, -1e9, device=logits_feasible.device)
    full_logits[feasible_indices[0], feasible_indices[1]] = logits_feasible
    return full_logits

def diagonal_symmetry_score(A, epsilon=1e-8):
    diff = A - A.T
    score = 1 - torch.norm(diff, p='fro') / (torch.norm(A, p='fro') + epsilon)
    return score

def normalization_score(A, ref=4.0, epsilon=1e-8):
    row_sums = A.sum(dim=1)
    col_sums = A.sum(dim=0)

    # Compute the amount each row/col sum exceeds the reference
    row_excess = torch.clamp(row_sums - ref, min=0.0)
    col_excess = torch.clamp(col_sums - ref, min=0.0)

    total_excess = row_excess.sum() + col_excess.sum()
    max_possible_excess = A.shape[0] * ref + A.shape[1] * ref

    # Normalize score: 1 = perfect, lower = more violation
    score = 1 - total_excess / (max_possible_excess + epsilon)
    return score


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
    N = num_planes * sats_per_plane
    assert positions.shape[0] == N, "positions must have num_planes * sats_per_plane rows"
    
    # initialize no‐links
    C = np.zeros((N, N), dtype=bool)
    
    for p in range(num_planes):
        for s in range(sats_per_plane):
            idx = p * sats_per_plane + s
            
            # 1) In‐plane neighbors (left/right on the ring)
            s_prev = (s - 1) % sats_per_plane
            s_next = (s + 1) % sats_per_plane
            idx_prev = p * sats_per_plane + s_prev
            idx_next = p * sats_per_plane + s_next
            
            C[idx, idx_prev] = True
            C[idx_prev, idx] = True
            C[idx, idx_next] = True
            C[idx_next, idx] = True
            
            # 2) Cross‐plane neighbors (same slot in adjacent planes)
            p_prev = (p - 1) % num_planes
            p_next = (p + 1) % num_planes
            idx_prev_plane = p_prev * sats_per_plane + s
            idx_next_plane = p_next * sats_per_plane + s
            
            C[idx, idx_prev_plane] = True
            C[idx_prev_plane, idx] = True
            C[idx, idx_next_plane] = True
            C[idx_next_plane, idx] = True

    return C


def FW(connectivity_matrix):
    num_nodes = connectivity_matrix.shape[0]
    distance_matrix = np.copy(connectivity_matrix)

    # Floyd-Warshall algorithm to compute all-pairs shortest paths (latencies)
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, k] != np.inf and distance_matrix[k, j] != np.inf:
                    distance_matrix[i, j] = np.minimum(distance_matrix[i, j], distance_matrix[i, k] + distance_matrix[k, j])

    return distance_matrix

def size_weighted_latency_matrix(connectivity_matrix, traffic_matrix):
    """
    Computes the size-weighted latency matrix based on connectivity and traffic.

    Args:
        connectivity_matrix (numpy.ndarray): A square matrix representing the
            connectivity or one-hop latency between nodes. A value of infinity
            (np.inf) indicates no direct connection.
        traffic_matrix (numpy.ndarray): A square matrix representing the traffic
            demand between each source-destination pair.

    Returns:
        numpy.ndarray: A square matrix representing the size-weighted latency
        between each source-destination pair.
    """
    num_nodes = connectivity_matrix.shape[0]
    distance_matrix = np.copy(connectivity_matrix)

    # Floyd-Warshall algorithm to compute all-pairs shortest paths (latencies)
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, k] != np.inf and distance_matrix[k, j] != np.inf:
                    distance_matrix[i, j] = np.minimum(distance_matrix[i, j], distance_matrix[i, k] + distance_matrix[k, j])

    # Element-wise multiplication of latency and traffic matrices
    size_weighted_latency = np.multiply(distance_matrix, traffic_matrix)

    return size_weighted_latency


def plot_connectivity(positions: np.ndarray, C: np.ndarray, figsize=(8,8)):
    """
    Plot satellites as points and ISL links as lines in 3D.

    Parameters
    ----------
    positions : np.ndarray, shape (N,3)
        Cartesian coordinates of each satellite.
    C : np.ndarray, shape (N,N), dtype=bool or 0/1
        Connectivity matrix: C[i,j]=True/1 if there is a link between i and j.
    """
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    # Plot satellites
    xs, ys, zs = positions[:,0], positions[:,1], positions[:,2]
    ax.scatter(xs, ys, zs, s=20, color='blue', label='Satellites')

    # Plot links
    N = positions.shape[0]
    # To avoid plotting each link twice, only draw for j > i
    for i in range(N):
        for j in range(i+1, N):
            if C[i, j]:
                xline = [positions[i,0], positions[j,0]]
                yline = [positions[i,1], positions[j,1]]
                zline = [positions[i,2], positions[j,2]]
                ax.plot(xline, yline, zline, color='gray', linewidth=0.5)

    # Styling
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Walker-Δ Constellation +‑Grid Connectivity')
    ax.legend()
    plt.tight_layout()
    plt.show()

# def harden_routing(R):
#     """
#     Convert a soft routing matrix of shape [i, d, i] into a hard routing matrix
#     where only the max along the last dimension is set to 1, others to 0.
    
#     Args:
#         R (torch.Tensor): A tensor of shape [curr_node, dest_node, intermediary_node]
    
#     Returns:
#         torch.Tensor: A hard routing matrix with the same shape as R
#     """
#     # Get index of max along the last dimension
#     max_indices = torch.argmax(R, dim=-1, keepdim=True)  # shape: [i, d, 1]
    
#     # Create one-hot matrix
#     hard_R = torch.zeros_like(R)
#     hard_R.scatter_(-1, max_indices, 1.0)
    
#     print(torch.sum(hard_R - R))
#     pdb.set_trace()

#     return hard_R

def harden_routing(R):
    """
    Convert a soft routing matrix [i, d, i] into a hard routing matrix by
    selecting the max only where the routing row is active (row sum == 1).

    Args:
        R (torch.Tensor): A tensor of shape [curr_node, dest_node, intermediary_node]

    Returns:
        torch.Tensor: A partially hardened routing matrix
    """
    # Sum across last dim to find active rows (routing rows that sum to ~1)
    row_sums = R.sum(dim=-1, keepdim=True)  # [i, d, 1]
    active_mask = (row_sums > 1e-5).float()  # binary mask of active rows

    # Get max index along last dim
    max_indices = torch.argmax(R, dim=-1, keepdim=True)  # [i, d, 1]

    # Create one-hot hard matrix
    hard_R = torch.zeros_like(R)
    hard_R.scatter_(-1, max_indices, 1.0)

    # Only apply hardening where the row was active
    # Otherwise, preserve the original soft R (which should be 0s anyway)
    out_R = active_mask * hard_R + (1 - active_mask) * R

    # print(torch.sum(out_R - R))
    # pdb.set_trace()

    return out_R

def great_circle_distance_matrix_cartesian(points_xyz, radius):
    """
    Compute the great circle distance matrix from a set of 3D Cartesian coordinates on a sphere.

    Args:
        points_xyz (torch.Tensor): Tensor of shape [N, 3], where each row is (x, y, z).
        radius (float): Radius of the sphere (e.g., Earth's radius in km).

    Returns:
        torch.Tensor: A [N, N] matrix of great circle distances between all pairs of points.
    """
    points_xyz = torch.from_numpy(points_xyz)
    # Normalize to unit vectors in case input isn't normalized
    unit_vecs = points_xyz / points_xyz.norm(dim=1, keepdim=True)

    # Compute cosine of angular distances via dot products
    cos_theta = torch.matmul(unit_vecs, unit_vecs.T).clamp(-1.0, 1.0)  # [N, N]

    # Arc cosine to get central angles (in radians)
    theta = torch.acos(cos_theta)

    # Great circle distances = θ * radius
    return theta * radius

def proportional_routing_table(distance_matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes a proportional routing table based on shortest path distances.

    For a given current_node (i) and final_destination (d), the proportion of
    traffic sent to a next_node (j) is inversely proportional to the sum of:
    1. The shortest path distance from i to j (dist(i, j)).
    2. The shortest path distance from j to d (dist(j, d)).

    R[i, d, j] = (1 / (dist(i, j) + dist(j, d))) / Sum_over_k (1 / (dist(i, k) + dist(k, d)))

    Args:
        distance_matrix (torch.Tensor): A [N, N] tensor where distance_matrix[u, v]
                                        represents the shortest path distance from
                                        node u to node v (e.g., output from Floyd-Warshall).
                                        Assumes unreachable paths are represented by
                                        very large numbers (e.g., float('inf')).

    Returns:
        torch.Tensor: A [N, N, N] routing table R, where R[i, d, j] is the
                      proportion of traffic from node i destined for node d
                      that should be routed to node j.
                      R[i, d, j] will be 0 if:
                      - i == d (current node is already the destination).
                      - j == i (not moving to a different node, typically not desired as a "hop").
                      - The path (i -> j -> d) is not valid (e.g., involves unreachable segments).
    """
    num_nodes = distance_matrix.shape[0]
    R_table = torch.zeros(num_nodes, num_nodes, num_nodes, dtype=torch.float)

    # Define a value to represent infinity for unreachable paths.
    # This should match how unreachable paths are represented in your distance_matrix.
    INF = float('inf') 

    for i in range(num_nodes):  # Iterate through current nodes (source)
        for d in range(num_nodes):  # Iterate through destination nodes
            if i == d:
                # If the current node is the destination, no routing is needed.
                # The corresponding slice of R_table (R[i, d, :]) remains all zeros,
                # as initialized, which is the correct behavior.
                continue

            # This tensor will store the unnormalized "preference" for each next_node (j)
            # for the current (i, d) pair.
            preferences_for_j = torch.zeros(num_nodes, dtype=torch.float)
            
            for j in range(num_nodes):  # Iterate through possible next nodes (intermediate hop)
                if i == j:
                    # We typically don't consider routing to the same node as a "hop"
                    # unless it's the destination itself (handled by the i == d check above).
                    # So, the preference for routing to self is 0.
                    continue

                dist_i_j = distance_matrix[i, j] # Shortest path distance from current to next hop
                dist_j_d = distance_matrix[j, d] # Shortest path distance from next hop to final destination

                # Check if either segment of the path (i -> j or j -> d) is unreachable
                # or if the combined cost is zero (which shouldn't happen for i != d, j != i unless distances are ill-defined)
                if dist_i_j == INF or dist_j_d == INF:
                    preferences_for_j[j] = 0.0 # Path is invalid, so no preference
                else:
                    # Calculate the cost: "that hop" (i to j) + "next node's distance from end destination" (j to d)
                    cost = dist_i_j + dist_j_d
                    
                    # Ensure cost is not zero for division, though for valid paths (i != d, j != i),
                    # cost should typically be positive. Add epsilon for numerical stability.
                    if cost <= 1e-9: # If cost is effectively zero, this path is not useful
                        preferences_for_j[j] = 0.0
                    else:
                        # Preference is inversely proportional to the cost
                        preferences_for_j[j] = 1.0 / cost

            # Normalize the preferences for this (i, d) pair so they sum to 1
            total_preference = preferences_for_j.sum()
            if total_preference > 0:
                # If there are valid next hops, normalize the preferences to get proportions
                R_table[i, d, :] = preferences_for_j / total_preference
            # If total_preference is 0, it means no valid next hops were found for this (i,d) pair.
            # In this case, R_table[i, d, :] will remain all zeros, indicating no outgoing traffic,
            # which is the correct behavior for traffic that cannot proceed.

    return R_table

def differentiable_floyd_warshall(weights_matrix: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Computes an approximate shortest path distance matrix using a differentiable
    Floyd-Warshall algorithm with softmin (LogSumExp) approximation.

    Args:
        weights_matrix (torch.Tensor): A [N, N] tensor representing direct edge weights.
                                        Assumes large values (e.g., float('inf')) for no direct edge.
                                        This matrix is the initial distance matrix (D_0).
        beta (float): Temperature parameter for the softmin approximation.
                      Smaller beta -> closer to true min, but potentially less smooth gradients
                      and numerical instability. Larger beta -> smoother gradients, but less
                      accurate approximation of the true shortest path.

    Returns:
        torch.Tensor: A [N, N] tensor of approximate shortest path distances.
    """
    num_nodes = weights_matrix.shape[0]

    distances = weights_matrix.clone() 
    
    distances.fill_diagonal_(0.0) 

    for k in range(num_nodes):

        term1 = -distances # This is an [N, N] tensor

        term2 = -(distances[:, k].unsqueeze(1) + distances[k, :].unsqueeze(0)) # This is an [N, N] tensor

        combined_terms = torch.stack([term1 / beta, term2 / beta], dim=0)
        
        min_approximated_neg_dist_div_beta = torch.logsumexp(combined_terms, dim=0)
        
        distances = -min_approximated_neg_dist_div_beta * beta
        


        distances.fill_diagonal_(0.0)

    return distances