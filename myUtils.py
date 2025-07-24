# Numerical and scientific computing
import numpy as np
import math
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.cm as cm # Import colormap module
from scipy.linalg import eigh # eigh is for symmetric/Hermitian matrices, which Laplacian is
from collections import Counter
# Import the specific flow algorithm module
import networkx.algorithms.flow as flow

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
import networkx as nx
import matplotlib.colors as mcolors # Import for PowerNorm

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

def hopEfficiencyMetric(
    origin_points: torch.Tensor, # Shape (N, 3)
    to_points_1: torch.Tensor,   # Shape (M, 3)
    to_points_2: torch.Tensor    # Shape (K, 3)
) -> torch.Tensor:
    return


def batch_similarity_metric_triangle_great_circle(
    origin_points: torch.Tensor, # Shape (N, 3) - e_g
    to_points_1: torch.Tensor,   # Shape (M, 3) - e_i
    to_points_2: torch.Tensor,   # Shape (K, 3) - e_d
    R_sphere: float = 6946000,             # Radius of the sphere
    epsilon: float = 100 #1e-8        # Small constant for numerical stability in the denominator
) -> torch.Tensor:
    
    N = origin_points.shape[0]
    M = to_points_1.shape[0]
    K = to_points_2.shape[0]
    
    #use concatenated vectors for full computation
    gcd_g_d = great_circle_distance_matrix_cartesian(
        np.concatenate([origin_points,to_points_2]),
        R_sphere
    )
    #then take subset, to get vector origin to dest
    gcd_g_d = gcd_g_d[0:len(origin_points), len(origin_points):]

    #use concatenated vectors for full computation
    gcd_i_d = great_circle_distance_matrix_cartesian(
        np.concatenate([to_points_1,to_points_2]),
        R_sphere
    )
    #then take subset, to get vector origin to dest
    gcd_i_d = gcd_i_d[0:len(to_points_1), len(to_points_1):]

    #use concatenated vectors for full computation
    gcd_g_i = great_circle_distance_matrix_cartesian(
        np.concatenate([origin_points,to_points_1]),
        R_sphere
    )
    #then take subset, to get vector origin to dest
    gcd_g_i = gcd_g_i[0:len(origin_points), len(origin_points):]

    #then expand dimensionality 
    gcd_g_d = gcd_g_d.unsqueeze(1) # Shape becomes (N, 1, K)
    gcd_i_d = gcd_i_d.unsqueeze(0) # Shape becomes (1, M, K)
    gcd_g_i = gcd_g_i.unsqueeze(2) # Shape becomes (N, M, 1)

    #then, compute hop efficiency
    hop_efficiency_metric = gcd_g_d/ (gcd_g_i + gcd_i_d + epsilon)

    #then, set entries for going farther away to be equal to 0
    hop_efficiency_metric[gcd_i_d > gcd_g_d] = 0
    
    #pdb.set_trace() 

    return hop_efficiency_metric


def batch_similarity_metric_new_v2(
    origin_points: torch.Tensor, # Shape (N, 3) - e_g
    to_points_1: torch.Tensor,   # Shape (M, 3) - e_i
    to_points_2: torch.Tensor,   # Shape (K, 3) - e_d
    R_sphere: float = 6946000,             # Radius of the sphere
    epsilon: float = 100 #1e-8        # Small constant for numerical stability in the denominator
) -> torch.Tensor:

    #pdb.set_trace()

    """
    Computes the "hop efficiency" metric for all combinations of arcs.
    The metric is defined as: alpha_g,i,d = GCD(e_g, e_d) / (GCD(e_i, e_d) + epsilon)

    Args:
        origin_points (torch.Tensor): Tensor of origin points (e_g). Shape (N, 3).
        to_points_1 (torch.Tensor): Tensor of intermediate points (e_i). Shape (M, 3).
        to_points_2 (torch.Tensor): Tensor of destination points (e_d). Shape (K, 3).
        R_sphere (float): The radius of the sphere on which the points lie.
        epsilon (float): Small constant added to the denominator for numerical stability
                         when GCD(e_i, e_d) is zero or very close to zero.

    Returns:
        torch.Tensor: The hop efficiency metric. Shape (N, M, K).
    """
    N = origin_points.shape[0]
    M = to_points_1.shape[0]
    K = to_points_2.shape[0]

    # --- Calculate GCD(e_g, e_d) ---
    # This represents the GCD from each origin point (e_g) to each destination point (e_d).
    # We need to broadcast origin_points (N, 3) and to_points_2 (K, 3)
    # to compute all N*K pairwise distances.
    # Unsqueeze origin_points to (N, 1, 3) and to_points_2 to (1, K, 3)
    # The _great_circle_distance_xyz function will handle the dot product and arccos.
    # Resulting shape: (N, K)
    # gcd_g_d = great_circle_distance_matrix_cartesian(
    #     origin_points.unsqueeze(1),  # (N, 1, 3)
    #     to_points_2.unsqueeze(0),    # (1, K, 3)
    #     R_sphere
    # )

    #use concatenated vectors for full computation
    gcd_g_d = great_circle_distance_matrix_cartesian(
        np.concatenate([origin_points,to_points_2]),
        R_sphere
    )
    #then take subset, to get vector origin to dest
    gcd_g_d = gcd_g_d[0:len(origin_points), len(origin_points):]

    # --- Calculate GCD(e_i, e_d) ---
    # This represents the GCD from each intermediate point (e_i) to each destination point (e_d).
    # We need to broadcast to_points_1 (M, 3) and to_points_2 (K, 3)
    # to compute all M*K pairwise distances.
    # Unsqueeze to_points_1 to (M, 1, 3) and to_points_2 to (1, K, 3)
    # Resulting shape: (M, K)
    # gcd_i_d = great_circle_distance_matrix_cartesian(
    #     to_points_1.unsqueeze(1),  # (M, 1, 3)
    #     to_points_2.unsqueeze(0),  # (1, K, 3)
    #     R_sphere
    # )

    #use concatenated vectors for full computation
    gcd_i_d = great_circle_distance_matrix_cartesian(
        np.concatenate([to_points_1,to_points_2]),
        R_sphere
    )
    #then take subset, to get vector origin to dest
    gcd_i_d = gcd_i_d[0:len(to_points_1), len(to_points_1):]

    # --- Compute the hop efficiency metric: alpha_g,i,d = GCD(e_g, e_d) / (GCD(e_i, e_d) + epsilon) ---
    # To get the final (N, M, K) shape, we need to expand dimensions for broadcasting.
    # numerator: gcd_g_d has shape (N, K). Expand to (N, 1, K).
    numerator = gcd_g_d.unsqueeze(1) # Shape becomes (N, 1, K)

    # denominator: gcd_i_d has shape (M, K). Expand to (1, M, K).
    # Add epsilon for numerical stability.
    denominator = gcd_i_d.unsqueeze(0) + epsilon # Shape becomes (1, M, K)

    # Perform the division. PyTorch's broadcasting rules will handle the expansion
    # from (N, 1, K) / (1, M, K) to (N, M, K).
    hop_efficiency_metric = denominator#/ denominator

    #post processing for numerical stability. ensures we only use valid hops
    #also ensures we dont have too extreme of ratio 
    
    #only take valid hops 
    hop_efficiency_metric[numerator < denominator ] = 0

    #hop_efficiency_metric[hop_efficiency_metric > 100] = 100

    return hop_efficiency_metric

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

def sharpen_top_n_plus(X, normLim=3, temperature=5.0):
    #iterate through plus instance
    for i in range(len(X)):
        #get top 2n entries for each plus instance
        sorted_X_plus, _ = torch.sort(torch.cat(X[i], X[:,i]), dim=1)
        ref_val = (sorted_X_plus[:, -normLim*2] + sorted_X_plus[:, -normLim*2 - 1]) / 2 #avg of components 
        ref_val = ref_val[:, np.newaxis]

        # Avoid division by zero
        eps = 1e-8
        x_pow = (X[i] / (ref_val + eps)) ** temperature
        complement_pow = ((1 - X[i]) / (1 - ref_val + eps)) ** temperature

        #sharpen 
        sharpened = x_pow / (x_pow + complement_pow + eps)
        X[i] = torch.clip(sharpened, 0, 1)

        #then store sharpened values
        X[i] = sharpened[0:len(X[i])]
        X[:,i] = sharpened[len(X[i]):]  

def sharpen_top_n_plus_more(X, normLim=3, temperature=5.0):
    if X.dim() != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("Input X must be a square 2D tensor.")

    sharpened_X = X.clone()
    eps = 1e-8

    for i in range(X.shape[0]):
        current_row = X[i, :]
        current_col = X[:, i]

        combined_elements = torch.cat((current_row, current_col))
        sorted_combined, _ = torch.sort(combined_elements)

        if sorted_combined.numel() < (normLim * 2 + 1):
             ref_val_scalar = sorted_combined[0]
        else:
             ref_val_scalar = (sorted_combined[-normLim*2] + sorted_combined[-normLim*2 - 1]) / 2

        ref_val_tensor = torch.tensor(ref_val_scalar, device=X.device, dtype=X.dtype).unsqueeze(0)

        x_pow_row = (current_row / (ref_val_tensor + eps)) ** temperature
        complement_pow_row = ((1 - current_row) / (1 - ref_val_tensor + eps)) ** temperature
        sharpened_row = x_pow_row / (x_pow_row + complement_pow_row + eps)
        sharpened_X[i, :] = torch.clip(sharpened_row, 0, 1)

        x_pow_col = (current_col / (ref_val_tensor + eps)) ** temperature
        complement_pow_col = ((1 - current_col) / (1 - ref_val_tensor + eps)) ** temperature
        sharpened_col = x_pow_col / (x_pow_col + complement_pow_col + eps)
        sharpened_X[:, i] = torch.clip(sharpened_col, 0, 1)

    return sharpened_X


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

def build_full_logits_route_old(logits_feasible, feasible_indices, shape, dstIndices):
    full_logits = torch.full(shape + (shape[0],), -1e9, device=logits_feasible.device)
    print(len(feasible_indices[0]))
    print(len(feasible_indices[1]))
    print(len(dstIndices))
    full_logits[feasible_indices[0], dstIndices, feasible_indices[1]] = logits_feasible
    return full_logits

def build_full_logits_route(logits_feasible, feasible_indices, shape, dstIndices):
    N = shape[0]
    full_logits = torch.full(shape + (N,), -1e9, device=logits_feasible.device)

    curr_indices_np = feasible_indices[0]
    inter_indices_np = feasible_indices[1]

    curr_indices = torch.from_numpy(curr_indices_np).to(logits_feasible.device)
    inter_indices = torch.from_numpy(inter_indices_np).to(logits_feasible.device)
    dst_indices_tensor = torch.from_numpy(dstIndices).to(logits_feasible.device)
    
    L = len(curr_indices)
    M = len(dst_indices_tensor)

    reshaped_logits_feasible = logits_feasible.reshape(L, M)

    curr_idx_expanded = curr_indices.unsqueeze(1)
    inter_idx_expanded = inter_indices.unsqueeze(1)
    dst_idx_expanded = dst_indices_tensor.unsqueeze(0)

    full_logits[curr_idx_expanded, dst_idx_expanded, inter_idx_expanded] = reshaped_logits_feasible
    
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

# def size_weighted_latency_matrix_networkx(connectivity_matrix: np.ndarray, traffic_matrix: np.ndarray):
#     """
#     Computes the size-weighted latency matrix based on connectivity and traffic,
#     leveraging NetworkX's Dijkstra's algorithm. This is more efficient than
#     Floyd-Warshall when the traffic matrix is sparse.

#     Args:
#         connectivity_matrix (numpy.ndarray): A square matrix representing the
#             connectivity or one-hop latency between nodes. A value of infinity
#             (np.inf) indicates no direct connection. Edge weights should be
#             non-negative for Dijkstra's.
#         traffic_matrix (numpy.ndarray): A square matrix representing the traffic
#             demand between each source-destination pair.

#     Returns:
#         numpy.ndarray: A square matrix representing the size-weighted latency
#         between each source-destination pair.
#     """
#     num_nodes = connectivity_matrix.shape[0]

#     # Initialize the distance matrix with infinity for all pairs
#     distance_matrix = np.full((num_nodes, num_nodes), np.inf)

#     # Create a NetworkX graph from the connectivity matrix
#     # We'll use a DiGraph because traffic might not flow symmetrically or costs might differ
#     G = nx.DiGraph()
#     for i in range(num_nodes):
#         G.add_node(i)
    
#     # Add edges to the graph where there's a connection (not infinity)
#     # The weight of the edge is the latency from the connectivity_matrix
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if connectivity_matrix[i, j] != np.inf and i != j: # Exclude self-loops from edges, though FW handles 0
#                 G.add_edge(i, j, weight=connectivity_matrix[i, j])

#     # Identify unique source nodes with non-zero traffic
#     # np.any(traffic_matrix, axis=1) returns a boolean array where True means the row has at least one non-zero
#     # np.where gets the indices of these rows (source nodes)
#     active_sources = np.where(np.any(traffic_matrix != 0, axis=1))[0]

#     # If traffic_matrix[i,j] != 0 implies traffic *from* i *to* j,
#     # then we only need to run Dijkstra from source nodes 'i' where traffic_matrix[i,:] has non-zero entries.
#     for source_node in active_sources:
#         # Run Dijkstra's algorithm from the current source node
#         # Returns a dictionary where keys are destinations and values are shortest path lengths
#         shortest_paths_from_source = nx.single_source_dijkstra_path_length(G, source_node, weight='weight')
        
#         # Populate the row in our distance_matrix
#         for dest_node, dist in shortest_paths_from_source.items():
#             distance_matrix[source_node, dest_node] = dist
            
#     # For self-loops (traffic from node to itself), distance is 0
#     np.fill_diagonal(distance_matrix, 0) # Assumes latency from node to itself is 0


#     # Element-wise multiplication of shortest path latency and traffic matrices
#     # If a path was unreachable (still np.inf in distance_matrix), the weighted latency will also be np.inf
#     size_weighted_latency = np.multiply(distance_matrix, traffic_matrix)

#     return size_weighted_latency

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

def calculate_network_metrics(connectivity_matrix: np.ndarray, traffic_matrix: np.ndarray):
    """
    Computes size-weighted latency and link utilization based on Floyd-Warshall
    shortest path routing.

    Args:
        connectivity_matrix (np.ndarray): A square matrix where connectivity_matrix[i,j]
            is the direct latency (cost) of the link from i to j.
            Use np.inf for no direct connection.
            Note: For standard shortest path, costs should be non-negative.
        traffic_matrix (np.ndarray): A square matrix where traffic_matrix[i,j]
            is the traffic demand from source i to destination j.

    Returns:
        tuple:
            - size_weighted_latency (np.ndarray): A square matrix representing
              the total latency for traffic between each source-destination pair.
              (latency * traffic).
            - link_utilization (np.ndarray): A square matrix where link_utilization[i,j]
              is the total traffic passing through the link from i to j.
    """
    num_nodes = connectivity_matrix.shape[0]
    
    # Initialize distance matrix based on connectivity matrix
    distance_matrix = np.copy(connectivity_matrix).astype(float)
    distance_matrix[~connectivity_matrix] = np.inf
    np.fill_diagonal(distance_matrix, 0)
    
    # Initialize next_hop matrix for path reconstruction
    # next_hop[i, j] stores the next node after i on the shortest path to j.
    # Initialize next_hop[i, j] = j if there's a direct connection, else -1 or num_nodes for no path
    next_hop = np.full((num_nodes, num_nodes), -1, dtype=int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                next_hop[i, j] = i # Or some indicator that it's the destination
            elif distance_matrix[i, j] != np.inf:
                next_hop[i, j] = j # Direct connection, next hop is the destination

    # --- Floyd-Warshall Algorithm with Path Reconstruction ---
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, k] != np.inf and distance_matrix[k, j] != np.inf:
                    if distance_matrix[i, k] + distance_matrix[k, j] < distance_matrix[i, j]:
                        distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]
                        next_hop[i, j] = next_hop[i, k] # The next hop from i to j is now the next hop from i to k

    # --- Calculate Link Utilization ---
    link_utilization = np.zeros((num_nodes, num_nodes), dtype=float)

    for src in range(num_nodes):
        for dest in range(num_nodes):
            if src == dest:
                continue # No traffic for self-loops in this context

            traffic = traffic_matrix[src, dest]
            if traffic > 0 and distance_matrix[src, dest] != np.inf: # Only consider if there's traffic and a path
                current_node = src
                # Reconstruct path and add traffic to links
                path_hops = 0 # To track hops and prevent infinite loops on invalid graphs
                while current_node != dest and path_hops < num_nodes: # Path can have at most V-1 hops in simple path
                    next_node = next_hop[current_node, dest]
                    if next_node == -1: # No path found, break
                        break
                    
                    link_utilization[current_node, next_node] += traffic
                    current_node = next_node
                    path_hops += 1
                
                # If path_hops reached num_nodes, it implies a loop or error in reconstruction for this path
                # This check helps debug if paths aren't forming correctly or graph has issues
                if current_node != dest and path_hops == num_nodes:
                    print(f"Warning: Path from {src} to {dest} exceeded max hops ({num_nodes}) during reconstruction. Likely a path issue or loop.")


    # --- Calculate Size-Weighted Latency ---
    # Element-wise multiplication of shortest path latency and traffic matrices
    # Paths that are np.inf (no connection) will result in np.inf weighted latency
    size_weighted_latency = np.multiply(distance_matrix, traffic_matrix)

    return size_weighted_latency, link_utilization, next_hop # Also returning next_hop for potential R_i,j,d insight


def plot_connectivity(positions: np.ndarray, 
                      C: np.ndarray, 
                      utilization: np.ndarray = None, 
                      figsize=(8,8), 
                      title_text = ": D",  
                      gamma_value = 0.8,
                      maxUtil = None ):
    """
    Plot satellites as points and ISL links as lines in 3D.
    Link colors vary by utilization if provided; otherwise, they default to gray.

    Parameters
    ----------
    
    positions : np.ndarray, shape (N,3)
        Cartesian coordinates of each satellite.
    C : np.ndarray, shape (N,N), dtype=bool or 0/1
        Connectivity matrix: C[i,j]=True/1 if there is a link between i and j.
    utilization : np.ndarray, shape (N,N), optional
        Utilization matrix: utilization[i,j] denotes the utilization of link (i,j).
        Higher values mean higher utilization. If None, links will be gray.
        The default is None.
    figsize : tuple, optional
        Figure size. The default is (8,8).
    title_text : str, optional
        Title for the plot. The default is ": D".
    gamma_value : float, optional
        Gamma value for utilization coloring. Changes the effective mapping from link utilization to color for links
    """

    active_util_values = utilization[C]
    # n, bins , patches = plt.hist(active_util_values, bins=20, range=(0, np.max(active_util_values)), edgecolor='black')
    # print(n)

    # plt.show()

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')
    #fig.set_facecolor('darkgrey') # Or any valid color string/hex code
    #ax.set_facecolor('darkgrey') # Or any valid color string/hex code
    
    # Plot satellites
    xs, ys, zs = positions[:,0], positions[:,1], positions[:,2]
    ax.scatter(xs, ys, zs, s=7, color='blue', label='Satellites')

    # Determine if utilization coloring should be applied
    use_utilization_coloring = utilization is not None and np.any(C) and np.any(utilization[C])

    if use_utilization_coloring:
        # Normalize utilization values only for existing links
        # Create a masked array or set non-connected link utilization to 0 for normalization
        # Or, just normalize over the values of existing links to avoid zeros from non-links skewing scale
        active_util_values = utilization[C] # Get utilization values only for active links
        
        # Ensure there's at least one non-zero active utilization to avoid division by zero
        if np.max(active_util_values) == 0:
            use_utilization_coloring = False # Fallback to default if all active utilizations are zero
        else:

            # #gamma_value = 0.4 # Experiment with this value (e.g., 0.1, 0.5)
            if(maxUtil == None):
                norm = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=np.max(active_util_values))
            else:
                norm = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=maxUtil)


            # colors = ["gray", "red"]

            # # Create the custom colormap
            # # The 'name' parameter is optional but good practice
            # cmap = mcolors.LinearSegmentedColormap.from_list("BluePurpleRed", colors)

            cmap = plt.get_cmap('Reds')

            #normalized_utilization = active_util_values / np.max(active_util_values)
            # Create a full normalized matrix for easier lookup during plotting
            full_normalized_utilization = np.zeros_like(utilization, dtype=float)
            full_normalized_utilization[C] = norm(active_util_values) # Apply normalized values back to C locations

            max_util_for_colorbar = np.max(active_util_values)
            link_default_linewidth = 1.5 # Restore original linewidth for default

            #pdb.set_trace()

    else:
        # Default settings if utilization is not provided or all active links have zero utilization
        link_default_color = 'gray'
        link_default_linewidth = 0.5 # Restore original linewidth for default

    # Plot links
    N = positions.shape[0]
    for i in range(N):  
        for j in range(i + 1, N): # To avoid plotting each link twice
            if C[i, j]: # If a link exists
                xline = [positions[i,0], positions[j,0]]
                yline = [positions[i,1], positions[j,1]]
                zline = [positions[i,2], positions[j,2]]

                if use_utilization_coloring:
                    # Get the normalized utilization for this link
                    # Use the stored normalized value
                    link_norm_util = max(full_normalized_utilization[i, j], full_normalized_utilization[j, i]) 
                    link_color = cmap(link_norm_util)[:-1] + (link_norm_util**gamma_value,) 

                    ax.plot(xline, yline, zline, color=link_color, linewidth=3)
                else:
                    ax.plot(xline, yline, zline, color=link_default_color, linewidth=link_default_linewidth)

    #create axis to show orientation
    axis_length = np.linalg.norm(positions[0]) * 1.2 # Make it a bit longer than the sphere's extent
    ax.plot([0, 0], [0, 0], [-axis_length, axis_length], color='black', linewidth=3, label='Polar Axis')

    # Styling
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-axis_length/1.5, axis_length/1.5)
    ax.set_ylim(-axis_length/1.5, axis_length/1.5)
    ax.set_zlim(-axis_length/1.5, axis_length/1.5)

    # No explicit legend for satellites if we're not using 'label' in scatter for color conflict
    # ax.legend() # Re-add if you put label back and handle it

    plt.tight_layout()

    # Additional mods
    ax.axis('off')
    #ax.set_title(title_text, fontsize=20)
    ax.grid(False)
    
    #ax.set_facecolor("Grey")
    #ax.set_facecolor("#a8a8a8")
    #plt.gca().set_facecolor('#f0f0f0')
    #plt.gcf().set_facecolor("#a8a8a8") 

    if use_utilization_coloring:
        # Crucially, pass the SAME `norm` object to ScalarMappable
        sm = cm.ScalarMappable(cmap=cmap, norm=norm) # Use the PowerNorm object
        sm.set_array(active_util_values) # Set the original values for the colorbar to map

        cbar = fig.colorbar(sm, ax=ax, shrink=1, aspect=30, pad=0, orientation='horizontal')

        cbar.set_label('Link Utilization (Bytes)', rotation=0, labelpad=5) # No rotation, smaller pad

        # 1. Get the formatter for the colorbar's ticks
        formatter = cbar.formatter

        # 2. Set the threshold for scientific notation
        #    'useOffset': False (removes the addition/subtraction of a constant from labels)
        #    'useMathText': True (renders exponents nicely with LaTeX-like math text)
        #    'scilimits': (lower, upper)
        #       Numbers with magnitude outside this range will be shown in scientific notation.
        #       e.g., (-2, 2) means numbers < 0.01 or > 100 will be scientific.
        #       Your numbers are large (20000), so if you set (0, 4) this means numbers outside
        #       10^0 and 10^4 (i.e., outside 1 and 10000) will use scientific.
        #       If you want 20000 to be scientific, you might set it to (0, 3) or similar.
        #       Let's try (0, 3) to convert 1000 and above to scientific.
        formatter.set_powerlimits((0, 3)) # Numbers outside [10^0, 10^3] (i.e., <1 or >1000) use scientific notation
        formatter.set_scientific(True) # Force scientific notation if conditions met
        formatter.set_useOffset(False) # Prevents offset (e.g. +1e5) if you don't want it
        formatter.set_useMathText(True) # Renders "x 10^y" nicely

        # 3. Update the colorbar's tick labels with the new formatter settings
        cbar.update_ticks()

    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.01, top=0.95) # Adjusted for 3D plot

    plt.show()

def plot_connectivity_old(positions: np.ndarray, C: np.ndarray, figsize=(8,8), title_text = ": D"):
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
    ax.scatter(xs, ys, zs, s=20, color='blue') #, label='Satellites')

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
    #ax.set_title('Walker-Δ Constellation +‑Grid Connectivity')
    ax.legend()
    plt.tight_layout()

    #additional mods 
    ax.axis('off')
    ax.set_title(title_text, fontsize=20) 
    ax.grid(False)

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
    unit_vecs = points_xyz / torch.norm(points_xyz, dim=1, keepdim=True)

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

def create_individual_param_reset_method(params_and_optimizers, epochs_to_track=3):
    """
    Creates methods to manage and reset specific nn.Parameter(s) and their optimizers
    to either a state 'epochs_to_track' epochs ago or to the overall best-performing state.

    Args:
        params_and_optimizers (list): A list of tuples, where each tuple is
                                      (parameter_tensor, optimizer_for_that_parameter).
                                      e.g., [(connectivity_logits, connOpt), (routing_logits, routeOpt)]
        epochs_to_track (int): The number of epochs to remember for rollback functionality.
    """
    state_history = {}
    best_loss_so_far = float('inf')
    best_state_info = None

    def update_state_history(current_epoch, current_loss):
        nonlocal best_loss_so_far, best_state_info

        current_states_snapshot = []
        for param, opt in params_and_optimizers:
            if param is not None and opt is not None:
                current_states_snapshot.append({
                    'param_state': param.data.clone(),
                    'optimizer_state': copy.deepcopy(opt.state_dict()) # Crucial deep copy
                })
            else:
                current_states_snapshot.append(None)

        state_history[current_epoch] = current_states_snapshot

        if len(state_history) > epochs_to_track + 2:
            oldest_epoch = min(state_history.keys())
            del state_history[oldest_epoch]

        if current_loss < best_loss_so_far:
            print(f"    --- New best loss found: {current_loss:.4f} (Epoch {current_epoch}) ---")
            best_loss_so_far = current_loss
            best_state_info = copy.deepcopy(current_states_snapshot) # Crucial deep copy

    def reset_to_previous_state(current_epoch):
        
        reset_epoch = current_epoch - epochs_to_track - 1

        if reset_epoch >= 0 and reset_epoch in state_history.keys():
            saved_states = state_history[reset_epoch]
            for i, (param, opt) in enumerate(params_and_optimizers):
                if param is not None and opt is not None and saved_states[i] is not None:
                    param.data.copy_(saved_states[i]['param_state'])
                    opt.load_state_dict(saved_states[i]['optimizer_state'])
                    # We'll use a .name attribute for cleaner printouts in this test
                    param_name = getattr(param, 'name', f"Parameter {i}")
                    print(f"    - Parameter '{param_name}' and its optimizer reset to state from epoch {reset_epoch}")
                elif param is not None and opt is not None:
                    param_name = getattr(param, 'name', f"Parameter {i}")
                    print(f"    - WARNING: State for '{param_name}' at epoch {reset_epoch} was not saved or is inactive.")
            print(f"All active parameters and optimizers reset to state from epoch {reset_epoch}")
        else:
            print(f"No saved state for epoch {reset_epoch} to reset to.")

    def reset_to_best_state():
        if best_state_info is not None:
            print(f"\n--- Resetting to best state with loss: {best_loss_so_far:.4f} ---")
            for i, (param, opt) in enumerate(params_and_optimizers):
                if param is not None and opt is not None and best_state_info[i] is not None:
                    param.data.copy_(best_state_info[i]['param_state'])
                    opt.load_state_dict(best_state_info[i]['optimizer_state'])
                    param_name = getattr(param, 'name', f"Parameter {i}")
                    print(f"    - Parameter '{param_name}' and its optimizer reset to best state.")
                elif param is not None and opt is not None:
                    param_name = getattr(param, 'name', f"Parameter {i}")
                    print(f"    - WARNING: Best state for '{param_name}' was not saved or is inactive.")
            print("--- Reset to best state complete ---\n")
        else:
            print("No best state recorded yet to reset to.")

    def find_closest_logit_history(target_logit_set):

        min_total_distance = float('inf')
        closest_epoch = None

        #for each epoch 
        for epoch_in_history, saved_states_snapshot in state_history.items():
            current_total_distance = 0.0
            
            #'optimizer_state':
            for i, (param_info, target_logit) in enumerate(saved_states_snapshot):
                if param_info is not None and target_logit is not None:
                    saved_param_data = param_info['param_state']
                    
                    if saved_param_data.shape != target_logit_set.shape:
                        current_total_distance = float('inf')
                        break
                    
                    distance = torch.dist(saved_param_data, target_logit_set, p=2)
                    current_total_distance += distance.item()
                elif param_info is None and target_logit is not None:
                    current_total_distance = float('inf')
                    break

            if current_total_distance < min_total_distance:
                min_total_distance = current_total_distance
                closest_epoch = epoch_in_history
        
        return closest_epoch, min_total_distance
    
    return update_state_history, reset_to_previous_state, reset_to_best_state, find_closest_logit_history

def graph_to_matrix_representation(graph: nx.Graph):# -> tuple[dict, np.ndarray]:

    if not graph.nodes:
        return {}, np.array([])

    # 1. Extract Node Positions
    node_positions = {}
    
    # Check if all nodes have 'pos' attribute and determine dimension
    first_node_id = list(graph.nodes())[0]
    if 'pos' not in graph.nodes[first_node_id]:
        raise ValueError("All nodes must have a 'pos' attribute for position extraction.")
    
    pos_dimension = len(graph.nodes[first_node_id]['pos'])

    for node_id in sorted(graph.nodes()): # Sort to ensure consistent matrix indexing
        pos = graph.nodes[node_id].get('pos')
        if pos is None:
            raise ValueError(f"Node {node_id} does not have a 'pos' attribute.")
        if len(pos) != pos_dimension:
            raise ValueError(f"Node {node_id} has position of dimension {len(pos)}, expected {pos_dimension}.")
        node_positions[node_id] = np.array(pos) # Store as numpy array for consistency

    hold =  [node_positions[ind] for ind in node_positions.keys()]
    hold_positions = np.vstack(hold)

    # 2. Create Connectivity / Edge Presence Matrix (Adjacency Matrix)
    num_nodes = graph.number_of_nodes()
    # Create a mapping from node ID to sorted index for matrix
    node_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(graph.nodes()))}
    
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for u, v in graph.edges():
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        adjacency_matrix[idx_u, idx_v] = 1
        adjacency_matrix[idx_v, idx_u] = 1 # Assuming undirected graph

    return hold_positions, adjacency_matrix

# The rest of your run_simulation function remains the same.
# Ensure that `myUtils` is correctly imported and `Baselines.funcedBaseline`
# (or the relevant functions from it) are accessible.


import math

def find_closest_divisor_to_sqrt(num_satellites: int) -> int:
    """
    Finds a divisor of num_satellites that is closest to its integer square root.

    Args:
        num_satellites (int): The number for which to find the divisor.
                              Must be a positive integer.

    Returns:
        int: The divisor of num_satellites that is closest to its integer square root.
             If there are two divisors equidistant from the integer square root,
             the smaller of the two is returned.

    Raises:
        ValueError: If num_satellites is not a positive integer.
    """
    if not isinstance(num_satellites, int) or num_satellites < 1:
        raise ValueError("num_satellites must be a positive integer.")

    # Handle the trivial case where num_satellites is 1
    if num_satellites == 1:
        return 1

    # Calculate the integer part of the square root
    target_sqrt = int(math.sqrt(num_satellites))

    # Iterate downwards from the target_sqrt to find a divisor
    # We look for the first divisor 'i' because its pair (num_satellites // i)
    # will be the 'other' divisor. These two divisors (i and num_satellites // i)
    # are the most likely candidates to be closest to the square root.
    for i in range(target_sqrt, 0, -1):
        if num_satellites % i == 0:
            divisor1 = i
            divisor2 = num_satellites // i

            # Compare which of the two divisors is closer to the original target_sqrt
            # If distances are equal, prefer the smaller divisor (divisor1)
            if abs(divisor1 - target_sqrt) <= abs(divisor2 - target_sqrt):
                return divisor1
            else:
                return divisor2

    # This part of the code should theoretically never be reached for num_satellites >= 1
    # as 1 is always a divisor, and the loop goes down to 1.
    return 1 # Fallback, though not expected to be hit


def size_weighted_latency_matrix_networkx(connectivity_matrix: np.ndarray, traffic_matrix: np.ndarray):
    num_nodes = connectivity_matrix.shape[0]

    distance_matrix = np.full((num_nodes, num_nodes), np.inf)

    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if connectivity_matrix[i, j] != np.inf and i != j:
                G.add_edge(i, j, weight=connectivity_matrix[i, j])

    active_sources = np.where(np.any(traffic_matrix != 0, axis=1))[0]

    for source_node in active_sources:
        shortest_paths_from_source = nx.single_source_dijkstra_path_length(G, source_node, weight='weight')
        for dest_node, dist in shortest_paths_from_source.items():
            distance_matrix[source_node, dest_node] = dist
            
    np.fill_diagonal(distance_matrix, 0) # Distance from node to itself is 0

    size_weighted_latency = np.multiply(distance_matrix, traffic_matrix)

    size_weighted_latency[np.isnan(size_weighted_latency)] = 0 # Replace NaNs (from 0*inf) with 0

    return size_weighted_latency

    #return [size_weighted_latency[traffic_matrix > 0]*traffic_matrix[traffic_matrix > 0]]

def plot_loss(epochs, losses):
    plt.rcParams.update({
        'font.size': 12,        # General font size for text
        'axes.labelsize': 14,   # Font size for x and y axis labels
        'axes.titlesize': 16,   # Font size for plot title
        'xtick.labelsize': 12,  # Font size for x-axis tick labels
        'ytick.labelsize': 12,  # Font size for y-axis tick labels
        'legend.fontsize': 12,  # Font size for legend
        'figure.titlesize': 18, # Font size for suptitle (if used)
        'figure.figsize': (8, 6), # Standard figure size (width, height) in inches
        'lines.linewidth': 2,   # Default line width
        'lines.markersize': 6,  # Default marker size
        'axes.grid': True,      # Enable grid by default (good for curves)
        'grid.alpha': 0.7,      # Transparency of the grid lines
        'grid.linestyle': '--', # Style of the grid lines
        'grid.linewidth': 0.5,  # Width of the grid lines
        'axes.edgecolor': 'black', # Color of the plot border
        'axes.linewidth': 1.0,  # Width of the plot border
    })

    plt.figure(figsize=(9, 6)) # Can override global figsize for specific plots
    plt.plot(epochs, losses, linestyle='-')


    plt.title('Model Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.6) # Can override global grid settings if needed
    plt.legend(loc='upper right', frameon=True, shadow=True, borderpad=1) # Customize legend
    plt.tight_layout() # Adjust plot parameters for a tight layout
    plt.show()

def calculate_algebraic_connectivity_torch(adj_matrix: torch.Tensor) -> torch.Tensor:
    if not isinstance(adj_matrix, torch.Tensor):
        raise TypeError("Input adj_matrix must be a PyTorch tensor.")
    
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1] or adj_matrix.shape[0] == 0:
        raise ValueError("Adjacency matrix must be a non-empty square 2D tensor.")

    if adj_matrix.dtype not in (torch.float32, torch.float64):
        adj_matrix = adj_matrix.to(torch.float32) # Or torch.float64 for higher precision

    num_vertices = adj_matrix.shape[0]

    if num_vertices < 2:
        return torch.tensor(0.0, dtype=adj_matrix.dtype, device=adj_matrix.device)

    degrees = torch.sum(adj_matrix, dim=1)
    D = torch.diag_embed(degrees)

    L = D - adj_matrix
    eigenvalues = torch.linalg.eigh(L).eigenvalues

    algebraic_connectivity = eigenvalues[1]

    return algebraic_connectivity

def calculate_generalized_algebraic_connectivity_torch(
    adj_matrix: torch.Tensor,
    node_importance_matrix_M: torch.Tensor
) -> torch.Tensor:

    num_vertices = adj_matrix.shape[0]

    if num_vertices < 2:
        return torch.tensor(0.0, dtype=adj_matrix.dtype, device=adj_matrix.device)

    degrees = torch.sum(adj_matrix, dim=1)
    D = torch.diag_embed(degrees)

    L = D - adj_matrix

    # --- Transformation for Generalized Eigenvalue Problem ---
    M_diag = torch.diag(node_importance_matrix_M)
    M_diag_sqrt_inv = 1.0 / torch.sqrt(M_diag)
    M_sqrt_inv = torch.diag_embed(M_diag_sqrt_inv) # This is M^(-1/2)

    L_prime = M_sqrt_inv @ L @ M_sqrt_inv

    eigenvalues = torch.linalg.eigh(L_prime).eigenvalues
    
    generalized_algebraic_connectivity = eigenvalues[1]

    return generalized_algebraic_connectivity


def calculate_algebraic_connectivity(adj_matrix):
    """
    Calculates the algebraic connectivity of a graph given its adjacency matrix.

    Args:
        adj_matrix (numpy.ndarray): A square 2D numpy array representing the
                                    adjacency matrix of the graph.
                                    Assumes it's an undirected graph (symmetric).

    Returns:
        float: The algebraic connectivity (Fiedler value) of the graph.
               Returns 0.0 if the graph is disconnected (or has multiple components),
               as the second smallest eigenvalue would be 0 in such cases.
               Returns None if the input is not a square matrix or is empty.
    """
    #adj_matrix = np.asarray(adj_matrix) # Ensure it's a numpy array

    # Check if the matrix is square and non-empty
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1] or adj_matrix.shape[0] == 0:
        print("Error: Adjacency matrix must be a non-empty square matrix.")
        return None

    num_vertices = adj_matrix.shape[0]

    # 1. Construct the Degree Matrix (D)
    # The degree of each vertex is the sum of its row (or column) in the adjacency matrix.
    degrees = np.sum(adj_matrix, axis=1)
    D = np.diag(degrees)

    # 2. Construct the Laplacian Matrix (L = D - A)
    L = D - adj_matrix

    # 3. Calculate the eigenvalues of the Laplacian Matrix
    # Using eigh for symmetric matrices is more efficient and numerically stable.
    # It returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = eigh(L)#, eigvals_only=True)

    # 4. Identify the Algebraic Connectivity
    # For a connected graph, the smallest eigenvalue is 0.
    # The algebraic connectivity is the second smallest eigenvalue.
    # We should account for potential floating-point inaccuracies
    # by checking if the first eigenvalue is very close to zero.

    if len(eigenvalues) < 2:
        # This case should ideally not happen for valid graphs with at least 1 vertex
        # but handles edge cases like 1x1 matrix.
        return 0.0 if len(eigenvalues) == 1 and np.isclose(eigenvalues[0], 0) else None

    # Sort eigenvalues (eigh already does this) and pick the second one.
    # We use a small tolerance to consider values very close to zero as zero.
    
    # Filter out eigenvalues very close to zero to properly identify the second smallest non-zero one
    non_zero_eigenvalues = eigenvalues[~np.isclose(eigenvalues, 0, atol=1e-9)]
    
    if len(non_zero_eigenvalues) == 0:
        # This implies all eigenvalues are zero, which is generally incorrect for a graph
        # unless it's a null graph or a single isolated vertex.
        return 0.0
    elif len(non_zero_eigenvalues) == 1:
        # If only one non-zero eigenvalue, the graph might be simple (e.g., K2)
        # or it means the graph is disconnected and the "second smallest" is still 0.
        # The true algebraic connectivity for a disconnected graph is 0.
        # In this specific context, the second smallest value in the sorted list (including 0s) is
        # the correct one to pick after the first 0.
        return eigenvalues[1] # If eigenvalues[0] is 0, this is the second smallest.
    else:
        # If there are multiple distinct non-zero eigenvalues, the smallest of them is the Fiedler value.
        # Since eigh sorts them, eigenvalues[1] will be the second smallest overall.
        return eigenvalues[1], eigenvectors[1]
    

def plot_edge_disjoint_paths_histogram(adj_matrix):
    """
    Calculates the number of edge-disjoint paths for all pairs in a graph
    defined by an adjacency matrix and plots a histogram of the results.

    This version uses the max-flow min-cut theorem and specifies the
    Edmonds-Karp algorithm for performance.

    Args:
        adj_matrix (list of lists or numpy array): The adjacency matrix of the graph.
    """
    # Create a graph from the adjacency matrix. For unweighted graphs,
    # networkx automatically assumes a capacity of 1 for each edge.
    G = nx.from_numpy_array(np.array(adj_matrix), edge_attr='capacity')
    
    # Get all unique pairs of nodes
    nodes = list(G.nodes())
    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]

    # Calculate the number of edge-disjoint paths for each pair using max flow
    path_counts = []
    print("Calculating edge-disjoint paths for all pairs...")
    for s, t in pairs:
        # The number of edge-disjoint paths equals the max flow in a unit-capacity network.
        # We specify Edmonds-Karp as requested.
        num_paths = nx.maximum_flow_value(G, s, t, flow_func=flow.edmonds_karp)
        path_counts.append(num_paths)
    print("Calculation complete.")

    # Count the frequency of each number of paths
    histogram_data = Counter(path_counts)

    # Prepare data for plotting
    paths = list(histogram_data.keys())
    counts = list(histogram_data.values())
    
    if not paths:
        print("No node pairs found or graph is empty.")
        return

    # --- Plotting the Histogram ---
    plt.figure(figsize=(10, 6))
    bar_container = plt.bar(paths, counts, color='teal', width=0.8)

    plt.xlabel("Number of Edge-Disjoint Paths", fontsize=12)
    plt.ylabel("Number of Satellite Pairs", fontsize=12)
    plt.title("Histogram of Edge-Disjoint Paths per Pair (using Edmonds-Karp)", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.bar_label(bar_container, fmt='{:,.0f}') # Add labels on top of bars
    
    # Set x-axis ticks to be integers for clarity
    max_paths = max(paths)
    plt.xticks(range(max_paths + 2))
    plt.xlim(-0.5, max_paths + 1.5)

    print("\n--- Histogram Data ---")
    print("(Number of Paths: Number of Pairs)")
    for path, count in sorted(histogram_data.items()):
        print(f"({path}, {count})")

    plt.show()
