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

def normalization_score(A, ref=1.0, epsilon=1e-8):
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

import numpy as np

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