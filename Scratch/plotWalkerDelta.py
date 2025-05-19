import numpy as np
import myUtils 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pdb 

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

#constellation params 
orbitRadius = 6.946e6   
numSatellites = 360
orbitalPlanes = 10
inclination = 80 
phasingParameter = 5

# suppose you have 10 planes, 36 sats per plane, and positions[i] corresponds
# to plane = i//36, slot = i%36
num_planes = 10
sats_per_plane = 36
positions,_ = myUtils.generateWalkerStarConstellationPoints(
    sats_per_plane*num_planes, inclination, num_planes, phasingParameter, orbitRadius
)
#pdb.set_trace()
# reshape to (N,3) if needed...
positions = np.reshape(positions, [np.shape(positions)[0]*np.shape(positions)[1], np.shape(positions)[2]])

C = build_plus_grid_connectivity(positions, num_planes, sats_per_plane)
print("Connectivity shape:", C.shape)
print("Example links for sat 0:", np.where(C[0])[0])

plot_connectivity(positions, C)