import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Constants ---
EARTH_RADIUS_KM = 6371  # Approximate Earth radius in kilometers

# --- Utility Functions ---

def to_radians(degrees):
    """Converts degrees to radians."""
    return degrees * (math.pi / 180)

def lat_lon_to_xyz(lat, lon, radius=1.0):
    """
    Converts latitude and longitude (degrees) to 3D Cartesian coordinates (x, y, z)
    on a sphere of given radius.
    """
    lat_rad = to_radians(lat)
    lon_rad = to_radians(lon)

    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = radius * math.cos(lat_rad) * math.sin(lon_rad)
    z = radius * math.sin(lat_rad)
    return np.array([x, y, z])

def normalize_vector(vec):
    """Normalizes a 3D vector (numpy array) to a unit vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return vec / norm

def cartesian_great_circle_distance(xyz1, xyz2):
    """
    Calculates the great-circle distance between two points given their
    3D Cartesian coordinates on a sphere.
    Assumes inputs are on a sphere of EARTH_RADIUS_KM.
    """
    # Normalize vectors to unit length for angle calculation
    unit_xyz1 = normalize_vector(xyz1)
    unit_xyz2 = normalize_vector(xyz2)

    # Dot product of two unit vectors gives the cosine of the angle between them
    dot_product = np.dot(unit_xyz1, unit_xyz2)

    # Clamp dot product to [-1, 1] to avoid floating point errors with acos
    dot_product = max(-1.0, min(1.0, dot_product))

    # Angle in radians
    angle_rad = math.acos(dot_product)

    # Great-circle distance = angle * radius
    distance = angle_rad * EARTH_RADIUS_KM
    return distance

def calculate_tir(eg_coords_xyz, ei_coords_xyz, ed_coords_xyz):
    """
    Calculates the Triangle Inequality Ratio (TIR) for a path e_g -> e_i -> e_d
    using XYZ Cartesian coordinates.
    TIR = (Direct Distance e_g to e_d) / (Distance e_g to e_i + Distance e_i to e_d)
    """
    dist_ge_ei = cartesian_great_circle_distance(eg_coords_xyz, ei_coords_xyz)
    dist_ei_ed = cartesian_great_circle_distance(ei_coords_xyz, ed_coords_xyz)
    dist_ge_ed = cartesian_great_circle_distance(eg_coords_xyz, ed_coords_xyz)

    sum_path_distances = dist_ge_ei + dist_ei_ed

    if sum_path_distances < 1e-9:  # Handle near-zero sum (points are practically identical)
        return 1.0
    
    # Ensure TIR doesn't exceed 1.0 due to floating point inaccuracies
    tir = dist_ge_ed / sum_path_distances
    return min(tir, 1.0) # Cap at 1.0 to handle potential floating point issues when points are collinear

def get_great_circle_arc_points(xyz1, xyz2, num_points=100):
    """
    Generates points along the great circle arc between two 3D vectors.
    """
    v1 = normalize_vector(xyz1)
    v2 = normalize_vector(xyz2)

    angle = math.acos(np.dot(v1, v2))

    if angle < 1e-9: # Points are identical or very close
        return np.array([v1]) * EARTH_RADIUS_KM
    elif abs(angle - math.pi) < 1e-9: # Points are antipodal
        # For antipodal points, cross with a non-collinear vector to get an axis.
        # Try (1,0,0), if collinear, try (0,1,0)
        temp_vec = np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(np.cross(v1, temp_vec)) < 1e-6:
            temp_vec = np.array([0.0, 1.0, 0.0])
        
        axis = normalize_vector(np.cross(v1, temp_vec))
        # Rotate v1 around this axis
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            rotation_matrix = rotation_matrix_from_axis_angle(axis, angle * t)
            points.append(np.dot(rotation_matrix, v1) * EARTH_RADIUS_KM)
        return np.array(points)
    else:
        axis = normalize_vector(np.cross(v1, v2))
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            rotation_matrix = rotation_matrix_from_axis_angle(axis, angle * t)
            points.append(np.dot(rotation_matrix, v1) * EARTH_RADIUS_KM)
        return np.array(points)

def rotation_matrix_from_axis_angle(axis, angle):
    """Generates a rotation matrix for a given axis and angle (Rodrigues' rotation formula)."""
    axis = normalize_vector(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.identity(3)
    R = I + math.sin(angle) * K + (1 - math.cos(angle)) * np.dot(K, K)
    return R

def plot_sphere_and_paths(eg_xyz, ei_xyz, ed_xyz, case_name, description, distances, tir_value):
    """
    Generates and displays a 3D plot of the sphere, points, and paths.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y_sphere = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z_sphere = EARTH_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='cyan', alpha=0.1, linewidth=0)

    # Plot points
    ax.scatter(eg_xyz[0], eg_xyz[1], eg_xyz[2], color='green', s=100, label='Eg (Given)', edgecolors='black')
    ax.scatter(ei_xyz[0], ei_xyz[1], ei_xyz[2], color='blue', s=100, label='Ei (Intermediate)', edgecolors='black')
    ax.scatter(ed_xyz[0], ed_xyz[1], ed_xyz[2], color='red', s=100, label='Ed (Destination)', edgecolors='black')

    # Plot paths
    # Path Eg to Ei
    path_ge_ei_points = get_great_circle_arc_points(eg_xyz, ei_xyz)
    ax.plot(path_ge_ei_points[:, 0], path_ge_ei_points[:, 1], path_ge_ei_points[:, 2],
            color='orange', linewidth=2, label='Path Eg-Ei')

    # Path Ei to Ed
    path_ei_ed_points = get_great_circle_arc_points(ei_xyz, ed_xyz)
    ax.plot(path_ei_ed_points[:, 0], path_ei_ed_points[:, 1], path_ei_ed_points[:, 2],
            color='orange', linewidth=2, label='Path Ei-Ed')

    # Direct Path Eg to Ed
    path_ge_ed_points = get_great_circle_arc_points(eg_xyz, ed_xyz)
    ax.plot(path_ge_ed_points[:, 0], path_ge_ed_points[:, 1], path_ge_ed_points[:, 2],
            color='purple', linestyle='--', linewidth=2, label='Direct Path Eg-Ed')

    # Set plot limits and labels
    limit = EARTH_RADIUS_KM * 1.1 # Extend limits slightly beyond sphere radius
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(f"{case_name}\n{description}\nTIR: {tir_value:.4f}", fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.set_aspect('equal') # Ensure spherical aspect ratio

    plt.tight_layout()
    plt.show()

# --- Test Cases (defined in Lat/Lon for easy understanding, converted to XYZ) ---
test_cases_lat_lon = [
    {
        "name": "Case 1: Ideal Path (TIR ≈ 1)",
        "description": "Intermediate point (Ei) is directly on the great-circle path from Eg to Ed.",
        "eg": (0, 0),
        "ei": (0, 45),
        "ed": (0, 90)
    },
    {
        "name": "Case 2: Significant Detour (TIR < 1)",
        "description": "Intermediate point (Ei) is off the direct path, creating a longer route.",
        "eg": (0, 0),
        "ei": (45, 45),  # North-east detour
        "ed": (0, 90)
    },
    {
        "name": "Case 3: Antipodal Destination (Ed is opposite Eg)",
        "description": "Eg and Ed are antipodal. Ei is an arbitrary point. TIR correctly reflects the detour.",
        "eg": (0, 0),
        "ei": (30, 90),
        "ed": (0, 180) # Antipodal to (0,0)
    },
    {
        "name": "Case 4: Antipodal Intermediate (Ei is opposite Eg)",
        "description": "Ei is antipodal to Eg. This forces a path back towards Ed. TIR remains valid.",
        "eg": (0, 0),
        "ei": (0, 180), # Antipodal to (0,0)
        "ed": (45, 90)
    },
    {
        "name": "Case 5: Short Path (US Cities)",
        "description": "Demonstrates TIR for shorter, more realistic distances.",
        "eg": (34.0522, -118.2437), # Los Angeles
        "ei": (37.7749, -122.4194), # San Francisco
        "ed": (39.7392, -104.9903)  # Denver
    },
    {
        "name": "Case 6: Identical Start and End Points",
        "description": "If Eg and Ed are the same, TIR should be 1.0.",
        "eg": (10, 10),
        "ei": (20, 20),
        "ed": (10, 10)
    },
    {
        "name": "Case 7: All Points Identical",
        "description": "If all three points are the same, TIR is 1.0.",
        "eg": (50, 50),
        "ei": (50, 50),
        "ed": (50, 50)
    }
]

# --- Run Test Cases and Plot ---
print("--- Triangle Inequality Ratio (TIR) Demonstration (XYZ Coordinates with Matplotlib) ---")
print(f"Earth Radius used for calculations: {EARTH_RADIUS_KM} km\n")

for i, case in enumerate(test_cases_lat_lon):
    # Convert Lat/Lon to XYZ for current test case
    eg_xyz = lat_lon_to_xyz(case['eg'][0], case['eg'][1], EARTH_RADIUS_KM)
    ei_xyz = lat_lon_to_xyz(case['ei'][0], case['ei'][1], EARTH_RADIUS_KM)
    ed_xyz = lat_lon_to_xyz(case['ed'][0], case['ed'][1], EARTH_RADIUS_KM)

    dist_ge_ei = cartesian_great_circle_distance(eg_xyz, ei_xyz)
    dist_ei_ed = cartesian_great_circle_distance(ei_xyz, ed_xyz)
    dist_ge_ed = cartesian_great_circle_distance(eg_xyz, ed_xyz)

    sum_path_distances = dist_ge_ei + dist_ei_ed
    tir_value = calculate_tir(eg_xyz, ei_xyz, ed_xyz)

    distances = {
        "ge_ei": dist_ge_ei,
        "ei_ed": dist_ei_ed,
        "ge_ed": dist_ge_ed,
        "sum_path": sum_path_distances
    }

    print(f"--- {case['name']} ---")
    print(f"Description: {case['description']}")
    print(f"Eg (Given Node): Lat {case['eg'][0]:.2f}, Lon {case['eg'][1]:.2f}")
    print(f"Ei (Intermediate Node): Lat {case['ei'][0]:.2f}, Lon {case['ei'][1]:.2f}")
    print(f"Ed (Destination Node): Lat {case['ed'][0]:.2f}, Lon {case['ed'][1]:.2f}")
    print(f"  Distance Eg to Ei: {distances['ge_ei']:.2f} km")
    print(f"  Distance Ei to Ed: {distances['ei_ed']:.2f} km")
    print(f"  Direct Distance Eg to Ed: {distances['ge_ed']:.2f} km")
    print(f"  Sum of Path Distances (Eg-Ei-Ed): {distances['sum_path']:.2f} km")
    print(f"  TIR (Direct / Sum): {tir_value:.4f}")
    print("-" * 40 + "\n")

    plot_sphere_and_paths(eg_xyz, ei_xyz, ed_xyz, case['name'], case['description'], distances, tir_value)

