#This file is for integrating with baseline / getting their results 
import myUtils
import numpy as np
import pdb 
import torch 
import lightweightMotif
import math 
from scipy.spatial.distance import cdist
import datetime
from multiprocessing import Process
import multiprocessing

#import their stuff 
import util
import networkx as nx
import sys

#import baseline
from Baselines.funcedBaseline import calculate_weighted_latency

# NumPy
np.random.seed(0)
# Torch 
torch.manual_seed(0)

#simulation params 
trafficScaling = 100000
max_hops    = 20       # how many hops to unroll
maxDemand   = 1.0
numFlows = 30
beam_budget = 4      # sum of beam allocations per node

#constellation params 
orbitRadius = 6.946e6   

numSatellites = 100
NUM_ORBITS = 20
NUM_SATS_PER_ORBIT = numSatellites / NUM_ORBITS
inclination = 80 
phasingParameter = 5

#earth params
EARTH_MEAN_RADIUS = 6371.0 # km

CORE_CNT = int(sys.argv[1])

#create funcs:
def convert_feasible_mask_to_valid_isls(feasible_mask: np.ndarray,
                                         sat_positions_tensor: np.ndarray,
                                         distance_matrix: torch.Tensor = None) -> list:
    """
    Converts a 2D boolean feasible mask into a list of valid ISL dictionaries.

    Args:
        feasible_mask (torch.Tensor): A 2D boolean tensor where feasible_mask[i, j]
                                      is True if a link from satellite i to satellite j is feasible.
                                      Assumed to be `(num_satellites, num_satellites)`.
        sat_positions_tensor (torch.Tensor): A tensor of shape (num_satellites, 3)
                                            containing the Cartesian coordinates for all satellites.
                                            This is used by compute_isl_length if distance_matrix is None.
        distance_matrix (torch.Tensor, optional): An optional 2D tensor of distances
                                                  corresponding to the feasible_mask.
                                                  If provided, distances will be directly
                                                  extracted from here. If None,
                                                  compute_isl_length will be called for each link.

    Returns:
        list: A list of dictionaries, where each dictionary represents a valid ISL
              in the format: {'sat_1': int, 'sat_2': int, 'dist_km': float}.
    """
    valid_isls = []
    num_satellites = feasible_mask.shape[0]

    for i in range(num_satellites):
        for j in range(i + 1, num_satellites): # Iterate through upper triangular part
            if feasible_mask[i, j]: # Check if the link is feasible
                if distance_matrix is not None:
                    # If distance_matrix is provided, use its value directly
                    dist = distance_matrix[i, j]

                valid_isls.append({
                    'sat_1': i,
                    'sat_2': j,
                    'dist_km': dist
                })
    return valid_isls

def find_motif_possibilities():
        
    #after feasible mask, generate ISLs 
    valid_isls = convert_feasible_mask_to_valid_isls(feasibleMask, torch.from_numpy(positions).to(dtype=torch.float32), dmat)


    """
    Get all feasible north and right links for the satellite with id 0
    """
    sat_id = 0
    ORB_OFFSET = NUM_ORBITS / 4
    valid_motif_links = {}
    valid_link_cnt = 0
    for i in range(len(valid_isls)):
        if valid_isls[i]["sat_1"] == sat_id and valid_isls[i]["sat_2"] > sat_id and \
                sat_positions[valid_isls[i]["sat_2"]]["orb_id"] < ORB_OFFSET:
            orb_id = math.floor(valid_isls[i]["sat_2"] / NUM_SATS_PER_ORBIT)
            sat_rel_id = valid_isls[i]["sat_2"] - sat_id - orb_id * NUM_SATS_PER_ORBIT
            if sat_rel_id - sat_id > NUM_SATS_PER_ORBIT / 4:
                sat_rel_id = sat_rel_id - NUM_SATS_PER_ORBIT
            if not (orb_id == 0 and sat_rel_id < 0):
                # print(valid_isls[i]["sat_2"], orb_id, sat_rel_id)
                valid_motif_links[valid_link_cnt] = {
                    "sat_id": valid_isls[i]["sat_2"],
                    "orb_id": orb_id,
                    "sat_rel_id": sat_rel_id
                }
                valid_link_cnt += 1
    # Combined motif possibilities
    # For same orbit, select the other link from different orbit
    motif_possibilities = {}
    motif_cnt = 0
    for i in range(len(valid_motif_links) - 1):
        for j in range(i + 1, len(valid_motif_links)):
            if not (valid_motif_links[i]["orb_id"] == 0 and valid_motif_links[j]["orb_id"] == 0) and not (
                    valid_motif_links[i]["sat_id"] == valid_motif_links[j]["sat_id"]):
                # print(valid_motif_links[i]["sat_id"], valid_motif_links[j]["sat_id"])
                motif_possibilities[motif_cnt] = {
                    "motif_cnt": motif_cnt,
                    "sat_1_id": valid_motif_links[i]["sat_id"],
                    "sat_1_orb_offset": valid_motif_links[i]["orb_id"],
                    "sat_1_sat_offset": valid_motif_links[i]["sat_rel_id"],
                    "sat_2_id": valid_motif_links[j]["sat_id"],
                    "sat_2_orb_offset": valid_motif_links[j]["orb_id"],
                    "sat_2_sat_offset": valid_motif_links[j]["sat_rel_id"],
                    "wStretch": -1.0,
                    "wHop": -1.0,
                    "wMetric": -1.0,
                    "future": None
                } 
                motif_cnt += 1
    return motif_possibilities
 
def ecef_to_lat_lon_alt_spherical(x: float, y: float, z: float) -> tuple:
    """
    Converts Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates (x, y, z)
    to geodetic latitude (deg, rad), longitude (deg, rad), and altitude (km)
    **assuming a perfectly spherical Earth**.

    Args:
        x, y, z (float): ECEF coordinates in kilometers.

    Returns:
        tuple: (lat_deg, lon_deg, alt_km, lat_rad, lon_rad)
    """
    # Distance from the Earth's center (radius of the satellite's position vector)
    r = math.sqrt(x**2 + y**2 + z**2)

    # Longitude
    lon_rad = math.atan2(y, x)
    lon_deg = math.degrees(lon_rad)

    # Latitude
    # z = r * sin(lat_rad)  => lat_rad = asin(z / r)
    if r == 0: # Handle the case where the position is at the origin
        lat_rad = 0.0 # Or raise an error, depending on desired behavior
    else:
        lat_rad = math.asin(z / r)
    lat_deg = math.degrees(lat_rad)

    # Altitude above the spherical Earth's surface
    alt_km = r - EARTH_MEAN_RADIUS

    return lat_deg, lon_deg, alt_km, lat_rad, lon_rad

def convert_positions_to_sat_dict(positions: np.ndarray) -> dict:
    """
    Converts a 3D NumPy array of satellite Cartesian positions
    (shape: num_shells x sats_per_shell x 3)
    into the 'sat_positions' dictionary format.

    Args:
        positions (np.ndarray): A NumPy array of satellite Cartesian coordinates
                                in kilometers, shape (num_shells, sats_per_shell, 3).

    Returns:
        dict: A dictionary where keys are unique satellite IDs (int) and values
              are dictionaries containing 'orb_id', 'orb_sat_id', 'lat_deg',
              'long_deg', 'alt_km', 'lat_rad', 'long_rad'.
    """
    sat_positions_dict = {}
    num_shells, sats_per_shell, _ = positions.shape

    for orb_id in range(num_shells):
        for orb_sat_id in range(sats_per_shell):
            sat_id = orb_id * sats_per_shell + orb_sat_id
            x, y, z = positions[orb_id, orb_sat_id]

            lat_deg, lon_deg, alt_km, lat_rad, lon_rad = ecef_to_lat_lon_alt_spherical(x, y, z)

            sat_positions_dict[sat_id] = {
                'orb_id': orb_id,
                'orb_sat_id': orb_sat_id,
                'lat_deg': lat_deg,
                'lat_rad': lat_rad,
                'long_deg': lon_deg,
                'long_rad': lon_rad,
                'alt_km': alt_km
            }
    return sat_positions_dict

def find_motif_possibilities():
    """
    Get all feasible north and right links for the satellite with id 0
    """
    sat_id = 0
    ORB_OFFSET = NUM_ORBITS / 4
    valid_motif_links = {}
    valid_link_cnt = 0
    for i in range(len(valid_isls)):
        if valid_isls[i]["sat_1"] == sat_id and valid_isls[i]["sat_2"] > sat_id and \
                sat_positions[valid_isls[i]["sat_2"]]["orb_id"] < ORB_OFFSET:
            orb_id = math.floor(valid_isls[i]["sat_2"] / NUM_SATS_PER_ORBIT)
            sat_rel_id = valid_isls[i]["sat_2"] - sat_id - orb_id * NUM_SATS_PER_ORBIT
            if sat_rel_id - sat_id > NUM_SATS_PER_ORBIT / 4:
                sat_rel_id = sat_rel_id - NUM_SATS_PER_ORBIT
            if not (orb_id == 0 and sat_rel_id < 0):
                # print(valid_isls[i]["sat_2"], orb_id, sat_rel_id)
                valid_motif_links[valid_link_cnt] = {
                    "sat_id": valid_isls[i]["sat_2"],
                    "orb_id": orb_id,
                    "sat_rel_id": sat_rel_id
                }
                valid_link_cnt += 1
    # Combined motif possibilities
    # For same orbit, select the other link from different orbit
    motif_possibilities = {}
    motif_cnt = 0
    for i in range(len(valid_motif_links) - 1):
        for j in range(i + 1, len(valid_motif_links)):
            if not (valid_motif_links[i]["orb_id"] == 0 and valid_motif_links[j]["orb_id"] == 0) and not (
                    valid_motif_links[i]["sat_id"] == valid_motif_links[j]["sat_id"]):
                # print(valid_motif_links[i]["sat_id"], valid_motif_links[j]["sat_id"])
                motif_possibilities[motif_cnt] = {
                    "motif_cnt": motif_cnt,
                    "sat_1_id": valid_motif_links[i]["sat_id"],
                    "sat_1_orb_offset": valid_motif_links[i]["orb_id"],
                    "sat_1_sat_offset": valid_motif_links[i]["sat_rel_id"],
                    "sat_2_id": valid_motif_links[j]["sat_id"],
                    "sat_2_orb_offset": valid_motif_links[j]["orb_id"],
                    "sat_2_sat_offset": valid_motif_links[j]["sat_rel_id"],
                    "wStretch": -1.0,
                    "wHop": -1.0,
                    "wMetric": -1.0,
                    "future": None
                } 
                motif_cnt += 1
    return motif_possibilities


def add_motif_links_to_graph(grph, motif):
    """
    Adds ISLs to graph based on the current motif
    :param grph: The graph under consideration with nodes being the satellites and/or cities
    :param motif: Motif containing the relative positions of the neighboring satellites
    :return: returns the updated graph
    """
    for i in sat_positions:
        sel_sat_id = util.get_neighbor_satellite(sat_positions[i]["orb_id"], sat_positions[i]["orb_sat_id"],
                                                 motif["sat_1_orb_offset"], motif["sat_1_sat_offset"], sat_positions,
                                                 NUM_ORBITS, NUM_SATS_PER_ORBIT)
        is_possible = util.check_edge_availability(grph, i, sel_sat_id)
        if is_possible:
            dist = util.compute_isl_length(i, sel_sat_id, sat_positions)
            grph.add_edge(i, sel_sat_id, length=dist)
        sel_sat_id = util.get_neighbor_satellite(sat_positions[i]["orb_id"], sat_positions[i]["orb_sat_id"],
                                                 motif["sat_2_orb_offset"], motif["sat_2_sat_offset"], sat_positions,
                                                 NUM_ORBITS, NUM_SATS_PER_ORBIT)
        is_possible = util.check_edge_availability(grph, i, sel_sat_id)
        if is_possible:
            dist = util.compute_isl_length(i, sel_sat_id, sat_positions)
            grph.add_edge(i, sel_sat_id, length=dist)
    print("total edges", grph.number_of_edges())
    return grph

def run_motif_analysis(grph, motif_cnt, motif, return_dict):
    """
    Runs motif analysis for individual motifs
    :param grph: The graph under consideration
    :param motif_cnt: Motif counter
    :param motif: Motif
    :param return_dict: The return values
    """
    grph = add_motif_links_to_graph(grph, motif)
    retVal = compute_metric_avoid_city(grph)
    motif["wMetric"] = retVal["wMetric"]
    motif["wStretch"] = retVal["avgWeightedStretch"]
    motif["wHop"] = retVal["avgWeightedHopCount"]
    motif["weightedLatency"] = retVal["weightedLatency"]

    return_dict[motif_cnt] = motif


def compute_metric_avoid_city(grph):
    """
    Computes city-pair wise stretch and hop count and the aggregate metric values
    :param grph: The graph with satellites and cities as nodes
    :return: Computed aggregated metric
    """

    weightSum = 0
    weightedStretchSum = 0
    weightedHopCountSum = 0
    weightedLatency = 0
    a = datetime.datetime.now()
    for i in range(len(city_pairs)):
        city1 = city_pairs[i]["city_1"]
        city2 = city_pairs[i]["city_2"]
        geoDist = city_pairs[i]["geo_dist"]
        try:
            util.add_coverage_for_city(grph, city1, city_coverage)
            util.add_coverage_for_city(grph, city2, city_coverage)
            distance = nx.shortest_path_length(grph, source=city1, target=city2, weight='length')
            path = nx.shortest_path(grph, source=city1, target=city2, weight='length')

            hops = path.__len__() - 1
            stretch = distance / geoDist

            #weight = city_positions[city1]["pop"] * city_positions[city2]["pop"] / 10000000
            weight = flow_pops[i] / 10000000

            weightSum += weight
            weightedStretchSum += stretch * weight
            weightedHopCountSum += hops * weight

            weightedLatency+=distance*flow_pops[i]

            util.remove_coverage_for_city(grph, city1, city_coverage)
            util.remove_coverage_for_city(grph, city2, city_coverage)
        except Exception as e:
            util.remove_coverage_for_city(grph, city1, city_coverage)
            util.remove_coverage_for_city(grph, city2, city_coverage)
            return_val = {
                "avgWeightedStretch": 99999.0,
                "avgWeightedHopCount": 99999.0,
                "wMetric": 99999.0,
                "weightedLatency": 9999999999999999999999
            }
            return return_val
    avgWeightedStretch = weightedStretchSum / weightSum
    avgWeightedHopCount = weightedHopCountSum / weightSum
    wMetric = avgWeightedStretch + avgWeightedHopCount

    b = datetime.datetime.now() - a
    print("time to compute metric:", b.seconds, ", avgWeightedStretch:", avgWeightedStretch, ", avgWeightedHopCount:",
          avgWeightedHopCount)

    return_val = {
        "avgWeightedStretch": avgWeightedStretch,
        "avgWeightedHopCount": avgWeightedHopCount,
        "wMetric": wMetric,
        "weightedLatency":weightedLatency

    }
    return return_val

# Generate constellation
positions, vecs = myUtils.generateWalkerStarConstellationPoints(numSatellites,
                                            inclination,
                                            NUM_ORBITS,
                                            phasingParameter,
                                            orbitRadius)

#positions: orbital shell # x satellite ind x xyz
#convert to proper form 
sat_positions = convert_positions_to_sat_dict(positions)

shapedPositions = np.reshape(positions, [np.shape(positions)[0]*np.shape(positions)[1], np.shape(positions)[2]])
#get reasonable connectivity
feasibleMask = myUtils.check_los(shapedPositions, shapedPositions)  # (dmat <= max_dist) & (~torch.eye(numSatellites, dtype=bool))   # feasible edges

#get distance matrix in km
dmat = cdist(shapedPositions, shapedPositions) * 10e3                            # [N,N]

#pdb.set_trace()

#then, get the valid ISLs
valid_isls = convert_feasible_mask_to_valid_isls(feasibleMask, positions, dmat)

#after getting the valid ISLs, get the motif possibilities
motif_possibilities = find_motif_possibilities()

# Get all motif possibilities
valid_motif_possibilities = find_motif_possibilities()

#generate city pairs
numCities = 100
src_indices = np.random.choice(np.arange(numSatellites), size=numFlows, replace=False)
available   = np.setdiff1d(np.arange(numSatellites), src_indices)
dst_indices = np.random.choice(available, size=numFlows, replace=False)

#get inter great circle dist
greatCircleCross = myUtils.great_circle_distance_matrix_cartesian(shapedPositions, orbitRadius)

#generate city_pairs 
city_pairs = {i: {'city_1': src, 'city_2': dst, 'geo_dist': greatCircleCross[src, dst].item()}
              for i, (src, dst) in enumerate(zip(src_indices, dst_indices))}

#generate locations
city_positions = shapedPositions
flow_pops = np.random.randint(10000000, size=numFlows)
#city_positions = [{'city': i, 'x': city_positions[i][0], 'y': city_positions[i][1], 'z': city_positions[i][2], 'pop': np.random.randint(10000000)}]

#generate coverage utilizing pass through logic....
city_coverage =  [
        {'city': i, 'sat': i, 'dist': 0.0}
        for i in range(numSatellites)
    ]

#generate initial graph 
G = nx.Graph()

# Add cities to graph 
for i in range(len(city_positions)):
    G.add_node(city_coverage[i]["city"])
    
# For each motif compute metrics
for i in range(0, len(valid_motif_possibilities), CORE_CNT):
    threads = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # Parallelize metric computation based on available cores
    for j in range(0, CORE_CNT):
        id = i + j
        if id < len(valid_motif_possibilities):
            print("Generating graph for motif:", valid_motif_possibilities[id]["sat_1_id"], ",",
                  valid_motif_possibilities[id]["sat_2_id"])
            p = Process(target=run_motif_analysis,
                        args=(G.copy(), id, valid_motif_possibilities[id], return_dict))
            threads.append(p)
    for x in threads:
        x.start()
    for x in threads:
        x.join()

    for value in return_dict.values():
        valid_motif_possibilities[value["motif_cnt"]]["wMetric"] = value["wMetric"]
        valid_motif_possibilities[value["motif_cnt"]]["wStretch"] = value["wStretch"]
        valid_motif_possibilities[value["motif_cnt"]]["wHop"] = value["wHop"]
        valid_motif_possibilities[value["motif_cnt"]]["weightedLatency"] = value["weightedLatency"]

# Get the best motif based on the aggregated metric value
print(min([x['weightedLatency'] for x in valid_motif_possibilities.values()]))
# please note, that existence of a motif does not imply a path exists between all pairs 
best_motif = util.get_best_motif_at_level(valid_motif_possibilities)

