import myUtils
import numpy as np
import torch
import math
from scipy.spatial.distance import cdist
import datetime
from multiprocessing import Process, Manager
import networkx as nx
#import Simulator.Baselines.util
from . import util
#from Baselines.TwentySevenK.satnetwork.github.io.scripts import util
import sys

# NumPy
np.random.seed(0)
# Torch 
torch.manual_seed(0)

def ecef_to_lat_lon_alt_spherical(x: float, y: float, z: float, EARTH_MEAN_RADIUS: float) -> tuple:
    """
    Converts Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates (x, y, z)
    to geodetic latitude (deg, rad), longitude (deg, rad), and altitude (km)
    **assuming a perfectly spherical Earth**.

    Args:
        x, y, z (float): ECEF coordinates in kilometers.

    Returns:
        tuple: (lat_deg, lon_deg, alt_km, lat_rad, lon_rad)
    """
    r = math.sqrt(x**2 + y**2 + z**2)

    lon_rad = math.atan2(y, x)
    lon_deg = math.degrees(lon_rad)

    if r == 0:
        lat_rad = 0.0
    else:
        lat_rad = math.asin(z / r)
    lat_deg = math.degrees(lat_rad)

    alt_km = r - EARTH_MEAN_RADIUS

    return lat_deg, lon_deg, alt_km, lat_rad, lon_rad


def convert_positions_to_sat_dict(positions: np.ndarray, EARTH_MEAN_RADIUS: float) -> dict:
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

            lat_deg, lon_deg, alt_km, lat_rad, lon_rad = ecef_to_lat_lon_alt_spherical(x, y, z, EARTH_MEAN_RADIUS)

            sat_positions_dict[sat_id] = {
                'orb_id': orb_id,
                'orb_sat_id': orb_sat_id,
                'lat_deg': lat_deg,
                'lat_rad': lat_rad,
                'long_deg': lon_deg,
                'long_rad': lon_rad,
                'alt_km': alt_km,
                'xyz': np.array([x, y, z])
            }
    return sat_positions_dict


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
        for j in range(i + 1, num_satellites):
            if feasible_mask[i, j]:
                if distance_matrix is not None:
                    dist = distance_matrix[i, j]

                valid_isls.append({
                    'sat_1': i,
                    'sat_2': j,
                    'dist_km': dist
                })
    return valid_isls


def find_motif_possibilities_func(valid_isls, sat_positions, NUM_ORBITS, NUM_SATS_PER_ORBIT):
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


def add_motif_links_to_graph(grph, motif, sat_positions, NUM_ORBITS, NUM_SATS_PER_ORBIT, feasibleMask):
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
        is_possible = util.check_edge_availability(grph, i, sel_sat_id) and feasibleMask[i,sel_sat_id]

        if is_possible:
            dist = np.linalg.norm(sat_positions[i]['xyz'] - sat_positions[sel_sat_id]['xyz'])
            #dist = util.compute_isl_length(i, sel_sat_id, sat_positions)
            grph.add_edge(i, sel_sat_id, length=dist)
        sel_sat_id = util.get_neighbor_satellite(sat_positions[i]["orb_id"], sat_positions[i]["orb_sat_id"],
                                                 motif["sat_2_orb_offset"], motif["sat_2_sat_offset"], sat_positions,
                                                 NUM_ORBITS, NUM_SATS_PER_ORBIT)
        is_possible = util.check_edge_availability(grph, i, sel_sat_id) and feasibleMask[i,sel_sat_id]
        if is_possible:
            dist = np.linalg.norm(sat_positions[i]['xyz'] - sat_positions[sel_sat_id]['xyz'])
            #dist = util.compute_isl_length(i, sel_sat_id, sat_positions)
            grph.add_edge(i, sel_sat_id, length=dist)
    return grph


def compute_metric_avoid_city(grph, city_pairs, city_coverage, flow_pops):
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

            hops = len(path) - 1
            stretch = distance / geoDist

            weight = flow_pops[i] / 10000000

            weightSum += weight
            weightedStretchSum += stretch * weight

            #weightedHopCountSum += hops * weight
            weightedHopCountSum += hops * flow_pops[i]

            weightedLatency += distance * flow_pops[i] / (3e8)

            util.remove_coverage_for_city(grph, city1, city_coverage)
            util.remove_coverage_for_city(grph, city2, city_coverage)
        except Exception as e:
            util.remove_coverage_for_city(grph, city1, city_coverage)
            util.remove_coverage_for_city(grph, city2, city_coverage)
            return_val = {
                "avgWeightedStretch": 99999.0,
                "avgWeightedHopCount": 99999.0,
                "wMetric": 99999.0,
                "weightedLatency": np.inf,
                "weightedHopCountSum": np.inf,
                "linkedGraph" : grph
            }
            return return_val
    avgWeightedStretch = weightedStretchSum / weightSum
    avgWeightedHopCount = weightedHopCountSum / weightSum
    wMetric = avgWeightedStretch + avgWeightedHopCount

    b = datetime.datetime.now() - a
    # print("time to compute metric:", b.seconds, ", avgWeightedStretch:", avgWeightedStretch, ", avgWeightedHopCount:",
    #       avgWeightedHopCount)

    return_val = {
        "avgWeightedStretch": avgWeightedStretch,
        "avgWeightedHopCount": avgWeightedHopCount,
        "wMetric": wMetric,
        "weightedLatency": weightedLatency,
        "weightedHopCountSum": weightedHopCountSum,
        "linkedGraph" : grph
    }
    return return_val


def run_motif_analysis(grph, motif_cnt, motif, return_dict, sat_positions, NUM_ORBITS, NUM_SATS_PER_ORBIT, city_pairs,
                       city_coverage, flow_pops, feasibleMask):
    """
    Runs motif analysis for individual motifs
    :param grph: The graph under consideration
    :param motif_cnt: Motif counter
    :param motif: Motif
    :param return_dict: The return values
    """

    grph = add_motif_links_to_graph(grph, motif, sat_positions, NUM_ORBITS, NUM_SATS_PER_ORBIT, feasibleMask)
    retVal = compute_metric_avoid_city(grph, city_pairs, city_coverage, flow_pops)
    motif["wMetric"] = retVal["wMetric"]
    motif["wStretch"] = retVal["avgWeightedStretch"]
    motif["wHop"] = retVal["avgWeightedHopCount"]
    motif["weightedLatency"] = retVal["weightedLatency"]
    motif["weightedHopCountSum"] = retVal["weightedHopCountSum"]
    motif["linkedGraph"] = retVal["linkedGraph"]


    return_dict[motif_cnt] = motif

def calculate_min_metric(
        CORE_CNT: int,
        numSatellites: int = 200,
        NUM_ORBITS: int = 20,
        inclination: int = 80,
        phasingParameter: int = 5,
        orbitRadius: float = 6.946e6,
        EARTH_MEAN_RADIUS: float = 6371.0e3,
        src_inds: [] = [],
        dst_inds: [] = [],
        demandVals: [] = [],
        multiProcessed = True,
        minMetric = "latency",
        feasibleMask = []
    ) -> float:
    """
    Calculates the weighted latency for a satellite constellation based on various parameters.

    Args:
        CORE_CNT (int): Number of cores to use for parallel processing.
        numSatellites (int): Total number of satellites in the constellation.
        NUM_ORBITS (int): Number of orbital planes.
        inclination (int): Inclination of the orbits in degrees.
        phasingParameter (int): Phasing parameter for the Walker Star constellation.
        orbitRadius (float): Radius of the satellite orbits in meters.
        EARTH_MEAN_RADIUS (float): Mean radius of the Earth in kilometers.
        numFlows (int): Number of city pairs (flows) to simulate.

    Returns:
        float: The weighted latency of the best motif found.
    """

    NUM_SATS_PER_ORBIT = numSatellites // NUM_ORBITS

    # Generate constellation and distance components 
    positions, vecs = myUtils.generateWalkerStarConstellationPoints(numSatellites,
                                                                    inclination,
                                                                    NUM_ORBITS,
                                                                    phasingParameter,
                                                                    orbitRadius)
    sat_positions = convert_positions_to_sat_dict(positions, EARTH_MEAN_RADIUS)
    shapedPositions = np.reshape(positions, [np.shape(positions)[0] * np.shape(positions)[1], np.shape(positions)[2]])
    feasibleMask = myUtils.check_los(shapedPositions, shapedPositions)
    dmat = cdist(shapedPositions, shapedPositions) 

    #generate possible topology configurations 
    valid_isls = convert_feasible_mask_to_valid_isls(feasibleMask, positions, dmat)
    valid_motif_possibilities = find_motif_possibilities_func(valid_isls, sat_positions, NUM_ORBITS,
                                                              NUM_SATS_PER_ORBIT)

    #format demand properly 
    greatCircleCross = myUtils.great_circle_distance_matrix_cartesian(shapedPositions, orbitRadius)
    city_pairs = {i: {'city_1': src, 'city_2': dst, 'geo_dist': greatCircleCross[src, dst].item()}
                  for i, (src, dst) in enumerate(zip(src_inds, dst_inds))}
    city_positions = shapedPositions
    flow_pops = demandVals
    
    city_coverage = [
        {'city': i, 'sat': i, 'dist': 0.0}
        for i in range(numSatellites)
    ]

    G = nx.Graph()

    for i in range(len(city_positions)):
        G.add_node(city_coverage[i]["city"], pos = city_positions[i])

    manager = Manager()
    return_dict = manager.dict()
    threads = []

    if(multiProcessed):
        for i in range(0, len(valid_motif_possibilities), CORE_CNT):
            for j in range(0, CORE_CNT):
                idx = i + j
                if idx < len(valid_motif_possibilities):
                    # print(f"Generating graph for motif: {valid_motif_possibilities[idx]['sat_1_id']}, "
                    #       f"{valid_motif_possibilities[idx]['sat_2_id']}")
                    p = Process(target=run_motif_analysis,
                                args=(G.copy(), idx, valid_motif_possibilities[idx], return_dict, sat_positions,
                                    NUM_ORBITS, NUM_SATS_PER_ORBIT, city_pairs, city_coverage, flow_pops, feasibleMask))
                    threads.append(p)
                    p.start()

            for p in threads:
                p.join()
            threads = [] # Clear threads for the next batch
    else:
        for i in range(0, len(valid_motif_possibilities)):
            # print(f"Generating graph for motif: {valid_motif_possibilities[i]['sat_1_id']}, "
            #       f"{valid_motif_possibilities[i]['sat_2_id']}")
            run_motif_analysis(G.copy(), i, valid_motif_possibilities[i], return_dict, sat_positions, NUM_ORBITS,
                               NUM_SATS_PER_ORBIT, city_pairs, city_coverage, flow_pops, feasibleMask)

    for value in return_dict.values():
        valid_motif_possibilities[value["motif_cnt"]]["wMetric"] = value["wMetric"]
        valid_motif_possibilities[value["motif_cnt"]]["wStretch"] = value["wStretch"]
        valid_motif_possibilities[value["motif_cnt"]]["wHop"] = value["wHop"]
        valid_motif_possibilities[value["motif_cnt"]]["weightedLatency"] = value["weightedLatency"]
        valid_motif_possibilities[value["motif_cnt"]]["linkedGraph"] = value["linkedGraph"]
        valid_motif_possibilities[value["motif_cnt"]]["weightedHopCountSum"] = value["weightedHopCountSum"]
    
    #min_motif_dict = min(valid_motif_possibilities.values(), key=lambda x: x['weightedLatency'])

    if(minMetric == "latency"):
        min_motif_dict = min(valid_motif_possibilities.values(), key=lambda x: x['weightedLatency'])
        [x['weightedLatency'] for x in valid_motif_possibilities.values()]
        return min_motif_dict['weightedLatency'], min_motif_dict['linkedGraph']

    if(minMetric == "hops"):
        min_motif_dict = min(valid_motif_possibilities.values(), key=lambda x: x['weightedHopCountSum'])
        #weightedHopCountSum = min_motif_dict['weightedHopCountSum']

        return min_motif_dict['weightedHopCountSum'], min_motif_dict['linkedGraph']
    
    return None
    
    best_motif = util.get_best_motif_at_level(valid_motif_possibilities)

    return best_motif['weightedLatency']


if __name__ == '__main__':
    # Example usage of the functionalized program
    # You would typically get CORE_CNT from sys.argv
    try:
        CORE_CNT_INPUT = int(sys.argv[1])
    except IndexError:
        print("Please provide CORE_CNT as a command-line argument.")
        sys.exit(1)

    weighted_latency_result = calculate_weighted_latency(CORE_CNT=CORE_CNT_INPUT)
    print(f"\nWeighted Latency of the best motif: {weighted_latency_result}")