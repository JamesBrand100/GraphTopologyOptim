import heapq
import sys
from collections import defaultdict
import random
import pdb 
import numpy as np

class Packet:
    def __init__(self, pid, src, dest, start_time, size):
        self.pid = pid
        self.src = src
        self.dest = dest
        self.start_time = start_time
        self.size = size  # in bits
        self.path = []
        self.current_node = src
        self.queue_times = []

class Event:
    def __init__(self, time, event_type, packet, node=None):
        self.time = time
        self.type = event_type
        self.packet = packet
        self.node = node  # For node-specific events
        
    def __lt__(self, other):
        return self.time < other.time

class NetworkSimulator:
    def __init__(self, nodes, topology, bandwidths):
        self.nodes = nodes
        self.topology = topology
        self.bandwidths = bandwidths
        self.routing_table = self.build_routing_table()
        
    def build_routing_table(self):
        # Initialize distance and next hop matrices
        num_nodes = len(self.nodes)
        node_index = {n: i for i, n in enumerate(self.nodes)}
        dist = [[sys.maxsize] * num_nodes for _ in range(num_nodes)]
        next_hop = [[-1] * num_nodes for _ in range(num_nodes)]
        
        # Initialize distances
        for u in self.nodes:
            for v in self.topology[u]:
                i = node_index[u]
                j = node_index[v]
                bw = self.bandwidths[u][v]
                dist[i][j] = 1 / bw  # Use inverse bandwidth as weight
                next_hop[i][j] = v
            dist[i][i] = 0
            
        # Floyd-Warshall algorithm
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_hop[i][j] = next_hop[i][k]
                        
        # Build routing table dictionary
        routing_table = np.ones([num_nodes, num_nodes], dtype = int)
        for u in self.nodes:
            i = node_index[u]
            for v in self.nodes:
                j = node_index[v]
                routing_table[u][v] = next_hop[i][j]
                
        return routing_table

def getFlowDelays(nodes, topology, bandwidths, traffic_set, sim):

    event_queue = []
    active_links = defaultdict(float)  # (u, v) -> next_available_time
    packet_records = []
    
    # Create initial events
    for pid, (src, dest, start_time, size) in enumerate(traffic_set):
        packet = Packet(pid, src, dest, start_time, size)
        heapq.heappush(event_queue, Event(start_time, 'packet_start', packet))
    
    while event_queue:
        event = heapq.heappop(event_queue)
        
        if event.type == 'packet_start':
            packet = event.packet
            if packet.current_node == packet.dest:
                continue
                
            next_node = sim.routing_table[packet.current_node][packet.dest]
            packet.path.append(packet.current_node)
            
            # Calculate transmission time
            bw = sim.bandwidths[packet.current_node][next_node]
            tx_time = packet.size / bw
            
            # Schedule arrival event
            arrival_time = event.time + tx_time
            heapq.heappush(event_queue, 
                Event(arrival_time, 'packet_arrive', packet, next_node))
            
        elif event.type == 'packet_arrive':
            packet = event.packet
            current_node = event.node
            packet.current_node = current_node
            
            if current_node == packet.dest:
                # Record completed packet
                packet_records.append({
                    'pid': packet.pid,
                    'total_delay': event.time - packet.start_time,
                    'path': packet.path + [current_node],
                    'queue_times': sum(packet.queue_times),
                    'packet size':  packet.size,
                })
                continue
                
            # Check if link is busy
            next_node = sim.routing_table[current_node][packet.dest]
            link = (current_node, next_node)
            
            # Calculate transmission time
            bw = sim.bandwidths[current_node][next_node]
            tx_time = packet.size / bw
            
            # Determine actual transmission start time
            if event.time >= active_links[link]:
                start_time = event.time
                queue_delay = 0
            else:
                queue_delay = active_links[link] - event.time
                start_time = active_links[link]
                
            # Update link availability and packet queue times
            active_links[link] = start_time + tx_time
            packet.queue_times.append(queue_delay)
            
            # Schedule next hop arrival
            heapq.heappush(event_queue,
                Event(start_time + tx_time, 'packet_arrive', packet, next_node))
    
    return packet_records
def generate_network_topology(
    num_nodes,
    links_per_node,
    max_bandwidth_per_node  # in Mbps
):
    """Generates a random network topology with bandwidth assignments
    
    Args:
        num_nodes: Number of nodes in the network
        links_per_node: Minimum number of links per node
        max_bandwidth_per_node: Maximum total bandwidth per node in Mbps
    
    Returns:
        Dictionary containing:
        - nodes: List of node IDs
        - topology: Adjacency list of connections
        - bandwidths: Bandwidth matrix in bps
    """
    nodes = list(range(num_nodes))
    topology = defaultdict(list)
    bandwidths = defaultdict(dict)
    
    # Create minimum spanning tree for connectivity
    unconnected = nodes[1:]
    connected = [nodes[0]]
    
    while unconnected:
        node_a = random.choice(connected)
        node_b = random.choice(unconnected)
        topology[node_a].append(node_b)
        topology[node_b].append(node_a)
        connected.append(node_b)
        unconnected.remove(node_b)
    
    # Add random links to reach desired connectivity
    for node in nodes:
        while len(topology[node]) < links_per_node:
            available = [n for n in nodes 
                        if n != node 
                        and n not in topology[node]
                        and len(topology[n]) < links_per_node*2]
            if not available:
                break
            neighbor = random.choice(available)
            topology[node].append(neighbor)
            topology[neighbor].append(node)
    
    # Assign bandwidths (symmetrical links)
    for node in nodes:
        total_bw = random.uniform(max_bandwidth_per_node/2, max_bandwidth_per_node)
        num_links = max(1, len(topology[node]))
        per_link_bw = total_bw / num_links
        
        for neighbor in topology[node]:
            if neighbor not in bandwidths[node]:
                bw = per_link_bw * 1e6  # Convert Mbps to bps
                bandwidths[node][neighbor] = bw
                bandwidths[neighbor][node] = bw
    
    return {
        'nodes': nodes,
        'topology': dict(topology),
        'bandwidths': dict(bandwidths)
    }

def generate_network_traffic(
    nodes,
    num_flows,
    time_frame,  # in seconds
    max_flow_duration=1.0  # in seconds 
):
    """Generates random traffic flows for a network
    
    Args:
        nodes: List of node IDs
        num_flows: Number of flows to generate
        time_frame: Simulation time window in seconds
        avg_flow_rate: Average flow rate in Mbps
    
    Returns:
        List of traffic flows as (src, dest, start_time, size_in_bits)
    """
    traffic_set = []
    
    for _ in range(num_flows):
        src, dest = random.sample(nodes, 2)
        start_time = random.uniform(0, time_frame)
        duration = random.uniform(0, max_flow_duration)
        size = 1e6 * duration  # bits, assuming 
        
        traffic_set.append((src, dest, start_time, int(size)))
    
    return traffic_set

def generate_network_traffic_new(
    nodes,
    time_frame,  # in seconds
    packet_size=1e6 * 3,  # default: 3 Mbs by default 
    rate_matrix=None,
    min_rate=10,  # packets/sec
    max_rate=1000  # packets/sec
):
    """Generates network traffic using Poisson processes with random rates
    
    Args:
        nodes: List of node IDs
        time_frame: Simulation time window in seconds
        packet_size: Size of each packet in bits
        rate_matrix: Optional pre-existing rate matrix (dict of dicts or 2D array)
        min_rate: Minimum packet rate (packets/sec) for random rates
        max_rate: Maximum packet rate (packets/sec) for random rates
    
    Returns:
        traffic_set: List of packets as (src, dest, timestamp, size_in_bits)
        rate_matrix: The rate matrix used (either provided or generated)
    """
    # Convert nodes to list if needed
    nodes = list(nodes)
    num_nodes = len(nodes)
    
    # Generate random rate matrix if not provided
    if rate_matrix is None:
        if isinstance(nodes[0], tuple):
            # If nodes are tuples, use just the IDs for the rate matrix
            node_ids = [n[0] for n in nodes]
        else:
            node_ids = nodes
            
        rate_matrix = defaultdict(dict)
        for src in node_ids:
            for dest in node_ids:
                if src == dest:
                    rate_matrix[src][dest] = 0  # no self-traffic
                else:
                    rate_matrix[src][dest] = random.uniform(min_rate, max_rate)
    
    traffic_set = []
    
    # Generate packets for each src-dest pair
    for src in nodes:
        src_id = src[0] if isinstance(src, tuple) else src
        for dest in nodes:
            dest_id = dest[0] if isinstance(dest, tuple) else dest
            
            if src_id == dest_id:
                continue  # skip self-traffic
                
            rate = rate_matrix[src_id][dest_id]
            if rate <= 0:
                continue  # skip pairs with zero rate
                
            # Generate Poisson process events
            avg_packets = rate * time_frame
            num_events = np.random.poisson(avg_packets)

            if num_events == 0:
                continue
                
            # Generate random timestamps
            timestamps = np.sort(np.random.uniform(0, time_frame, num_events))
            
            # Add packets to traffic set
            for ts in timestamps:
                traffic_set.append((src, dest, float(ts), packet_size))
    
    # Sort traffic by timestamp
    traffic_set.sort(key=lambda x: x[2])
    
    return traffic_set, rate_matrix

# Test Case 1: Over-queued (bottleneck link)
def test_over_queued():
    nodes = [0, 1, 2]
    topology = {
        0: [1],
        1: [0, 2],
        2: [1]
    }
    bandwidths = {
        0: {1: 1e6},    # 1 Mbps
        1: {0: 1e6, 2: 1e6},
        2: {1: 1e6}
    }
    
    # 3 packets with 0.0001s spacing (10x faster than bottleneck)
    traffic_set = [
        (0, 2, 0.0, 1500*8),
        (0, 2, 0.0001, 1500*8),
        (0, 2, 0.0002, 1500*8)
    ]
    
    results = getFlowDelays(nodes, topology, bandwidths, traffic_set)
    for res in results:
        print(f"Packet {res['pid']}: {res['total_delay']*1000:.2f}ms")

# Test Case 2: Perfectly queued
def test_perfect_queued():
    nodes = [0, 1, 2]
    topology = {
        0: [1],
        1: [0, 2],
        2: [1]
    }
    bandwidths = {
        0: {1: 1e6},
        1: {0: 1e6, 2: 1e6},
        2: {1: 1e6}
    }
    
    # Packet spacing = 1500*8 / 1e6 = 0.012s
    traffic_set = [
        (0, 2, 0.0, 1500*8),
        (0, 2, 0.012, 1500*8),
        (0, 2, 0.024, 1500*8)
    ]
    
    results = getFlowDelays(nodes, topology, bandwidths, traffic_set)
    for res in results:
        print(f"Packet {res['pid']}: {res['total_delay']*1000:.2f}ms")

# Test Case 3: Under-queued
def test_under_queued():
    nodes = [0, 1, 2]
    topology = {
        0: [1],
        1: [0, 2],
        2: [1]
    }
    bandwidths = {
        0: {1: 1e6},
        1: {0: 1e6, 2: 1e6},
        2: {1: 1e6}
    }
    
    # Packet spacing > 0.012s
    traffic_set = [
        (0, 2, 0.0, 1500*8),
        (0, 2, 0.013, 1500*8),
        (0, 2, 0.026, 1500*8)
    ]

    # Combined configuration
    config = {
        'nodes': network['nodes'],
        'topology': network['topology'],
        'bandwidths': network['bandwidths'],
        'traffic_set': traffic
    }

    #now, generate simulator
    sim = NetworkSimulator(network['nodes'], network['topology'], network['bandwidths'])
    
    results = getFlowDelays(nodes, topology, bandwidths, traffic_set, sim)
    for res in results:
        print(f"Packet {res['pid']}: {res['total_delay']*1000:.2f}ms")

if __name__ == "__main__":
    # print("Over-queued Test:")
    # test_over_queued()
    
    # print("\nPerfectly queued Test:")
    # test_perfect_queued()
    
    # print("\nUnder-queued Test:")
    # test_under_queued()

    # #then, generate network over the set of nodes randomly (no given rates)
    # traffic_old = generate_network_traffic(
    #     nodes=network['nodes'],
    #     num_flows=10000,
    #     time_frame=1,  # 10 second simulation
    #     max_flow_duration=10 # 10 Mbps average
    # )

    # Example usage:
    #first, generate random topology that satisfies spanning attribute
    #this is the BW of links/ connectivity 
    network = generate_network_topology(
        num_nodes=10,
        links_per_node=3,
        max_bandwidth_per_node=1000  # 1 Gbps
    )

    #this is the function for generating time locations of packets 
    traffic, rate_matrix = generate_network_traffic_new(
        nodes=network['nodes'],
        time_frame=1, 
    )
    
    # Combined configuration
    config = {
        'nodes': network['nodes'],
        'topology': network['topology'],
        'bandwidths': network['bandwidths'],
        'traffic_set': traffic
    }

    #now, generate simulator
    sim = NetworkSimulator(network['nodes'], network['topology'], network['bandwidths'])

    #and then get routing information
    routingTable = sim.routing_table

    #To use with previous simulator:
    results = getFlowDelays(
        config['nodes'],
        config['topology'],
        config['bandwidths'],
        config['traffic_set'],
        sim
    )

    #get delays 
    delays = [res['total_delay'] for res in results]
    #get sizes 
    sizes = [res['packet size'] for res in results]  # Fixed typo: 'packet' not 'packet'
    #then get avgDelay or 1/throughput 
    avgDelay = sum( size * delay for size, delay in zip(sizes, delays)) / sum(sizes)  # Removed comma and fixed calculation
    
    print(avgDelay)
    
    #now, we create network to predict the throughput off this config, so utilizing 
    #nodes, topology, bws, traffic....

    #should use flows as well....so predefined routing tables with floyd warshall into our algorithm.....
    #possibly difficult as....outside the pipeline in general....

    #im slightly confused as to how 

