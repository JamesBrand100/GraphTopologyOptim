import numpy as np
import heapq
from collections import defaultdict
import random

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
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix  # shape: (N, N), ∞ where no link
        self.num_nodes = len(adj_matrix)
        self.routing_table = self.build_routing_table()

    def build_routing_table(self):
        N = self.num_nodes
        dist = self.adj_matrix.copy()
        next_hop = np.full((N, N), -1, dtype=int)

        for i in range(N):
            for j in range(N):
                if i != j and not np.isinf(dist[i][j]):
                    next_hop[i][j] = j
                elif i == j:
                    dist[i][j] = 0

        # Floyd-Warshall
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_hop[i][j] = next_hop[i][k]

        return next_hop

def getFlowDelays(adj_matrix, traffic_set, sim):
    #create event queue, 
    event_queue = []
    #not sure what this is used for 
    active_links = defaultdict(float)
    #create record set for packets 
    packet_records = []

    #create events for starting transmit of packets 
    for pid, (src, dest, start_time, size) in enumerate(traffic_set):
        packet = Packet(pid, src, dest, start_time, size)
        heapq.heappush(event_queue, Event(start_time, 'packet_start', packet))

    #while there are still events to process 
    while event_queue:
        #get the nearest one 
        event = heapq.heappop(event_queue)
        #get the packet from event 
        packet = event.packet

        #if its the start of journey 
        if event.type == 'packet_start':
            if packet.current_node == packet.dest:
                continue

            #look for next hop 
            next_node = sim.routing_table[packet.current_node][packet.dest]
            packet.path.append(packet.current_node)

            #get info for knowing when it arrives 
            bw_inv = sim.adj_matrix[packet.current_node][next_node]
            tx_time = packet.size * bw_inv

            heapq.heappush(event_queue,
                Event(event.time + tx_time, 'packet_arrive', packet, next_node))

        #if packet arrive event 
        elif event.type == 'packet_arrive':
            current_node = event.node
            packet.current_node = current_node

            #finish event if processed 
            if current_node == packet.dest:
                packet_records.append({
                    'pid': packet.pid,
                    'total_delay': event.time - packet.start_time,
                    'path': packet.path + [current_node],
                    'queue_times': sum(packet.queue_times),
                    'packet_size': packet.size,
                })
                continue
            
            #look for next hop and create used link 
            next_node = sim.routing_table[current_node][packet.dest]
            link = (current_node, next_node)

            #use queried BW 
            bw_inv = sim.adj_matrix[current_node][next_node]
            tx_time = packet.size * bw_inv

            #if our start time of transmit is greater than ending usage of that active link, assign it w/out delay
            if event.time >= active_links[link]:
                start_time = event.time
                queue_delay = 0
            #if we encounter queuing issues, then: 
            else:
                #q delay is the end of usage - the time we would start originally 
                queue_delay = active_links[link] - event.time
                start_time = active_links[link]

            active_links[link] = start_time + tx_time
            packet.queue_times.append(queue_delay)

            heapq.heappush(event_queue,
                Event(start_time + tx_time, 'packet_arrive', packet, next_node))

    return packet_records

def getFlowDelaysWithStates(adj_matrix, traffic_set, sim, n, frame_interval=1.0):

    #set up vars including:
    #event q, links used, records of transmitted packets
    event_queue = []
    active_links = defaultdict(float)
    packet_records = []

    num_nodes = adj_matrix.shape[0]
    queue_matrix = np.zeros((num_nodes, num_nodes))
    frames = []

    next_frame_time = 0
    total_sim_time = 0

    #create packet start events 
    for pid, (src, dest, start_time, size) in enumerate(traffic_set):
        packet = Packet(pid, src, dest, start_time, size)
        heapq.heappush(event_queue, Event(start_time, 'packet_start', packet))
        total_sim_time = max(total_sim_time, start_time)

    #while we have events to process 
    while event_queue and len(frames) < n:
        event = heapq.heappop(event_queue)
        packet = event.packet
        current_time = event.time
        total_sim_time = max(total_sim_time, current_time)

        # Save frame if needed
        while current_time >= next_frame_time and len(frames) < n:
            frames.append(queue_matrix.copy())
            next_frame_time += frame_interval

        if event.type == 'packet_start':
            if packet.current_node == packet.dest:
                continue

            next_node = sim.routing_table[packet.current_node][packet.dest]
            packet.path.append(packet.current_node)

            bw_inv = sim.adj_matrix[packet.current_node][next_node]
            tx_time = packet.size * bw_inv

            heapq.heappush(event_queue,
                Event(current_time + tx_time, 'packet_arrive', packet, next_node))

        elif event.type == 'packet_arrive':
            current_node = event.node
            packet.current_node = current_node

            if current_node == packet.dest:
                packet_records.append({
                    'pid': packet.pid,
                    'total_delay': current_time - packet.start_time,
                    'path': packet.path + [current_node],
                    'queue_times': sum(packet.queue_times),
                    'packet_size': packet.size,
                })
                continue

            next_node = sim.routing_table[current_node][packet.dest]
            link = (current_node, next_node)

            bw_inv = sim.adj_matrix[current_node][next_node]
            tx_time = packet.size * bw_inv

            if current_time >= active_links[link]:
                start_time = current_time
                queue_delay = 0
            else:
                queue_delay = active_links[link] - current_time
                start_time = active_links[link]

            # Update the queue matrix
            queue_matrix[current_node][next_node] += packet.size

            # Schedule the packet to move after transmission
            active_links[link] = start_time + tx_time
            packet.queue_times.append(queue_delay)

            # Hacky callback using lambda to simulate end-of-tx queue update
            heapq.heappush(event_queue, Event(start_time + tx_time, 'dequeue_callback',
                                              packet, next_node))
            heapq.heappush(event_queue,
                Event(start_time + tx_time, 'packet_arrive', packet, next_node))

        elif event.type == 'dequeue_callback':
            # This special event handles queue reduction after transmission
            src = packet.path[-1]
            dst = event.node
            queue_matrix[src][dst] -= packet.size

    return packet_records, frames

def generate_adjacency_matrix(num_nodes, links_per_node, max_bandwidth):
    #initially create no-link adjacency matrix 
    adj = np.full((num_nodes, num_nodes), 1e6)

    #iterate through the node nums 
    for i in range(num_nodes):
        #get a number of neighbors 
        neighbors = random.sample([j for j in range(num_nodes) if j != i], links_per_node)
        for j in neighbors:
            #randomly sample bandwidth 
            bw = random.uniform(1, max_bandwidth / links_per_node) * 1e6  # in bps
            adj[i][j] = adj[j][i] = 1 / bw
    return adj

def generate_network_traffic_new(
    num_nodes,
    time_frame,  # in seconds
    packet_size=1e6 * 3,  # default: 3 Mbs by default 
    rate_matrix=None,
    min_rate=50,  # packets/sec
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
    nodes = [*range(num_nodes)]
    
    # Generate random rate matrix if not provided
    if rate_matrix is None:

        rate_matrix = np.zeros([num_nodes,num_nodes])
        for src in nodes:
            for dest in nodes:
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
            #timestamps = np.sort(np.random.uniform(0, time_frame, num_events))
            timestamps = np.sort(np.linspace(0, time_frame, num_events))
            
            # Add packets to traffic set
            for ts in timestamps:
                traffic_set.append((src, dest, float(ts), packet_size))
    
    # Sort traffic by timestamp
    traffic_set.sort(key=lambda x: x[2])
    
    return traffic_set, rate_matrix

# --------- Test Code ---------
if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)

    #create storage for nodes, links, max BW 
    num_nodes = 5
    links_per_node = 2
    max_bw = 100  # Mbps

    #create storage for BW and connectivity 
    adj_matrix = generate_adjacency_matrix(num_nodes, links_per_node, max_bw)

    #create simulator from adj_matrix, store routing table  
    sim = NetworkSimulator(adj_matrix)
    routingTable = sim.routing_table

    #store traffic and associated rate matrix 
    traffic_set, rate_matrix = generate_network_traffic_new(num_nodes, 1) 

    #get the flow delays for these components 
    records = getFlowDelays(adj_matrix, traffic_set, sim)

    #get delays 
    delays = [res['total_delay'] for res in records]
    #get sizes 
    sizes = [res['packet_size'] for res in records]  # Fixed typo: 'packet' not 'packet'
    #then get avgDelay or 1/throughput 
    avgDelay = sum( size * delay for size, delay in zip(sizes, delays)) / sum(sizes)  # Removed comma and fixed calculation
    
    print(avgDelay)
    