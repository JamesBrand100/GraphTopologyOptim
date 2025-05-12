import heapq

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

def getFlowDelays(nodes, topology, bandwidth_matrix, traffic_set, routing_policy):
    # Simulation state
    event_queue = []
    active_links = {}  # (node1, node2) -> end_time
    node_queues = {node: [] for node in nodes}
    packet_records = []
    
    # Initialize events for packet starts
    for pid, (src, dest, start_time, size) in enumerate(traffic_set):
        packet = Packet(pid, src, dest, start_time, size)
        heapq.heappush(event_queue, Event(start_time, 'packet_start', packet))
    
    while event_queue:
        event = heapq.heappop(event_queue)
        
        if event.type == 'packet_start':
            # Start packet journey
            packet = event.packet
            next_hop = routing_policy(packet.current_node, packet.dest)
            packet.path.append(packet.current_node)
            
            # Schedule arrival at first hop
            transmission_time = calculate_transmission_time(
                packet.size,
                bandwidth_matrix[packet.current_node][next_hop]
            )
            
            heapq.heappush(event_queue, Event(
                event.time + transmission_time,
                'packet_arrival',
                packet,
                next_hop
            ))
            
        elif event.type == 'packet_arrival':
            packet = event.packet
            current_node = event.node
            
            if current_node == packet.dest:
                # Record final delay
                packet_records.append({
                    'pid': packet.pid,
                    'total_delay': event.time - packet.start_time,
                    'path': packet.path
                })
                continue
                
            # Determine next hop using routing policy
            next_hop = routing_policy(current_node, packet.dest)
            packet.current_node = current_node
            packet.path.append(current_node)
            
            # Calculate transmission time
            bandwidth = bandwidth_matrix[current_node][next_hop]
            transmission_time = packet.size / bandwidth
            
            # Check link availability
            link = (current_node, next_hop)
            current_link_end = active_links.get(link, 0)
            
            if event.time >= current_link_end:
                # Link is free, transmit immediately
                start_time = event.time
                end_time = start_time + transmission_time
                active_links[link] = end_time
                packet.queue_times.append(0)
            else:
                # Link is busy, add to queue
                queue_delay = current_link_end - event.time
                start_time = current_link_end
                end_time = start_time + transmission_time
                active_links[link] = end_time
                packet.queue_times.append(queue_delay)
                node_queues[current_node].append(packet)
            
            # Schedule next hop arrival
            heapq.heappush(event_queue, Event(
                end_time,
                'packet_arrival',
                packet,
                next_hop
            ))
    
    return packet_records

def calculate_transmission_time(packet_size, bandwidth):
    return packet_size / bandwidth  # Simplified model

# Example usage:
if __name__ == "__main__":
    # Sample input data
    nodes = ['A', 'B', 'C', 'D']
    topology = {
        'A': ['B'],
        'B': ['A', 'C'],
        'C': ['B', 'D'],
        'D': ['C']
    }
    bandwidth_matrix = {
        'A': {'B': 1e6},  # 1 Mbps
        'B': {'A': 1e6, 'C': 2e6},
        'C': {'B': 2e6, 'D': 1e6},
        'D': {'C': 1e6}
    }
    
    def simple_routing(current_node, dest):
        # Simplified shortest path routing
        routes = {
            'A': {'D': 'B'},
            'B': {'D': 'C'},
            'C': {'D': 'D'},
            'D': {'D': 'D'}
        }
        return routes[current_node][dest]
    
    traffic_set = [
        ('A', 'D', 0, 1500*8),  # 1500 byte packet
        ('A', 'D', 0.1, 1500*8)
    ]
    
    results = getFlowDelays(nodes, topology, bandwidth_matrix, traffic_set, simple_routing)
    
    for result in results:
        print(f"Packet {result['pid']}:")
        print(f"  Total delay: {result['total_delay']:.6f}s")
        print(f"  Path: {result['path']}")