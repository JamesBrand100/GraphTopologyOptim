import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import newPacketSim

class FlowDelayGNN(nn.Module):
    def __init__(self, 
                 node_dim=32, 
                 link_dim=32, 
                 flow_dim=64,
                 n_iterations=8):
        super().__init__()
        #iterations defines how many times you pass information to your neighbors
        #allows you to propagate info thorugh
        #slightly similar to # hidden states in RNNs (as this uses GRUs)
        #could set it to graph's diameter for reliable info transfer 
        self.n_iterations = n_iterations
        
        # Initial Embeddings
        # so the dimensionality of compression of data / items 
        # with regards to each component of the data 
        self.node_embed = nn.Linear(2, node_dim)  # [bandwidth, is_router]
        self.link_embed = nn.Linear(2, link_dim)  # [capacity, load]
        self.flow_embed = nn.Linear(3, flow_dim)  # [traffic, packets, packet_size]
        
        # Update Networks
        # so the components used to actually process information 
        self.flow_update = nn.GRU(link_dim, flow_dim)
        self.link_update = nn.GRUCell(flow_dim, link_dim)
        self.node_update = nn.GRUCell(link_dim, node_dim)
        
        # Readout
        # then, this is the component used for defining how we combine information to get last prediction 
        self.readout = nn.Sequential(
            nn.Linear(flow_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, graph_data):
        """Simplified forward pass
        Args:
            graph_data: Dict containing:
                - nodes: Tensor of shape [num_nodes, 2] (bandwidth, node_type)
                - links: Tensor of shape [num_links, 2] (capacity, connected_nodes)
                - flows: Tensor of shape [num_flows, 3] (traffic, packets, pkt_size)
                - link_to_flow: Adjacency matrix [num_links, num_flows]
                - flow_to_link: Adjacency matrix [num_flows, num_links]
        """
        # Initialize states from embeddings 
        node_state = F.relu(self.node_embed(graph_data['nodes']))
        link_state = F.relu(self.link_embed(graph_data['links']))
        flow_state = F.relu(self.flow_embed(graph_data['flows']))
        
        # Message Passing for items within graph 
        for _ in range(self.n_iterations):
            # Flows aggregate link states
            link_messages = torch.matmul(graph_data['link_to_flow'], link_state)
            _, flow_state = self.flow_update(link_messages.unsqueeze(0), flow_state.unsqueeze(0))
            flow_state = flow_state.squeeze(0)
            
            # Links aggregate flow states
            flow_messages = torch.matmul(graph_data['flow_to_link'], flow_state)
            link_state = self.link_update(flow_messages, link_state)
            
            # Nodes aggregate connected links
            link_agg = torch.matmul(graph_data['node_link_adj'], link_state)
            node_state = self.node_update(link_agg, node_state)
        
        # Predict delays from readout method
        delay = self.readout(flow_state)

        #squeeze to reduce dimensionality 
        return delay.squeeze()

#create model 
model = FlowDelayGNN() 

#num train iterations
numTrainIters = 100

#begin training loop
for ind in range(numTrainIters): 
    #first, generate data 
    # Example usage:
    network = newPacketSim.generate_network_topology(
        num_nodes=10,
        min_links_per_node=3,
        max_bandwidth_per_node=1000  # 1 Gbps
    )
    traffic = newPacketSim.generate_network_traffic(
        nodes=network['nodes'],
        num_flows=10000,
        time_frame=1,  # 10 second simulation
        max_flow_duration=10 # 10 Mbps average
    )

    # Combined configuration
    config = {
        'nodes': network['nodes'],
        'topology': network['topology'],
        'bandwidths': network['bandwidths'],
        'traffic_set': traffic
    }

    #after generating traffic, use it to iterate with NN 