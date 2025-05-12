import torch
from torch.nn import Linear, ReLU
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import anotherPacketSim
import pdb
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import Sequential, GCNConv, NNConv, GENConv
from torch_geometric.utils import dense_to_sparse
from torch.nn.utils import clip_grad_norm_

class FeedForwardNN(nn.Module):
    def __init__(self, separateNetworkDims, jointNetworkDims, activation='relu', dropout_rate=0.5):
        """
        Args:
            separateNetworkDims: dimensions for each branch
            jointNetworkDims: dimensions for combined branches
            activation (str): Activation function ('relu', 'sigmoid', 'tanh', 'leaky_relu')
            dropout_rate (float): Dropout probability (0 = no dropout)
        """

        #please note, separateNetworkDims[-1] = jointNetworkDims[0]*3 
        #as we have 3 branches we are superimposing 

        super(FeedForwardNN, self).__init__()

        #Create networks for the three inputs 
        
        #Topology
        self.topologyNetwork = self._build_layers(separateNetworkDims, activation, dropout_rate)

        #Traffic Information/Rates
        self.trafficNetwork = self._build_layers(separateNetworkDims, activation, dropout_rate)

        #Routing Table 
        self.routingNetwork = self._build_layers(separateNetworkDims, activation, dropout_rate)
        
        #Create network for joint output 
        self.jointNetwork = self._build_layers(jointNetworkDims, activation, dropout_rate)

    def _build_layers(self, layer_dims, activation, dropout_rate):
        layers = []
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))

            if i < len(layer_dims)-2:
                layers.append(self._get_activation(activation))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)
        
    def _get_activation(self, name):
        """Helper to get activation function"""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(name.lower(), nn.ReLU())  # default to ReLU
    
    def forward(self, x):
        #get network inputs 
        newX = [(torch.flatten(ans.float())).to(device) for ans in x]

        #put network inputs on device 
        #[newXComponent.to(device) for newXComponent in newX]

        topologyOut = self.topologyNetwork(newX[0])
        trafficOut = self.trafficNetwork(newX[1])
        routingOut = self.routingNetwork(newX[2])

        #get concatenated outputs 
        concated = torch.concat([topologyOut, trafficOut, routingOut])

        #put concatenated into network
        output = self.jointNetwork(concated)
        
        return output

class GCNNetwork(nn.Module):
    def __init__(self, jointNetworkDims, activation='relu', dropout_rate=0.3):
        super(GCNNetwork, self).__init__()
        
        self.activation = self._get_activation(activation)
        self.dropout_rate = dropout_rate

        # Topology: Assume scalar node features (in_channels=1)
        self.topologyNetwork = self._build_gcn_branch(
            in_channels=1, hidden_channels=128, out_dim=int(jointNetworkDims[0] // 3)
        )

        # Traffic: Assume 10-dimensional node features
        self.trafficNetwork = self._build_gcn_branch(
            in_channels=10, hidden_channels=128, out_dim=int(jointNetworkDims[0] // 3)
        )

        # Routing Table: Treated as dense MLP
        self.routingNetwork = self._build_mlp_branch(
            layer_dims=[100, 80, 80, 60, int(jointNetworkDims[0] // 3)]
        )

        # Final joint MLP
        self.jointNetwork = self._build_mlp_branch(jointNetworkDims)

    class AttentionPooling(nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            # Learn a query vector to compute attention scores
            self.query = nn.Linear(feature_dim, 1)  # Maps each feature to a score

        def forward(self, x):
            """
            Args:
                x: Input tensor of shape [num_nodes, feature_dim] (e.g., [10, 50])
            Returns:
                pooled: Output tensor of shape [feature_dim] (e.g., [50])
            """
            # Compute attention scores [num_nodes, 1]
            scores = self.query(x)  
            scores = torch.softmax(scores, dim=0)  # Softmax over nodes

            # Weighted sum of features [feature_dim]
            pooled = torch.sum(scores * x, dim=0)  
            return pooled
        
    def _build_gcn_branch(self, in_channels, hidden_channels, out_dim, num_gcn_layers=2, num_mlp_layers=5):
        layers = nn.ModuleList()

        # --- GCN Layers ---
        for i in range(num_gcn_layers):
            input_dim = in_channels if i == 0 else hidden_channels
            layers.append(GENConv(input_dim, hidden_channels, edge_dim=1))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(self.activation)
            layers.append(nn.Dropout(self.dropout_rate))

        # --- MLP Layers ---
        for i in range(num_mlp_layers):
            input_dim = hidden_channels
            output_dim = hidden_channels if i < num_mlp_layers - 1 else out_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < num_mlp_layers - 1:
                layers.append(self.activation)
                layers.append(nn.Dropout(self.dropout_rate))

        # --- Optional: Attention pooling ---
        layers.append(self.AttentionPooling(feature_dim=out_dim))

        return layers

    def _build_mlp_branch(self, layer_dims):
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                layers.append(self.activation)
                layers.append(nn.Dropout(self.dropout_rate))
        return nn.Sequential(*layers)

    def _get_activation(self, name):
        return {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }.get(name.lower(), nn.ReLU())

    
    def forward(self, x):
        
        #first, read inputs in  
        topology = ((x[0].float())).to(device)
        traffic = ((x[1].float())).to(device)
        routing = ((x[2].float())).to(device)

        #then do formatting for each 
        
        #Topology Formatting and normalization 
        #first, convert to edge creation 
        topology = dense_to_sparse(topology)
        #then, format with BW sum for first component, edge presence for second, and edge weights for third 
        topology = [torch.sum(x[0].float(), axis = 1)/torch.sum(x[0]), topology[0], (topology[1].unsqueeze(0).T)/(torch.sum(topology[1])) ]

        #Traffic Formatting and normalization 
        traffic = dense_to_sparse(traffic)
        traffic = [torch.eye(num_nodes) , traffic[0], (traffic[1].unsqueeze(0).T)/(torch.sum(traffic[1])) ]
        
        #Routing formatting 
        routing = torch.flatten(routing)

        #pdb.set_trace()

        #Then, go through networks 

        #Topology forward 
        #should have sums of edges for first input 
        topologyOut = self.topologyNetwork[0](topology[0].unsqueeze(1), topology[1], edge_attr = topology[2])  # Unpack GCNConv and apply manually

        for i, layer in enumerate(self.topologyNetwork[1:]):
            if isinstance(layer,GCNConv) or isinstance(layer,GENConv):
                topologyOut = layer(topologyOut, topology[1], edge_attr = topology[2])
            else:
                topologyOut = layer(topologyOut)
        
        #topologyOut = torch.mean(topologyOut, dim=0)

        #Traffic forward 
        trafficOut = self.trafficNetwork[0](*traffic)  # Unpack GCNConv and apply manually
        for i, layer in enumerate(self.trafficNetwork[1:]):
            if isinstance(layer,GCNConv) or isinstance(layer,GENConv):
                trafficOut = layer(trafficOut, traffic[1], edge_attr = traffic[2])            
            else:
                trafficOut = layer(trafficOut)
        
        #pdb.set_trace()
        #trafficOut = torch.mean(trafficOut, dim=0)

        #Routing forward                 
        routingOut = self.routingNetwork(routing)

        #get concatenated outputs 
        concated = torch.concat([topologyOut, trafficOut, routingOut])

        #put concatenated into network
        output = self.jointNetwork(concated)
        
        return output
    
#first, generate predictor network architecture and components
#last dim is # of outputs  
predictorNetwork = GCNNetwork([150,100,30,7])
#predictorNetwork = FeedForwardNN([100,300,200,100,30,30], [90,90,30,1])

loss_fn = nn.SmoothL1Loss()               # Example: Mean Squared Error
#loss_fn = nn.MSELoss() 
optimizer = optim.Adam(predictorNetwork.parameters(), lr=1e-5)

#put on GPU for faster training 
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
predictorNetwork.to(device)

#then, specify the number of iterations for training
numIterations = 2000
writer = SummaryWriter()


#reduce randomness by having this outside the loop...

num_nodes = 10
links_per_node = 3
max_bw = 100

#store created network topology (so constant bandwidth)
adj_matrix = anotherPacketSim.generate_adjacency_matrix(num_nodes = num_nodes, 
                                                        links_per_node = links_per_node, 
                                                        max_bandwidth = max_bw)

#create simulator from adj_matrix, storing routing table  
sim = anotherPacketSim.NetworkSimulator(adj_matrix)
routingTable = sim.routing_table

#training loop
for i in range(numIterations): 

    predictorNetwork.train()  # Set model to training mode

    """Create Data"""

    #store traffic and associated rate matrix 
    traffic_set, rate_matrix = anotherPacketSim.generate_network_traffic_new(num_nodes, 1) 

    #get the flow delays for these components 
    records = anotherPacketSim.getFlowDelays(adj_matrix, traffic_set, sim)

    #get delays 
    delays = torch.tensor([res['total_delay'] for res in records])
    #get sizes 
    sizes = torch.tensor([res['packet_size'] for res in records])  # Fixed typo: 'packet' not 'packet'
    #now, produce statistics on this including: quantiles, variance, etc.
    
    #quantiles
    q = torch.linspace(0,1,5, dtype = float)
    quantiles = torch.quantile(delays, q)

    #then get avgDelay or 1/throughput, for target evaluation 
    avgDelay = sum( size * delay for size, delay in zip(sizes, delays)) / sum(sizes)  # Removed comma and fixed calculation
    varOfDelay = sum(size * (delay - avgDelay) ** 2 for size, delay in zip(sizes, delays)) / sum(sizes)
    
    #then, concat these together for a more robust distribution 
    targets = torch.cat([avgDelay.unsqueeze(0),quantiles,varOfDelay.unsqueeze(0)])

    """Start training operation"""
    #step many times
    for j in range(5):
        #forward pass 
        predict = predictorNetwork.forward(torch.tensor(np.array([adj_matrix, rate_matrix, routingTable])))

        #Compute loss 
        #target_tensor = torch.tensor([targets], dtype=torch.float32, device=predict.device)
        loss = loss_fn(predict, targets)

        optimizer.zero_grad()

        #Backward pass 
        loss.backward()

        clip_grad_norm_(predictorNetwork.parameters(), max_norm=1.0)

        #Optimizer step 
        optimizer.step()

        if(j == 1 or j == 30):

            #get loss value and print it 
            loss_value = loss.item()          
            absolute_loss = abs(loss_value) 

            writer.add_scalar("Loss/train", loss.item(), i)
            print("Diff in predict for mean delay: " + str(abs(predict[0] - targets[0])))
            print("Mean delay " + str(targets[0]))
            #print(absolute_loss)

            #print(targets)
    # print("% Diff")
    #print(100*((predict-target_tensor)/target_tensor))


writer.flush()
writer.close()