import torch
from torch.nn import Linear, ReLU
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import anotherPacketSim
import pdb
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.utils import dense_to_sparse

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
        """
        Args:
            separateNetworkDims: dimensions for each branch
            jointNetworkDims: dimensions for combined branches
            activation (str): Activation function ('relu', 'sigmoid', 'tanh', 'leaky_relu')
            dropout_rate (float): Dropout probability (0 = no dropout)
        """

        #as we have 3 branches we are superimposing 
        super(GCNNetwork, self).__init__()

        #Create networks for the three inputs 
        
        #Topology
        self.topologyNetwork = [
            (GCNConv(in_channels = 1, out_channels = 200)),
            ReLU(inplace=True),
            (GCNConv(in_channels = 200, out_channels = 200)),
            ReLU(inplace=True),
            Linear(200, 200),
            ReLU(inplace=True),
            (GCNConv(in_channels = 200, out_channels = 200)),
            ReLU(inplace=True),
            (GCNConv(in_channels = 200, out_channels = 200)),
            ReLU(inplace=True),
            Linear(200, 200),
            ReLU(inplace=True),
            Linear(200, int(jointNetworkDims[0]/3)),
            nn.Dropout(dropout_rate),
        ]

        #Traffic Information/Rates
        self.trafficNetwork = [
            (GCNConv(10, 200)),
            ReLU(inplace=True),
            (GCNConv(200, 200)),
            ReLU(inplace=True),
            Linear(200, 200),
            ReLU(inplace=True),
            (GCNConv(200, 200)),
            ReLU(inplace=True),
            (GCNConv(200, 200)),
            ReLU(inplace=True),
            Linear(200, 200),
            ReLU(inplace=True),
            Linear(200, int(jointNetworkDims[0]/3)),
            nn.Dropout(dropout_rate),
        ]

        #Routing Table 
        self.routingNetwork = self._build_layers([100,50,50,int(jointNetworkDims[0]/3)], activation, dropout_rate)
        
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
    
    def _build_branch(self, numOut, activation, dropout_rate):

        model = Sequential('x, edge_index', 'edge value', [
            (GCNConv(10, 100, 3, 50), 'x, edge_index -> x', ' edge value'),
            ReLU(inplace=True),
            Linear(100, numOut),
            nn.Dropout(dropout_rate),
        ])

        return model
        
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
        
        #first, read inputs in  
        topology = ((x[0].float())).to(device)
        traffic = ((x[1].float())).to(device)
        routing = ((x[2].float())).to(device)

        #then do formatting for each 
        #first, convert to edge creation 
        topology = dense_to_sparse(topology)

        #then, format with BW sum for first component, edge presence for second, and edge weights for third 
        topology = [torch.sum(x[0].float(), axis = 1)/torch.sum(x[0]), topology[0], topology[1].unsqueeze(0).T ]

        #topology = [torch.sum(x[0].float(), axis = 1), topology[0], topology[1].unsqueeze(0).T ]

        traffic = dense_to_sparse(traffic)
        traffic = [torch.eye(num_nodes) , traffic[0], traffic[1].unsqueeze(0).T ]
        
        routing = torch.flatten(routing)

        #then, go through branched networks 
        #GCN Conv layer is special (first one)

        #should have sums of edges for first input 
        topologyOut = self.topologyNetwork[0](topology[0].unsqueeze(1), topology[1], edge_weight = topology[2])  # Unpack GCNConv and apply manually

        #need to examine shape here
        #pdb.set_trace()

        for i, layer in enumerate(self.topologyNetwork[1:]):
            if isinstance(layer,GCNConv):
                topologyOut = layer(topologyOut, topology[1], edge_weight = topology[2])
            else:
                topologyOut = layer(topologyOut)
        
        topologyOut = torch.mean(topologyOut, dim=0)

        trafficOut = self.trafficNetwork[0](*traffic)  # Unpack GCNConv and apply manually
        for i, layer in enumerate(self.trafficNetwork[1:]):
            if isinstance(layer,GCNConv):
                trafficOut = layer(trafficOut, traffic[1], edge_weight = traffic[2])            
            else:
                trafficOut = layer(trafficOut)
        trafficOut = torch.mean(trafficOut, dim=0)

        #trafficOut = self.trafficNetwork(*traffic)
                
        routingOut = self.routingNetwork(routing)

        #get concatenated outputs 
        concated = torch.concat([topologyOut, trafficOut, routingOut])

        #put concatenated into network
        output = self.jointNetwork(concated)
        
        return output
    
#first, generate predictor network architecture and components 
predictorNetwork = GCNNetwork([150,100,30,1])
#predictorNetwork = FeedForwardNN([100,300,200,100,30,30], [90,90,30,1])

loss_fn = nn.SmoothL1Loss()               # Example: Mean Squared Error
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
    delays = [res['total_delay'] for res in records]
    #get sizes 
    sizes = [res['packet_size'] for res in records]  # Fixed typo: 'packet' not 'packet'
    #then get avgDelay or 1/throughput, for target evaluation 
    target = sum( size * delay for size, delay in zip(sizes, delays)) / sum(sizes)  # Removed comma and fixed calculation
    
    """Start training operation"""

    #step many times
    for j in range(30):
        #forward pass 
        predict = predictorNetwork.forward(torch.tensor(np.array([adj_matrix, rate_matrix, routingTable])))

        #Compute loss 
        target_tensor = torch.tensor([target], dtype=torch.float32, device=predict.device)
        loss = loss_fn(predict, target_tensor)

        #Backward pass 
        loss.backward()

        #Optimizer step 
        optimizer.step()

    #get loss value and print it 
    loss_value = loss.item()          
    absolute_loss = abs(loss_value) 

    writer.add_scalar("Loss/train", loss.item(), i)
    print("Loss")
    print(absolute_loss)
    print(target)
    # print("% Diff")
    #print(100*((predict-target_tensor)/target_tensor))


writer.flush()
writer.close()