import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import myUtils
import pdb
from torch.utils.tensorboard import SummaryWriter   # <-- import
import copy 
import random
import json
import argparse
from Baselines.funcedBaseline import *

import pdb

#training params
gamma       = 3.0      # sharpness for alignment

#simulation params 
trafficScaling = 100000
max_hops    = 20       # how many hops to unroll
maxDemand   = 1.0
#numFlows = 30
beam_budget = 4      # sum of beam allocations per node

#constellation params 
orbitRadius = 6.946e6   
#numSatellites = 100
#orbitalPlanes = 10
inclination = 80 
phasingParameter = 5

EARTH_MEAN_RADIUS = 6371.0e3

numSatellites =  200
orbitalPlanes = 20

positions, vecs = myUtils.generateWalkerStarConstellationPoints(numSatellites,
                                                inclination,
                                                orbitalPlanes,
                                                phasingParameter,
                                                orbitRadius)
positions = np.reshape(positions, [np.shape(positions)[0]*np.shape(positions)[1], np.shape(positions)[2]])

gridPlusConn = myUtils.build_plus_grid_connectivity(positions,
                                                    orbitalPlanes,
                                                    int(numSatellites/orbitalPlanes))

#disconnect one satellite to see its effect  
gridPlusConn[0,gridPlusConn[0]]  = False
gridPlusConn[gridPlusConn[:,0],0] = False

val, vec = myUtils.calculate_algebraic_connectivity(gridPlusConn)

pdb.set_trace()