#get code to run 
from walkerDeltaTopologyOptimization import run_simulation

#then, generate input dictionary 
inputs = {
    'numFlows': 30,
    'epochs': 200, 
    'numSatellites': 200, 
    'orbitalPlanes': 20, 
    'routingMethod': 'LOSweight', 
    'lr': 0.01, 
    'fileNameBase': "LOSweightSmallConst"
} 

#then, iterate over and modify inputs  
for i in range(10): 
    numFlows = i * 10
    inputs['fileName'] = inputs['fileName'] + str(numFlows)
    run_simulation(**inputs)

#run_simulation(**inputs)

