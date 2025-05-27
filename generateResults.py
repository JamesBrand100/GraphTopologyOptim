#get code to run 
from walkerDeltaTopologyOptimization import run_simulation

#then, generate input dictionary 
inputs = {
    'numFlows': 30,
    'numSatellites': 200, 
    'orbitalPlanes': 20, 
    'routingMethod': 'LOSweight',
    'epochs': 200, 
    'lr': 0.05, 
    'fileToSaveTo': "LOSweightSmallConst"
} 

#then, iterate over and modify inputs  
for i in range(1,11): 

    #generate and modify non default params 
    numFlows = i * 10
    inputs['numFlows'] = numFlows
    inputs['fileToSaveTo'] = "Data/" + inputs['fileToSaveTo'] + str(numFlows)
    
    #run the function for generating the simulation <3 
    run_simulation(**inputs)

#run_simulation(**inputs)

