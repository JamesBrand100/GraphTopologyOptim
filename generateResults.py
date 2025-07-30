#get code to run 
from walkerDeltaTopologyOptimization import run_simulation

import time
from datetime import datetime, timedelta
import myUtils

def pause_then_run(delay_seconds=3600, task=None):
    """Pauses for a given time and then runs a task (if provided)."""
    target_time = datetime.now() + timedelta(seconds=delay_seconds)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Pausing for {delay_seconds // 60} minutes. Will resume at {target_time.strftime('%H:%M:%S')}...")
    time.sleep(delay_seconds)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Resuming...")
    if task:
        task()

#pause_then_run()

#then, generate input dictionary 
inputs = {
    'numFlows': 30,
    'numSatellites': 360, 
    'orbitalPlanes': 15, 
    'routingMethod': 'LOSweight',
    'epochs': 100, #200
    'lr': 0.030, #.03
    'fileToSaveTo': "RoutingLogitsSmallConst",
    'metricToOptimize': "hops",
    'demandDist' : "popBased",
    'inLineness' : "GCD"
} 

#360, 24 
#thought: we might have been using LearnedLogits for this.... nah that was before .

#then, iterate over and modify inputs  
for i in range(1,10): 
    #store numSatellites 
    numSatellites = 40*i 

    #get # orbital planes 
    numPlanes = myUtils.find_closest_divisor_to_sqrt(numSatellites)

    #update inputs 
    inputs['numSatellites'] = numSatellites
    inputs['orbitalPlanes'] = numPlanes
    inputs['fileToSaveTo'] = "Data/" + "VariableConstHopsBasedGCDLogit" + str(numSatellites)

    #create debug 
    print("Iteration # : " + str(i))
    
    #run the function for generating the simulation <3 
    run_simulation(**inputs)

#then, iterate over and modify inputs  
for i in range(1,10): 
    #store numSatellites 
    numSatellites = 40*i 

    #get # orbital planes 
    numPlanes = myUtils.find_closest_divisor_to_sqrt(numSatellites)

    #update inputs 
    inputs['numSatellites'] = numSatellites
    inputs['orbitalPlanes'] = numPlanes
    inputs['fileToSaveTo'] = "Data/" + "VariableConstLatencyBasedGCDLogit" + str(numSatellites)
    inputs['metricToOptimize'] = "latency"

    #create debug 
    print("Iteration # : " + str(i))
    
    #run the function for generating the simulation <3 
    run_simulation(**inputs)

#then, iterate over and modify inputs  
# for i in range(1,10): 
#     #store numSatellites 
#     numSatellites = 40*i 

#     #get # orbital planes 
#     numPlanes = myUtils.find_closest_divisor_to_sqrt(numSatellites)

#     #update inputs 
#     inputs['numSatellites'] = numSatellites
#     inputs['orbitalPlanes'] = numPlanes
#     inputs['fileToSaveTo'] = "Data/" + "VariableConstHopsBasedAngleLogit" + str(numSatellites)
#     inputs['metricToOptimize'] = "hops"
#     inputs['inLineness'] = "angle"

#     #create debug 
#     print("Iteration # : " + str(i))
    
#     #run the function for generating the simulation <3 
#     run_simulation(**inputs)

# #then, iterate over and modify inputs  
# for i in range(1,10): 
#     #store numSatellites 
#     numSatellites = 40*i 

#     #get # orbital planes 
#     numPlanes = myUtils.find_closest_divisor_to_sqrt(numSatellites)

#     #update inputs 
#     inputs['numSatellites'] = numSatellites
#     inputs['orbitalPlanes'] = numPlanes
#     inputs['fileToSaveTo'] = "Data/" + "VariableConstLatencyBasedAngleLogit" + str(numSatellites)
#     inputs['metricToOptimize'] = "latency"
#     inputs['inLineness'] = "angle"

#     #create debug 
#     print("Iteration # : " + str(i))
    
#     #run the function for generating the simulation <3 
#     run_simulation(**inputs)


