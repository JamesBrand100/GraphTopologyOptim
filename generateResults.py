#get code to run 
from walkerDeltaTopologyOptimization import run_simulation

import time
from datetime import datetime, timedelta

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
    'epochs': 200, #200
    'lr': 0.030, #.03
    'fileToSaveTo': "RoutingLogitsSmallConst"
} 

#thought: we might have been using LearnedLogits for this.... nah that was before .

#then, iterate over and modify inputs  
for i in range(1,11): 
    #generate and modify non default params 
    print("Iteration # : " + str(i))
    numFlows = i * 15
    inputs['numFlows'] = numFlows
    inputs['fileToSaveTo'] = "Data/" + "RoutingLogitsSmallConst" + str(numFlows)

    #run the function for generating the simulation <3 
    run_simulation(**inputs)

#run_simulation(**inputs)

