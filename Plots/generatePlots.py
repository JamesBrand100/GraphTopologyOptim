import pdb
import json
import matplotlib.pyplot as plt
import re
import os

data_dir = '../Data/LOSweightConst360/'  # Assuming the script is run from the parent directory of 'Data'
files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

flows = []
my_method_latencies = []
grid_plus_latencies = []
motif_latencies = []

for filename in sorted(files):
    match = re.search(r'SmallConst(\d+)', filename)
    if match:
        flow_value = int(match.group(1))
        flows.append(flow_value)
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as f:
            metrics = json.load(f)
            my_method_latencies.append(metrics["My method size weighted latency"])
            grid_plus_latencies.append(metrics["Grid plus size weighted latency"])
            motif_latencies.append(metrics["Motif size weighted latency"])

# Sort all three lists based on the 'flows' values
sorted_data = sorted(zip(flows, my_method_latencies, grid_plus_latencies, motif_latencies))
flows, my_method_latencies, grid_plus_latencies, motif_latencies = zip(*sorted_data)

#pdb.set_trace()

plt.figure(figsize=(10, 6))
plt.plot(flows, my_method_latencies, marker='o', label='My method size weighted latency')
plt.plot(flows, grid_plus_latencies, marker='x', label='Grid plus size weighted latency')
plt.plot(flows, motif_latencies, marker='.', label='Motif size weighted latency')


plt.title('Average Network Delay vs. Number of Flows (300 Satellites)')
plt.xlabel('Number of Flows')
plt.ylabel('Average Network Delay (ms)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()