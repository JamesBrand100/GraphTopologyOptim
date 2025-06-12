import pdb
import json
import matplotlib.pyplot as plt
import re
import os

data_dir = '../Data/VariableConst/'  # Assuming the script is run from the parent directory of 'Data'
files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

x = []
#numSats = []
my_method_latencies = []
grid_plus_latencies = []
motif_latencies = []

for filename in sorted(files):
    #Change this line for each type of run / analysis
    match = re.search(r'RoutingLogitsConst(\d+)', filename)
    if match:
        flow_value = int(match.group(1))
        x.append(flow_value)
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as f:
            metrics = json.load(f)
            my_method_latencies.append(metrics["My method size weighted latency"])
            grid_plus_latencies.append(metrics["Grid plus size weighted latency"])
            motif_latencies.append(metrics["Motif size weighted latency"])

# Sort all three lists based on the 'flows' values
sorted_data = sorted(zip(x, my_method_latencies, grid_plus_latencies, motif_latencies))
x, my_method_latencies, grid_plus_latencies, motif_latencies = zip(*sorted_data)

#pdb.set_trace()

plt.figure(figsize=(10, 6))
plt.plot(x, my_method_latencies, marker='o', label='Differential Method')
plt.plot(x, grid_plus_latencies, marker='x', label='Grid Plus')
plt.plot(x, motif_latencies, marker='.', label='Motif')

plt.rcParams.update({
    'font.size': 16,          # Default font size
    'axes.titlesize': 14,     # Title size
    'axes.labelsize': 1200,     # Axis label size
    'xtick.labelsize': 17,    # X-tick label size
    'ytick.labelsize': 17,    # Y-tick label size
    'legend.fontsize': 17     # Legend font size
})

#plt.title('Hops * Traffic vs. Number of Satellites')
plt.xlabel('Number of Satellites', fontsize = 16)
plt.ylabel('Weighted Network Delay (seconds * bytes)', fontsize = 16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()