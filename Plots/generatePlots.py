import pdb
import json
import matplotlib.pyplot as plt
import re
import os

data_dir = '../Data/GCDLogits'  # Assuming the script is run from the parent directory of 'Data'
files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

x = []
#numSats = []
my_method_latencies = []
grid_plus_latencies = []
motif_latencies = []

optimizedVar = "latency"

for filename in sorted(files):
    #Change this line for each type of run / analysis
    if(optimizedVar == "latency"):
        strStart = "VariableConstLatencyBasedGCDLogit"
        #match = re.search(r'VariableConstLatencyPopBased(\d+)', filename)
    if(optimizedVar == "hop"):
        strStart = "VariableConstHopsBasedGCDLogit"
        #match = re.search(r'VariableConstHopPopBased(\d+)', filename)
    match = re.search(strStart+r'(\d+)', filename)
    if match:
        flow_value = int(match.group(1))
        x.append(flow_value)
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as f:

            # Read the entire file content as a string
            file_content = f.read()

            #pdb.set_trace()

            # Define the prefix to remove
            prefix_to_remove = strStart#"VariableConstLatencyPopBased"

            # Check if the content starts with the prefix and remove it
            if file_content.startswith(prefix_to_remove):
                json_string = file_content[len(prefix_to_remove):]
            else:
                # If the prefix isn't found, assume it's pure JSON or raise an error
                # For robustness, you might want to log a warning here if it's unexpected
                json_string = file_content

            #pdb.set_trace()
            metrics = json.loads(json_string)
            totalDemand = metrics["Total Demand"]
            my_method_latencies.append((metrics["My method size weighted latency"]+ metrics["Latency Addendum"])/totalDemand)
            grid_plus_latencies.append((metrics["Grid plus size weighted latency"]+ metrics["Latency Addendum"])/totalDemand)
            motif_latencies.append((metrics["Motif size weighted latency"] + metrics["Latency Addendum"])/totalDemand)
            print(metrics["Latency Addendum"])

# Sort all three lists based on the 'flows' values
sorted_data = sorted(zip(x, my_method_latencies, grid_plus_latencies, motif_latencies))

x, my_method_latencies, grid_plus_latencies, motif_latencies = zip(*sorted_data)
import numpy as np 
x  = np.array(x)
#pdb.set_trace()

# plt.bar(x_pos + offset1, my_method_latencies_sorted, bar_width, label='Differential Method')
# # Adjusted for the [1:] slicing if it's consistently needed for Grid Plus, otherwise use full list
# # If grid_plus_latencies_sorted[1:] means the second value onwards, then x_pos[1:] should also be used
# plt.bar(x_pos[1:] + offset2, grid_plus_latencies_sorted[1:], bar_width, label='Grid Plus')
# plt.bar(x_pos + offset3, motif_latencies_sorted, bar_width, label='Motif')

# # Set x-axis ticks to be at the center of each group of bars
# # This is crucial for clear labeling of the x-axis values
# plt.xticks(x_pos, x_sorted)

#plot = "barplot"

if(optimizedVar == "hop"):
    plot = "lineplot"
if(optimizedVar == "latency"):
    plot = "barplot"

if(plot == "barplot"):
    bar_width = 5 # Increased from default 0.8 to 0.25 for a noticeable thickness

    # Calculate the offsets for each set of bars so they don't overlap
    # For 3 sets of bars, you typically want (bar_width * -1), 0, (bar_width * 1)
    # or (bar_width * -1.5), (bar_width * -0.5), (bar_width * 0.5) for more spacing
    offset1 = -bar_width
    offset2 = 0 # Center bar
    offset3 = bar_width

    plt.figure(figsize=(10, 6)) # You might still want to adjust this for better visuals

    plt.bar(x[1:] + offset1, my_method_latencies[1:], bar_width,label='GTopOpt')
    # Keep the [1:] slicing if it's intentional for Grid Plus data to skip the first point
    plt.bar(x[1:] + offset2, grid_plus_latencies[1:] , bar_width,label='Grid Plus')
    plt.bar(x[1:] + offset3, motif_latencies[1:], bar_width,label='Motif')

if(plot == "lineplot"):
    plt.plot(x[1:], my_method_latencies[1:], marker='o', label='GTopOpt')
    plt.plot(x[1:], grid_plus_latencies[1:], marker='x', label='Grid Plus')
    plt.plot(x[1:], motif_latencies[1:], marker='.', label='Motif')

    plt.ylim(bottom=0)

    #exit 


plt.rcParams.update({
    'font.size': 16,          # Default font size
    'axes.titlesize': 14,     # Title size
    'axes.labelsize': 1200,     # Axis label size
    'xtick.labelsize': 17,    # X-tick label size
    'ytick.labelsize': 17,    # Y-tick label size
    'legend.fontsize': 15    # Legend font size
})

#plt.title('Hops * Traffic vs. Number of Satellites')
plt.xlabel('Number of Satellites', fontsize = 16)
plt.ylabel('Weighted Network Delay (seconds)', fontsize = 16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()