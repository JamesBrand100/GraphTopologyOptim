import numpy as np

flows_matrix = np.arange(30)
num_top_flows = 10
flat_flows = flows_matrix[flows_matrix > 0]

if len(flat_flows) > num_top_flows:
    threshold_value = np.partition(flat_flows, -num_top_flows)[-num_top_flows]
    flows_matrix[flows_matrix < threshold_value] = 0
else:
    pass

print(flows_matrix)