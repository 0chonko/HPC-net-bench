import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# Function to extract data from a file
substring = "[Average]"

def extract_data(file_path):
    with open(file_path, 'r') as file:
        data = '\n'.join(line.strip() for line in file if substring in line)

    transfer_sizes = [int(match.group(1)) for match in re.finditer(r'Transfer size \(B\):\s+(\d+)', data)]
    bandwidths = [float(match.group(1)) for match in re.finditer(r'Bandwidth \(GB/s\):\s+([\d.]+)', data)]

    return pd.DataFrame({'Transfer Size (B)': transfer_sizes, 'Bandwidth (GB/s)': bandwidths})

def unpack(file_paths):
    files = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        parts = file_name.split('_')
        
        if len(parts) == 6:
            machines, experiments, types, topology, container, _ = parts
        elif len(parts) == 5:
            machines, experiments, types, topology, _ = parts
            container = "native"
        else:
            raise ValueError(f"Unexpected file name format: {file_name}")
        
        df = extract_data(file_path)
        files.append([machines, experiments, types, topology, container, df])
    
    return files

label_colors = {'Baseline': 'blue', 'CudaAware': 'red', 'Nccl': 'green', 'Nvlink': 'gray', 'Aggregated': 'orange'}
label_machines = {'snellius': 'Snellius'}
label_topologies = {'halfnode': 'Single-node', 'wholenode': 'Multi-nodes'}
label_experiments = {'-pp-': 'Ping-pong', '-a2a-': 'AllToAll', '-ar-': 'AllReduce', '-hlo-': 'Halo', '-mpp-': 'Multi-Ping-Pong'}
label_containers = {'native': 'Native', 'hybrid': 'Hybrid', 'contained': 'Contained'}

def lighten_color(color, factor=0.5):
    """Lighten a given color by a factor."""
    color = mcolors.to_rgb(color)
    return mcolors.to_hex([min(1, c + (1 - c) * factor) for c in color])

def plot_performance(file_paths):
    all_types = []
    all_machines = []
    all_topologies = []
    all_experiments = []
    all_containers = []
    files = unpack(file_paths)

    for i in files:
        if i[0] not in all_machines:
            all_machines.append(i[0])
        if i[1] not in all_experiments:
            all_experiments.append(i[1])
        if i[2] not in all_types:
            all_types.append(i[2])
        if i[3] not in all_topologies:
            all_topologies.append(i[3])
        if i[4] not in all_containers:
            all_containers.append(i[4])

    plots = {}
    for m in all_machines:
        for e in all_experiments:
            for t in all_topologies:
                if (e != "mpp" or t != "singlenode"):
                    plots[m+'-'+e+'-'+t] = {}

    for f in files:
        key = f"{f[0]}-{f[1]}-{f[3]}"
        type_container = f"{f[2]}-{f[4]}"
        plots[key][type_container] = f[5]

    line_order = [f"{t}-{c}" for t in label_colors.keys() for c in label_containers.keys()]

    for topology in all_topologies:
        for key in plots:
            if topology in key:
                plt.figure(figsize=(12, 6))
                for line in line_order:
                    if line in plots[key]:
                        linestyle = '--' if 'multinode' in line else '-'
                        transfer_size = plots[key][line]['Transfer Size (B)']
                        bandwidth = plots[key][line]['Bandwidth (GB/s)']
                        type_name = line.split('-')[0]
                        container_name = line.split('-')[1]

                        max_bandwidth = max(plots[key][line]['Bandwidth (GB/s)'], default=0)
                        
                        color = label_colors[type_name] if container_name != 'hybrid' else lighten_color(label_colors[type_name], 0.3)
                        linestyle = '--' if container_name == 'hybrid' else '-'

                        if (type_name != 'Nvlink' or (not '-hlo-' in key)) and (type_name != "Aggregated" or ('-mpp-' in key)):
                            sns.lineplot(x=transfer_size, y=bandwidth, label=f"{type_name} {label_containers[container_name]} (up to {max_bandwidth} GB/s)",
                                         linestyle=linestyle, linewidth=4, color=color, marker='.', markersize=30)

                for k in label_experiments:
                    if k in key:
                        e = label_experiments[k]
                for k in label_machines:
                    if k in key:
                        m = label_machines[k]
                for k in label_topologies:
                    if k in key:
                        t = label_topologies[k]

                if m == 'Leonardo':
                    if t == 'Single-node':
                        peak = 100 if e == 'Ping-pong' else 300
                    else:
                        peak = 12.5 if e == 'Ping-pong' else 50
                elif m == 'LUMI':
                    if t == "2-4":
                        peak = 25
                    else:
                        raise RuntimeError(f"Topology {t} unknown")
                elif m == 'Snellius':
                    peak = 200
                else:
                    raise RuntimeError(f"Machine {m} unknown")

                plt.xscale('log', base=2)
                plt.yscale('log')
                plt.xlabel('Transfer Size (B)')
                plt.ylabel('Bandwidth (GB/s)')
                plt.title(f'{m} {e} {t} Performance Comparison')
                plt.legend()
                plt.grid(True)

                plt.savefig(f'sout/{m}{e}{t}Bandwidth.png')
                plt.show()

directory_path = 'sout/native-2-nodes/'
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.out')]
plot_performance(file_paths)
