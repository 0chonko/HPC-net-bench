import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np


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

label_colors = {'Nccl': '#1f77b4', 'Nvlink': '#ff7f0e', 'Baseline': '#2ca02c', 'CudaAware': '#d62728', 'Aggregated': '#9467bd'}
label_machines = {'snellius': 'Snellius'}
label_topologies = {'halfnode': 'Single-node', 'wholenode': 'Multi-nodes'}
label_experiments = {'-pp-': 'Ping-pong', '-a2a-': 'AllToAll', '-ar-': 'AllReduce', '-hlo-': 'Halo', '-mpp-': 'Multi-Ping-Pong'}
label_containers = {'native': 'Native', 'hybrid': 'Hybrid', 'contained': 'Contained'}
label_architectures = {'a100': 'A100', 'h100': 'H100'}


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

    print('all_types: ' + str(all_types))
    print('all_machines: ' + str(all_machines))
    print('all_topologies: ' + str(all_topologies))
    print('all_experiments: ' + str(all_experiments))
    print('all_containers: ' + str(all_containers))

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
                plt.figure(figsize=(2.2, 3))
                for line in line_order:
                    if line in plots[key]:
                        print('Key: ', key)
                        print('Line: ', line)

                        linestyle = '--' if 'wholenode' in line else '-'
                        transfer_size = plots[key][line]['Transfer Size (B)']
                        bandwidth = plots[key][line]['Bandwidth (GB/s)']
                        type_name = line.split('-')[0]
                        container_name = line.split('-')[1]

                        print(container_name)
                        if len(plots[key][line]['Bandwidth (GB/s)']) > 0:
                            max_bandwidth = max(plots[key][line]['Bandwidth (GB/s)'])
                            print('max_bandwidth: ', max_bandwidth)

                        color = label_colors[type_name] if container_name != 'hybrid' else lighten_color(label_colors[type_name], 0.3)
                        linestyle = '--' if container_name == 'hybrid' else '-'

                        if ((not '-hlo-' in key)) and (type_name != "Aggregated" or ('-mpp-' in key)):
                            marker = {'Baseline': 'x', 'CudaAware': 's', 'Nccl': 'o', 'Nvlink': 'v'}
                            # sns.lineplot(transfer_size, bandwidth, label="%s %s" % (type_name, label_containers[container_name]), linestyle=linestyle, color=label_colors[type_name], markersize=2, marker=marker[type_name], linewidth=1.7)
                            
                            sns.lineplot(x=transfer_size, y=bandwidth, label=f"{type_name} {label_containers[container_name]}",
                                         linestyle=linestyle, linewidth=2.2, color=color, marker='.', markersize=10 if container_name == 'hybrid' else 12)# plt.annotate(f"Max Bandwidth: {max_bandwidth} GB/s", xy=(transfer_size, max_bandwidth), xytext=(transfer_size, max_bandwidth), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

                            

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
                        if e == 'Ping-pong':
                            peak = 160
                        else:
                            peak = 480
                    else:
                        if e == 'Ping-pong':
                            peak = 12.5
                        else:
                            peak = 50
                elif m == 'LUMI':
                    if t == "2-4":
                        peak = 25
                    else:
                        raise RuntimeError(f"Topology {t} unknown")
                elif m == 'Snellius':
                    if t == 'Single-node':
                        if e == 'Ping-pong':
                            peak = 100
                        else:
                            peak = 300
                    else:
                        if e == 'Ping-pong':
                            peak = 25
                        else:
                            peak = 100
                else:
                    raise RuntimeError(f"Machine {m} unknown")

                print("[m, t, e] = [%s, %s, %s] ---> peak = %d" % (m, t, e, peak))

                plt.xscale('log')
                #plt.yscale('log')
                plt.xlabel('Transfer Size (B)', fontsize=12)
                plt.ylabel('Bandwidth (GB/s)', fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.xlim(1, 1e9)
                # plt.ylim(0, 1e3)
                # plt.title(e + ' ' + t, fontsize=10)
                print("line_order:", line_order)
                # plt.legend(loc='lower right', fontsize=8, ncols=1)
                plt.legend().remove()
                plt.grid(True, linestyle='--')
                plt.minorticks_on()
                # plt.axhline(y=peak, color='black', linestyle='--', label='Theoretical peak (%s GB/s)' % peak)
                # plt.text(1, peak, 'H100 peak (%s GB/s)' % peak, fontsize=7, va='bottom', ha='left')
                #plt.axhline(y=peak, color='black', linestyle='--', label='Theoretical peak (%s GB/s)' % peak)
                # plt.text(1, peak, 'A100 peak (%s GB/s)' % peak, fontsize=7, va='bottom', ha='left')
                # plt.annotate('Theoretical peak (%s GB/s)' % peak, xy=(1, peak), xytext=(1, peak-40), fontsize=7)
                plt.savefig('%s%s%s%sBandwidth_h100.svg' % ('sout/', m, e, t), bbox_inches='tight', dpi=100)
                plt.show()
# directory_path1 = 'sout/native/'
# directory_path2 = 'sout/hybrid/'

directory_path1 = 'sout/native-2-nodes_h100'
directory_path2 = 'sout/hybrid-2-nodes_h100'

file_paths1 = [os.path.join(directory_path1, file) for file in os.listdir(directory_path1) if file.endswith('.out')]
file_paths2 = [os.path.join(directory_path2, file) for file in os.listdir(directory_path2) if file.endswith('.out')]
file_paths = file_paths1 + file_paths2

# file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.out')]

# plotTwoArchitectures(file_paths)
plot_performance(file_paths)
