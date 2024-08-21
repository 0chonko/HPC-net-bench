'''
Notes:
    - machine name should not contain underscores (otherwise `unpack` fails)
'''

import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def unpack(file_paths):
    files=[]
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        machines, experiments, types, topology, _ = file_name.split('_')
        # linestyle = '-' if "Internode" in label else '--'
        df = extract_data(file_path)
        files.append([machines,experiments,types,topology,df])
        #print(files)
    return files


# Function to extract data from a file
substring = "[Average]"
def extract_data(file_path):
    transfer_sizes = []
    bandwidths = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'Transfer size \(B\):\s+(\d+).*Energy end:\s+([\d.]+)', line)
            if match:
                transfer_sizes.append(int(match.group(1)))
                bandwidths.append(float(match.group(2)))

    # Group bandwidths by transfer size
    grouped_bandwidths = {}
    for i in range(len(transfer_sizes)):
        transfer_size = transfer_sizes[i]
        if transfer_size not in grouped_bandwidths:
            grouped_bandwidths[transfer_size] = []
        grouped_bandwidths[transfer_size].append(bandwidths[i])

    # Calculate energy difference for each group
    energy_per_byte = []
    for transfer_size, group_bandwidths in grouped_bandwidths.items():
        energy_diff = group_bandwidths[-1] - group_bandwidths[0]  # Last minus first

        energy_per_byte.append((energy_diff / 1000))  # Convert to Joules and divide by transfer size (B)

    # return pd.DataFrame({'Transfer Size (B)': list(grouped_bandwidths.keys()), 'Energy End': energy_per_byte})
    return pd.DataFrame({'Transfer Size (B)': transfer_sizes, 'Energy End': bandwidths})


    # return pd.DataFrame({'Transfer Size (B)': transfer_sizes, 'Energy End': bandwidths})


lable_colors = { 'Baseline': 'blue', 'CudaAware': 'red', 'Nccl': 'green', 'Nvlink': 'gray', 'Aggregated': 'orange'}

# lable_machines = { 'leonardo': 'Leonardo', 'marzola': 'Marzola'}
lable_machines = { 'snellius': 'Snellius'}
label_topologies = { 'halfnode': 'Single-node', 'wholenode': 'Multi-nodes'}
# label_topologies = { '2_4': 'Two-nodes(4-GPUs-each)'}
lable_experiments = { '-pp-': 'Ping-pong', '-a2a-': 'AllToAll', '-ar-': 'AllReduce', '-hlo-': 'Halo', '-mpp-': 'Multi-Ping-Pong'}
# lable_experiments = { '-a2a-': 'AllToAll', '-ar-': 'AllReduce'}

# Function to plot performance comparison
def plot_performance(file_paths):

    all_types = []
    all_machines = []
    all_topologyes = []
    all_experiments = []
    files = unpack(file_paths)
    for i in files:
        if i[0] not in all_machines:
            all_machines.append(i[0])
        if i[1] not in all_experiments:
            all_experiments.append(i[1])
        if i[2] not in all_types:
            all_types.append(i[2])
        if i[3] not in all_topologyes:
            all_topologyes.append(i[3])

    print('all_types: ' + str(all_types))
    print('all_machines: ' + str(all_machines))
    print('all_topologyes: ' + str(all_topologyes))
    print('all_experiments: ' + str(all_experiments))

    plots={}
    for m in all_machines:
        for e in all_experiments:
            for t in all_topologyes:
                if (e != "mpp" or t != "singlenode"):
                    plots[m+'-'+e+'-'+t]={}

    for f in files:
        plots[f[0]+'-'+f[1]+'-'+f[3]][f[2]] = f[4]

    line_order = list(lable_colors.keys())

    for topology in all_topologyes:
        for key in plots:
            if topology in key:
                plt.figure(figsize=(10, 6))
                for line in line_order:
                    if line in plots[key]:
                        print('Key: ', key)
                        print('Line: ', line)

                        linestyle = '--' if 'multinode' in line else '-'
                        transfer_size = plots[key][line]['Transfer Size (B)']
                        bandwidth = plots[key][line]['Energy End']

                        print('transfer_size: ', transfer_size)
                        print('bandwidth: ', bandwidth)
                        print('linestyle: ', linestyle)
                        print('color: ', lable_colors[line])

                        if (len(plots[key][line]['Energy End']) > 0):
                            max_bandwidth = max(plots[key][line]['Energy End'])
                            print('max_bandwidth: ', max_bandwidth)

                        if ((line != 'Nvlink' or (not '-ar-' in key and not '-hlo-' in key)) and (line != "Aggregated" or ('-mpp-' in key))):
                            plt.plot(transfer_size, bandwidth, label="%s (up to %d Joules)" % (line, max_bandwidth), linestyle=linestyle, color=lable_colors[line])

                #e = 'ping-pong' if 'pp' in key else 'all2all'
                for k in lable_experiments:
                    if k in key:
                        e = lable_experiments[k]
                #m = 'Leonardo' if 'leonardo' in key else 'Marzola'
                for k in lable_machines:
                    if k in key:
                        m = lable_machines[k]
                #t = 'SingleNode' if 'singlenode' in key else 'MultiNode'
                for k in label_topologies:
                    if k in key:
                        t = label_topologies[k]


                # print("[m, t, e] = [%s, %s, %s] ---> peak = %d" % (m, t, e, peak))
                # plt.axhline(y=peak, color='red', linestyle='--', label='Theoretical peak (%s GB/s)' % peak)

                plt.xscale('log', base=2)
                # plt.xscale('linear')
                plt.yscale('linear')
                plt.xlabel('Transfer Size (B)')
                plt.ylabel('Energy End')

                plt.title(m + ' ' + e + ' ' + t + ' Performance Comparison')
                print("line_order:", line_order)
                #plt.legend(legend_order)
                plt.legend()
                plt.grid(True)

                plt.savefig('%s%s%s%sBandwidth.png' % (directory_path, m, e, t))
                plt.show()

# Specify the directory containing the files
directory_path = 'sout/energy/'

# Get a list of file paths in the directory
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.out')]

# Plot performance comparison
plot_performance(file_paths)
