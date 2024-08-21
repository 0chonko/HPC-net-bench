import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from io import StringIO



# Initialize an empty DataFrame to hold all data
combined_data = pd.DataFrame()

# List of transfer sizes (1, 2, 4, ..., 536870912)
transfer_sizes = [2**i for i in range(30)]

# List of data folders
# data_folders = ['pp_nvlink', 'pp_nccl', 'pp_baseline', 'pp_cudaaware']
# data_folders = ['a2a_nccl', 'a2a_nvlink', 'a2a_baseline', 'a2a_cudaaware']

experiment = 'pp'
data_folders = [f'{experiment}_nccl', f'{experiment}_baseline', f'{experiment}_cudaaware']
# folders = ['sout/native','sout/hybrid/power_data', 'sout/contained/power_data']
folders = ['sout/native_h100','sout/hybrid_h100']


# Colors for each data folder
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

# Line styles for each source
line_styles = {'2-nodes': '-', '4-nodes': '--'}

# Load data from main folder
for folder in data_folders:
    for size in transfer_sizes:
        file_path = f'{folders[0]}/{folder}/gpuStats_3_{size}.csv'
        if os.path.exists(file_path):
            print(file_path)
            data = pd.read_csv(file_path)
            data['transfer_size'] = size  # Add a column for transfer size
            data['data_folder'] = folder  # Add a column for data folder
            data['source'] = '2-nodes'  # Add a column for source
            combined_data = pd.concat([combined_data, data], axis=0)

# Load data from additional folder
for folder in data_folders:
    for size in transfer_sizes:
        file_path = f'{folders[1]}/{folder}/gpuStats_0_{size}.csv'
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data['transfer_size'] = size  # Add a column for transfer size
            data['data_folder'] = folder  # Differentiate from main folder
            data['source'] = '4-nodes'  # Add a column for source
            combined_data = pd.concat([combined_data, data], axis=0)

# Verify columns in the combined data
print("Column names in the combined data:", combined_data.columns)

# Column names
timestamp_col = 'timestamp'
power_draw_col = ' power_draw_w'
gpu_util_col = ' utilization_gpu'
mem_util_col = ' utilization_memory'
temperature_col = ' temperature_gpu'
memory_free_mib = ' memory_free_mib'
memory_used_mib  = ' memory_used_mib'
total_energy_consumption = ' total_energy_consumption_j'
cols = [timestamp_col, power_draw_col, gpu_util_col, mem_util_col, temperature_col, memory_free_mib, memory_used_mib, total_energy_consumption]



cols_to_plot= [memory_used_mib, power_draw_col, mem_util_col]
# cols_to_plot= [power_draw_col]
# cols_to_plot= [mem_util_col]




# Check if the columns exist
for col in [timestamp_col, power_draw_col, gpu_util_col, mem_util_col, temperature_col, memory_free_mib, memory_used_mib, total_energy_consumption]:
    if col not in combined_data.columns:
        raise KeyError(f"Column '{col}' not found in the CSV file.")

# Calculate the difference between the last and first value for each dataset
difference = combined_data.groupby(['data_folder', 'transfer_size'])[total_energy_consumption].last() - combined_data.groupby(['data_folder', 'transfer_size'])[total_energy_consumption].first()
difference = difference.reset_index()

# Marker shapes for each folder
marker_shapes = ['o', 's', 'D', '^']

# Plot the power draw, GPU utilization, and memory utilization in one figure

for j in cols_to_plot:
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 1, 1)
    for folder_index, folder in enumerate(data_folders):
        for source in line_styles.keys():
            transfer_sizes_filtered = []
            max_values = []
            for size in transfer_sizes:
                subset = combined_data[(combined_data['transfer_size'] == size) & 
                                       (combined_data['data_folder'] == folder) & 
                                       (combined_data['source'] == source)]
                if not subset.empty:
                    top_values = subset[j].nlargest(int(len(subset) * 0.09))
                    avg_top_values = top_values.mean()
                    transfer_sizes_filtered.append(size)
                    max_values.append(avg_top_values)
            if max_values:
                plt.plot(transfer_sizes_filtered, max_values, 
                         color=colors[folder_index], 
                         label=f"{folder}_{source}", 
                         marker=marker_shapes[folder_index],
                         markersize=2,  # Adjust the marker size as needed
                         linestyle=line_styles[source], 
                         linewidth=2)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)

    plt.xlabel('Transfer Size (bytes)', fontsize=14)
    plt.ylabel(j, fontsize=14)
    plt.title(j + ' vs. Transfer Size', fontsize=12)
    plt.grid(True, color="black", linewidth="0.5", linestyle="--")
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'interconnect-benchmark-{experiment}-{j}.svg', bbox_inches='tight')
    print(f'interconnect-benchmark-{experiment}-{j}-multinode.svg')

# # Plot for total energy consumption
# plt.subplot(3, 2, 6)
# for folder_index, folder in enumerate(data_folders):
#     for source in line_styles.keys():
#         energy_consumption_diff = []
#         for size in transfer_sizes:
#             subset = combined_data[(combined_data['transfer_size'] == size) & 
#                                    (combined_data['data_folder'] == folder) & 
#                                    (combined_data['source'] == source)]
#             if not subset.empty:
#                 first_value = subset[total_energy_consumption].iloc[0]
#                 last_value = subset[total_energy_consumption].iloc[-1]
#                 energy_diff = last_value - first_value
#                 energy_consumption_diff.append((size, energy_diff))
#         if energy_consumption_diff:
#             sizes, energy_diffs = zip(*energy_consumption_diff)
#             plt.plot(sizes, energy_diffs, 
#                      color=colors[folder_index], 
#                      marker=marker_shapes[folder_index], 
#                      linestyle=line_styles[source], 
#                      label=f"{folder}_{source}",
#                      linewidth=2)

# plt.xlabel('Transfer Size (bytes)')
# plt.ylabel(cols[len(cols)])
# plt.title(cols[len(cols)] + ' vs. Transfer Size')
# plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")
# plt.legend()
# plt.xscale('log')

# # Adding zoomed-in window
# axins = inset_axes(plt.gca(), width="30%", height="30%", loc='upper center', borderpad=2)

# # Plotting within the zoomed-in window
# for folder_index, folder in enumerate(data_folders):
#     for source in line_styles.keys():
#         energy_consumption_diff = []
#         for size in transfer_sizes:
#             subset = combined_data[(combined_data['transfer_size'] == size) & 
#                                    (combined_data['data_folder'] == folder) & 
#                                    (combined_data['source'] == source)]
#             if not subset.empty:
#                 first_value = subset[total_energy_consumption].iloc[0]
#                 last_value = subset[total_energy_consumption].iloc[-1]
#                 energy_diff = last_value - first_value
#                 energy_consumption_diff.append((size, energy_diff))
#         if energy_consumption_diff:
#             sizes, energy_diffs = zip(*energy_consumption_diff)
#             axins.plot(sizes, energy_diffs, 
#                        color=colors[folder_index], 
#                        marker=marker_shapes[folder_index], 
#                        linestyle=line_styles[source], 
#                        label=f"{folder}_{source}",
#                        linewidth=2)

# # Zooming in on a specific region
# x1, x2, y1, y2 = 1e7, 1e9, 0, 0.40  # Adjust these values as needed
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.grid(True, color="grey", linewidth="1.4", linestyle="-.")
# axins.set_xscale('log')

