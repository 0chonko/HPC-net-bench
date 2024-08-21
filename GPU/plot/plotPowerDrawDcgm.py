import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from io import StringIO
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np



# Initialize an empty DataFrame to hold all data
combined_data = pd.DataFrame()

# List of transfer sizes (1, 2, 4, ..., 536870912)
transfer_sizes = [2**i for i in range(20,30)]


experiment = 'a2a'
data_folders = [f'{experiment}_nccl', f'{experiment}_nvlink', f'{experiment}_baseline', f'{experiment}_cudaaware']
folders = ['sout/native-2-nodes_h100','sout/hybrid-2-nodes-h100', 'sout/contained/power_data']

# Colors for each data folder
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Line styles for each source
line_styles = {'Native': '-', 'Hybrid': '--'}



# Load data from main folder
for path in folders[:2]:
    for j in [0, 3]:
        for folder in data_folders:
            for size in transfer_sizes:
                file_path = f'{path}/{folder}/DCGMI_gpuStats_{j}_{size}.csv'
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path, header=0, names=['#Entity', 'ID', 'SMACT', 'SMOCC', 'DRAMA', 'PCITX', 'PCIRX', 'NVLTX', 'NVLRX', 'extra'])
                        
                    # Drop the extra column
                    data = data.drop(columns=['extra'])                # Adjust columns to include ID and correct offset
                    data['transfer_size'] = size  # Add a column for transfer size
                    data['data_folder'] = folder  # Add a column for data folder
                    data['source'] = 'Native' if "native" in path else 'Hybrid'  # Add a column for source
                    combined_data = pd.concat([combined_data, data], axis=0)


print("finished loading data")
# Load data from additional folder
# for folder in data_folders:
#     for size in transfer_sizes:
#         file_path = f'{folders[1]}/{folder}/DCGMI_gpuStats_3_{size}.csv'
#         if os.path.exists(file_path):
#             data = pd.read_csv(file_path)
#             data['transfer_size'] = size  # Add a column for transfer size
#             data['data_folder'] = folder  # Differentiate from main folder
#             data['source'] = 'hybrid'  # Add a column for source
#             combined_data = pd.concat([combined_data, data], axis=0)

# Verify columns in the combined data
print("Column names in the combined data:", combined_data.columns)


# Column names
device_col = '#Entity'
id_col = 'ID'
sm_active_col = 'SMACT'
sm_occupancy_col = 'SMOCC'
dram_active_col = 'DRAMA'
gpu_util_col = 'PCITX'
mem_copy_util_col = 'PCIRX'
nvl_tx_col = 'NVLTX'
nvl_rx_col = 'NVLRX'

cols = [device_col, id_col, sm_active_col, sm_occupancy_col, dram_active_col, gpu_util_col, mem_copy_util_col, nvl_tx_col, nvl_rx_col]



def lighten_color(color, factor=0.5):
    """Lighten a given color by a factor."""
    color = mcolors.to_rgb(color)
    return mcolors.to_hex([min(1, c + (1 - c) * factor) for c in color])


# cols_to_plot= [sm_active_col, sm_occupancy_col, mem_copy_util_col]
# cols_to_plot= [sm_active_col]
# cols_to_plot= [sm_occupancy_col]
# cols_to_plot= [mem_copy_util_col]
cols_to_plot= [sm_active_col, sm_occupancy_col, dram_active_col, gpu_util_col, mem_copy_util_col, nvl_tx_col, nvl_rx_col]



# Check if the columns exist
for col in [device_col, id_col, sm_active_col, sm_occupancy_col, dram_active_col, gpu_util_col, mem_copy_util_col, nvl_tx_col, nvl_rx_col]:
    if col not in combined_data.columns:
        raise KeyError(f"Column '{col}' not found in the CSV file.")

# Calculate the difference between the last and first value for each dataset
# difference = combined_data.groupby(['data_folder', 'transfer_size'])[total_energy_consumption].last() - combined_data.groupby(['data_folder', 'transfer_size'])[total_energy_consumption].first()
# difference = difference.reset_index()

# Marker shapes for each folder
marker_shapes = ['o', 's', 'D', '^']


folder_labels = {
    f'{experiment}_nccl': 'NCCL',
    f'{experiment}_nvlink': 'TrivialStaging',
    f'{experiment}_baseline': 'CUDA IPC',
    f'{experiment}_cudaaware': 'CUDA-Aware MPI'
}
# Plot the power draw, GPU utilization, and memory utilization in one figure

for j in cols_to_plot:

    plt.figure(figsize=(2.5, 4))

    plt.subplot(1, 1, 1)
    for folder_index, folder in enumerate(data_folders):
        for source in line_styles.keys():
            transfer_sizes_filtered = []
            max_values = []
            lower_bounds = []
            upper_bounds = []
            for size in transfer_sizes:
                subset = combined_data[(combined_data['transfer_size'] == size) & 
                                    (combined_data['data_folder'] == folder) & 
                                    (combined_data['source'] == source)]
                if not subset.empty:
                    top_values = subset[j].apply(pd.to_numeric, errors='coerce').nlargest(int(len(subset) * 0.06))
                    avg_top_values = top_values.mean()
                    transfer_sizes_filtered.append(size)
                    max_values.append(avg_top_values)

                 # Calculate IQR for the current size
                q1 = np.percentile(top_values, 25)
                q3 = np.percentile(top_values, 75)
                iqr = q3 - q1
                
                # Define the lower and upper bounds based on the IQR
                lower_bound = max(q1 - iqr, 0)  # Ensure the lower bound is not below 0
                upper_bound = q3 + iqr
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)

            if max_values:
                color = colors[folder_index] if source != 'Hybrid' else lighten_color(colors[folder_index], 0.3)
                linestyle = '--' if source == 'Hybrid' else '-'
                sns.lineplot(x=transfer_sizes_filtered, y=max_values, 
                        color=color, 
                         label=f"{folder_labels[folder]} ({source})", 
                         marker='.',
                         markersize=12,  # Adjust the marker size as needed
                         linestyle=linestyle, 
                         linewidth=2.2,
                         errorbar='sd')
                plt.xlim(1e6, 1e9)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.fill_between(transfer_sizes_filtered, lower_bounds, upper_bounds, color=color, alpha=0.2)

    plt.xlabel('Transfer Size (bytes)', fontsize=10)
    plt.ylabel(j, fontsize=10)
    # plt.title(j + ' vs. Transfer Size', fontsize=12)
    plt.grid()
    plt.legend()
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig(f'interconnect-benchmark-{experiment}-{j}.svg', bbox_inches='tight')
    print(f'interconnect-benchmark-{experiment}-{j}.svg')


# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Initialize an empty DataFrame to hold all data
# combined_data = pd.DataFrame()

# # List of transfer sizes (1, 2, 4, ..., 536870912)
# transfer_sizes = [2**i for i in range(30)]

# # List of data folders
# data_folders = ['ar_nccl', 'ar_nvlink', 'ar_baseline', 'ar_cudaaware']
# folders = ['sout/native','sout/hybrid/power_data', 'sout/contained/power_data']

# # Colors for each data folder
# colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

# # Line styles for each source
# line_styles = {'sout': '-', 'hybrid': '--'}

# # Load data from main folder
# for folder in data_folders:
#     for size in transfer_sizes:
#         file_path = f'{folders[0]}/{folder}/gpuStats_3_{size}.csv'
#         if os.path.exists(file_path):
#             data = pd.read_csv(file_path)
#             data['transfer_size'] = size  # Add a column for transfer size
#             data['data_folder'] = folder  # Add a column for data folder
#             data['source'] = 'sout'  # Add a column for source
#             combined_data = pd.concat([combined_data, data], axis=0)

# # Load data from additional folder
# for folder in data_folders:
#     for size in transfer_sizes:
#         file_path = f'{folders[1]}/{folder}/gpuStats_0_{size}.csv'
#         if os.path.exists(file_path):
#             data = pd.read_csv(file_path)
#             data['transfer_size'] = size  # Add a column for transfer size
#             data['data_folder'] = folder  # Differentiate from main folder
#             data['source'] = 'hybrid'  # Add a column for source
#             combined_data = pd.concat([combined_data, data], axis=0)

# # Verify columns in the combined data
# print("Column names in the combined data:", combined_data.columns)

# # Column names
# timestamp_col = 'timestamp'
# power_draw_col = 'power_draw_w'
# gpu_util_col = 'utilization_gpu'
# mem_util_col = 'utilization_memory'
# clocks_throttle_reasons_active = 'clocks_throttle_reasons_active'
# memory_free_mib = 'memory_free_mib'
# memory_used_mib  = 'memory_used_mib'
# total_energy_consumption = 'total_energy_consumption_j'
# cols = [timestamp_col, power_draw_col, gpu_util_col, mem_util_col, clocks_throttle_reasons_active, memory_free_mib, memory_used_mib, total_energy_consumption]
# cols_to_plot= [power_draw_col, memory_used_mib]

# # Check if the columns exist
# for col in cols:
#     if col not in combined_data.columns:
#         raise KeyError(f"Column '{col}' not found in the CSV file.")

# # Calculate the difference between the last and first value for each dataset
# difference = combined_data.groupby(['data_folder', 'transfer_size'])[total_energy_consumption].last() - combined_data.groupby(['data_folder', 'transfer_size'])[total_energy_consumption].first()
# difference = difference.reset_index()

# # Marker shapes for each folder
# marker_shapes = ['o', 's', 'D', '^']

# # Plot the power draw, GPU utilization, and memory utilization in one figure
# plt.figure(figsize=(9, 6))

# for i in range(len(cols_to_plot)):
#     plt.subplot(2, 1, i+1)
#     for folder_index, folder in enumerate(data_folders):
#         for source in line_styles.keys():
#             transfer_sizes_filtered = []
#             max_values = []
#             for size in transfer_sizes:
#                 subset = combined_data[(combined_data['transfer_size'] == size) & 
#                                        (combined_data['data_folder'] == folder) & 
#                                        (combined_data['source'] == source)]
#                 if not subset.empty:
#                     top_values = subset[cols_to_plot[i]].nlargest(int(len(subset) * 0.05))
#                     avg_top_values = top_values.mean()
#                     transfer_sizes_filtered.append(size)
#                     max_values.append(avg_top_values)
#             if max_values:
#                 plt.plot(transfer_sizes_filtered, max_values, 
#                          color=colors[folder_index], 
#                          label=f"{folder}_{source}", 
#                          marker=marker_shapes[folder_index],
#                          markersize=2,  # Adjust the marker size as needed
#                          linestyle=line_styles[source], 
#                          linewidth=2)
#                 plt.xticks(fontsize=12)
#                 plt.yticks(fontsize=12)

#     plt.xlabel('Transfer Size (bytes)', fontsize=14)
#     plt.ylabel(cols_to_plot[i], fontsize=14)
#     plt.title(cols_to_plot[i] + ' vs. Transfer Size', fontsize=10)
#     plt.grid(True, color="black", linewidth="0.8", linestyle="-.")
#     plt.legend()
#     plt.xscale('log')

# plt.tight_layout()
# plt.show()


# plt.savefig('sout/DCGMI_StatsGPU3.png')
