import os
import pandas as pd
import matplotlib.pyplot as plt

# Initialize an empty DataFrame to hold all data
combined_data = pd.DataFrame()

# List of transfer sizes (1, 2, 4, ..., 536870912)
transfer_sizes = [2**29]

# List of data folders
data_folders = ['a2a_nccl']

# Additional data folder
folders = ['sout','sout/hybrid/power_data', 'sout/contained/power_data']

# Colors for each data folder
colors = ['#1f77b4', '#ff7f0e', '#d62728']

# Line styles for each data source
line_styles = {'sout': '-', 'hybrid': '--', 'contained' : '-.'}

# Load data from main folder
for folder in data_folders:
    for size in transfer_sizes:
        file_path = f'sout/{folder}/gpuStats_0_{size}.csv'
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data['transfer_size'] = size  # Add a column for transfer size
            data['data_folder'] = folder  # Add a column for data folder
            data['source'] = 'sout'  # Add a column for source
            combined_data = pd.concat([combined_data, data], axis=0)

# Load data from additional folder
for folder in data_folders:
    for size in transfer_sizes:
        file_path = f'{additional_folder}/{folder}/gpuStats_0_{size}.csv'
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data['transfer_size'] = size  # Add a column for transfer size
            data['data_folder'] = folder + '_hybrid'  # Differentiate from main folder
            data['source'] = 'hybrid'  # Add a column for source
            combined_data = pd.concat([combined_data, data], axis=0)

# Verify columns in the combined data
print("Column names in the combined data:", combined_data.columns)

# Replace with the actual column names if they are different
timestamp_col = 'timestamp'
power_draw_col = ' power_draw_w'
gpu_util_col = ' utilization_gpu'
mem_util_col = ' utilization_memory'
clocks_throttle_reasons_active = ' temperature_gpu'
memory_free_mib = ' memory_free_mib'
memory_used_mib = ' memory_used_mib'
total_energy_consumption = ' total_energy_consumption_j'

# Array of the above columns
cols = [timestamp_col, power_draw_col, gpu_util_col, mem_util_col, memory_used_mib, clocks_throttle_reasons_active, memory_free_mib, total_energy_consumption]

# Find the minimum timestamp of all data
min_timestamp = combined_data[timestamp_col].min()
print("Minimum timestamp:", min_timestamp)

# Find the minimum timestamp for each data folder
min_timestamps = combined_data.groupby('data_folder')[timestamp_col].min()
print("Minimum timestamps for each data folder:")
print(min_timestamps)

# Subtract the minimum timestamp from all timestamps
for folder in combined_data['data_folder'].unique():
    folder_min_timestamp = min_timestamps[folder]
    combined_data.loc[combined_data['data_folder'] == folder, timestamp_col] -= (folder_min_timestamp - min_timestamp)

# Subtract the global minimum timestamp and convert to milliseconds
global_min_timestamp = combined_data[timestamp_col].min()
combined_data[timestamp_col] = (combined_data[timestamp_col] - global_min_timestamp) / 1e6

# Check if the columns exist
for col in [timestamp_col, power_draw_col, gpu_util_col, mem_util_col, clocks_throttle_reasons_active, memory_free_mib, memory_used_mib]:
    if col not in combined_data.columns:
        raise KeyError(f"Column '{col}' not found in the CSV file.")

# Marker shapes for each folder
marker_shapes = ['o', 's', 'D', '^']  # Circle, Square, Diamond, Triangle

# Plot the power draw, GPU utilization, and memory utilization in one figure
plt.figure(figsize=(20, 15))

for i in range(1, 5):
    # Subplot 1: Power Draw
    plt.subplot(3, 2, i)
    for folder_index, folder in enumerate(combined_data['data_folder'].unique()):
        for size in transfer_sizes:
            subset = combined_data[(combined_data['transfer_size'] == size) & (combined_data['data_folder'] == folder)].sort_values(by=timestamp_col)
            line_style = line_styles['sout'] if 'hybrid' not in folder else line_styles['hybrid']
            plt.plot(subset[timestamp_col], subset[cols[i]], 
                        color=colors[folder_index % len(colors)], 
                        linestyle=line_style,
                        label=folder)

    plt.xlabel('Time(ms)')
    plt.ylabel(cols[i])
    plt.title(cols[i] + ' vs. Transfer Size')
    plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")
    plt.legend()
    plt.show()

    max_value = combined_data[cols[i]].max()
    # Connect the max value with a horizontal line
    plt.axhline(y=max_value, color='red', linestyle='--')

# Adjust the layout to avoid overlapping labels
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig('interconnect-benchmark-energy.png')

# Show the figure
plt.show()
