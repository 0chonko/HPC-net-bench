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
transfer_sizes = [2**i for i in range(1,30)]

# Calculate MiB from transfer_sizes
transfer_sizes_mib = [size / (2**20) for size in transfer_sizes]
# List of data folders
# data_folders = ['pp_nvlink', 'pp_nccl', 'pp_baseline', 'pp_cudaaware']
# data_folders = ['a2a_nccl', 'a2a_nvlink', 'a2a_baseline', 'a2a_cudaaware']
sns.set_style("whitegrid")


experiments = ['ar', 'a2a', 'pp']
# experiments = ['pp']
# experiment = 'pp'
architecture = 'a100'
# folders = ['sout/native_h100','sout/hybrid_h100', 'sout/contained/power_data']
folders = ['sout/native/bu3_GOOD PREVIOUSLY','sout/hybrid', 'sout/contained/power_data']
# folders = ['sout/hybrid', 'sout/contained/power_data']




# Colors for each data folder
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
# label_colors = {'Baseline': '#1f77b4', 'CudaAware': '#ff7f0e', 'Nccl': '#2ca02c', 'Nvlink': '#d62728', 'Aggregated': '#9467bd'}

# Line styles for each source
line_styles = {'Native': '-', 'Hybrid': '--'}

LOAD_PARTIAL = True

def lighten_color(color, factor=0.5):
    """Lighten a given color by a factor."""
    color = mcolors.to_rgb(color)
    return mcolors.to_hex([min(1, c + (1 - c) * factor) for c in color])

def hr_size(size):
    if size < 1024:
        return str(int(size)) + "B"
    elif size < 1024*1024:
        return str(int(size / 1024)) + "KiB"
    elif size < 1024*1024*1024:
        return str(int(size / (1024*1024))) + "MiB"
    elif size < 1024*1024*1024*1024:
        return str(int(size / (1024*1024*1024))) + "GiB"


# for i in [0, 3]:
for experiment in experiments:
    data_folders = [f'{experiment}_nccl', f'{experiment}_nvlink', f'{experiment}_baseline', f'{experiment}_cudaaware']
    for i in [0,3]:
        # Load data from main folder
        for folder in data_folders:
            for size in transfer_sizes:
                file_path = f'{folders[0]}/{folder}/gpuStats_{i}_{size}.csv'
                if os.path.exists(file_path):
                    if folder == 'ar_nccl' and i == 0 and size == 268435456:
                            continue
                    # if experiment == 'pp' and i != 0 or i != 3:
                    #     continue
                    if LOAD_PARTIAL:
                        data = pd.read_csv(file_path, nrows=int(0.1 * len(pd.read_csv(file_path))))
                    else:
                        data = pd.read_csv(file_path)
                    data['transfer_size'] = hr_size(size)  # Add a column for transfer size
                    data['data_folder'] = folder  # Add a column for data folder
                    data['source'] = 'Native'  # Add a column for source
                    combined_data = pd.concat([combined_data, data], axis=0)

                else:
                    print(f'File not found: {file_path}')

        # Load data from additional folder
        for folder in data_folders:
            for size in transfer_sizes:
                file_path = f'{folders[1]}/{folder}/gpuStats_{i}_{size}.csv'
                if os.path.exists(file_path):
                    # print(file_path)
                    if folder == 'a2a_baseline' and i == 3 and size == 536870912:
                            continue
                    data = pd.read_csv(file_path)
                    data['transfer_size'] = hr_size(size)  # Add a column for transfer size
                    data['data_folder'] = folder  # Differentiate from main folder
                    data['source'] = 'Hybrid'  # Add a column for source
                    combined_data = pd.concat([combined_data, data], axis=0)
                # else:
                    # print(f'File not found: {file_path}')

    # Custom legend labels
    folder_labels = {
        f'{experiment}_nccl': 'NCCL',
        f'{experiment}_nvlink': 'CUDA IPC',
        f'{experiment}_baseline': 'TrivialStaging',
        f'{experiment}_cudaaware': 'CUDA-Aware MPI'
    }
    # Verify columns in the combined data

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


    col_labels = {
        timestamp_col: 'Timestamp',
        power_draw_col: 'Power Draw (W)',
        gpu_util_col: 'GPU Utilization (%)',
        mem_util_col: 'Memory Utilization (%)',
        temperature_col: 'Temperature (°C)',
        memory_free_mib: 'Memory Free (MiB)',
        memory_used_mib: 'Memory Used (MiB)',
        total_energy_consumption: 'Total Energy Consumption (J)'
    }

    cols_to_plot= [power_draw_col, gpu_util_col, mem_util_col]





    # Check if the columns exist
    for col in [timestamp_col, power_draw_col, gpu_util_col, mem_util_col, temperature_col, memory_free_mib, memory_used_mib, total_energy_consumption]:
        if col not in combined_data.columns:
            raise KeyError(f"Column '{col}' not found in the CSV file.")

    # Calculate the difference between the last and first value for each dataset
    difference = combined_data.groupby(['data_folder', 'transfer_size'])[total_energy_consumption].last() - combined_data.groupby(['data_folder', 'transfer_size'])[total_energy_consumption].first()
    difference = difference.reset_index()

    # Marker shapes for each folder
    marker_shapes = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'X']

    # Plot the power draw, GPU utilization, and memory utilization in one figure

    for j in cols_to_plot:
        plt.figure(figsize=(2, 2))
        
        for folder_index, folder in enumerate(data_folders):
            for source in line_styles.keys():
                transfer_sizes_filtered = []
                percentage_differences = []
                max_values_native = []
                max_values_hybrid = []
                iqr_low_native = []
                iqr_high_native = []
                iqr_low_hybrid = []
                iqr_high_hybrid = []
                
                for size, size_mib in zip(transfer_sizes, transfer_sizes_mib):
                    subset_native = combined_data[(combined_data['transfer_size'] == hr_size(size)) & 
                                        (combined_data['data_folder'] == folder) & 
                                        (combined_data['source'] == 'Native')]

                    subset_hybrid = combined_data[(combined_data['transfer_size'] == hr_size(size)) &   
                                        (combined_data['data_folder'] == folder) &
                                        (combined_data['source'] == 'Hybrid')]

                    
                    
                    if not subset_native.empty:
                        top_values_native = subset_native[j].nlargest(int(len(subset_native) * 1))
                        avg_top_values_native = top_values_native.median()
                        transfer_sizes_filtered.append(hr_size(size))
                        max_values_native.append(avg_top_values_native)
                        top_values_hybrid = subset_hybrid[j].nlargest(int(len(subset_hybrid) * 1))
                        avg_top_values_hybrid = top_values_hybrid.median()
                        max_values_hybrid.append(avg_top_values_hybrid)
                        # print(avg_top_values_native, avg_top_values_hybrid)
                        percentage_diff = ((avg_top_values_hybrid - avg_top_values_native) / avg_top_values_native) * 100
                        percentage_differences.append(percentage_diff)
  
                        lower_quartile_native = pd.Series(top_values_native).quantile(0.25)
                        upper_quartile_native = pd.Series(top_values_native).quantile(0.75)
                        lower_quartile_hybrid = pd.Series(top_values_hybrid).quantile(0.25)
                        upper_quartile_hybrid = pd.Series(top_values_hybrid).quantile(0.75)

                        iqr_low_native.append(lower_quartile_native)
                        iqr_high_native.append(upper_quartile_native)
                        iqr_low_hybrid.append(lower_quartile_hybrid)
                        iqr_high_hybrid.append(upper_quartile_hybrid)

                        # Calculate the difference of IQRs between Hybrid and Native
                        iqr_high_diff = np.array(iqr_high_hybrid) - np.array(iqr_high_native)

                        # Convert the difference to percentage
                        percentage_high_diff = (iqr_high_diff / np.array(iqr_high_native)) * 100

                        # Calculate the difference of IQRs between Hybrid and Native
                        iqr_low_diff = np.array(iqr_low_hybrid) - np.array(iqr_low_native)

                        # Convert the difference to percentage
                        percentage_low_diff = (iqr_low_diff / np.array(iqr_low_native)) * 100

                
                # print(percentage_differences)
                if source == 'Hybrid':
                    color = colors[folder_index] if source != 'Hybrid' else lighten_color(colors[folder_index], 0.3)
                    linestyle = '-' if source == 'Hybrid' else '-'
                    sns.lineplot(x=transfer_sizes_filtered[::-6], y=percentage_differences[::-6], 
                                 color=color, 
                                 label=f"{folder_labels[folder]} ({source})", 
                                 marker=marker_shapes[folder_index],
                                 markersize=8,  # Adjust the marker size as needed
                                 sort=True,  # Disable sorting of x-axis values
                                 linestyle=linestyle, 
                                 linewidth=1)
                    # Add error bars using Matplotlib
                    print(transfer_sizes_filtered[::-6])
                    print(transfer_sizes_filtered)
                    plt.errorbar(x=transfer_sizes_filtered[::-6], 
                                y=percentage_differences[::-6], 
                                yerr=[np.abs(percentage_low_diff[::-6]), np.abs(percentage_high_diff[::-6])], 
                                fmt='none',  # 'fmt' is set to 'none' to avoid re-plotting data points
                                ecolor=color, 
                                capsize=3)  # Adjust capsize as needed

                    



                    # plt.xticks(transfer_sizes_filtered, [f'{x:.1f} MiB' for x in transfer_sizes_mib], fontsize=12, rotation=45)
                    plt.yticks(fontsize=9)
                    plt.xticks(fontsize=7.5)
                    # Set xticks and labels
                    # xticks = [transfer_sizes_filtered[0], transfer_sizes_filtered[len(transfer_sizes_filtered)//2], transfer_sizes_filtered[-1]]
                    # xticklabels = [f'{x:.1f} MiB' for x in xticks]
        
        # plt.ylim(-1, 1)  # Adjust as needed
        plt.title(f'{experiment.upper()} - {col_labels[j]}', fontsize=10)
        plt.xlabel('Message Size', fontsize=10)
        
        # plt.ylabel(col_labels[j] + ' difference(%)', fontsize=10)
        # plt.legend()  # Ensure the legend is displayed
        plt.legend().remove()




        # SHOW ONLY LEGEND
        ############################
        # plt.gca().set_frame_on(False)
        # plt.gca().axes.xaxis.set_visible(False)
        # plt.gca().axes.yaxis.set_visible(False)

        # legend = plt.legend(ncol=4)
        # legend.get_frame().set_alpha(None)
        # legend.get_frame().set_facecolor((1, 1, 1, 1))
        ############################

        plt.grid(True, linestyle='--')
        # if j == cols_to_plot[0]:
            # plt.legend()
        # plt.xscale('log')
            # plt.yscale('log')

        plt.minorticks_off()
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        plt.tight_layout()
        # plt.show()  # Display each figure separately

        filename = f'interconnect-benchmark-{experiment}-{j.strip()}-{architecture}.pdf'
        plt.savefig(filename, bbox_inches='tight', format='pdf', dpi=300)
        # print(filename)



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

