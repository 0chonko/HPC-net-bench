import os

data_folders = [
    'ar_nccl', 'a2a_baseline', 'a2a_cudaaware', 'a2a_nvlink',
    'ar_nvlink', 'pp_cudaaware', 'ar_cudaaware', 'ar_baseline',
    'a2a_nccl', 'pp_nvlink', 'pp_baseline', 'pp_nccl'
]

new_headers = [
    "device,ID,DCGM_FI_DEV_POWER_USAGE,DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,"
    "DCGM_FI_DEV_GPU_UTIL,DCGM_FI_DEV_MEM_COPY_UTIL,DCGM_FI_PROF_SM_ACTIVE,"
    "DCGM_FI_PROF_SM_OCCUPANCY,DCGM_FI_PROF_DRAM_ACTIVE,DCGM_FI_DEV_CPU_UTIL_TOTAL\n"
]

# for folder in data_folders:
#     folder_path = os.path.join('sout/native/', folder)
#     for filename in os.listdir(folder_path):
#         if filename.startswith("DCGM") and filename.endswith(".csv"):
#             filepath = os.path.join(folder_path, filename)

#             with open(filepath, 'r') as f:
#                 lines = f.readlines()

#             # Replace the first three lines with new headers
#             cleaned_lines = new_headers + lines[2:]

#             with open(filepath, 'w') as f:
#                 f.writelines(cleaned_lines)

# print("Data cleaning complete!")


for folder in data_folders:
    folder_path = os.path.join('sout/native/', folder)
    for filename in os.listdir(folder_path):
        if filename.startswith("gpuStats") and filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            # print(filepath)
            with open(filepath, 'r') as f:
                lines = f.readlines()

            cleaned_lines = []
            header_found = False

            for i, line in enumerate(lines):
                if i == 0:
                    cleaned_lines.append("timestamp, temperature_gpu, power_draw_w,utilization_gpu, utilization_memory, memory_used_mib, memory_free_mib, total_energy_consumption_j\n")
                else:
                    cleaned_lines.append(line)

            with open(filepath, 'w') as f:
                # print("writing: ", cleaned_lines)
                f.writelines(cleaned_lines)

print("Data cleaning complete!")






# for folder in data_folders:
#     for filename in os.listdir(folder):
#         if filename.endswith(".csv"):
#             filepath = os.path.join(folder, filename)

#             with open(filepath, 'r') as f:
#                 lines = f.readlines()

#             cleaned_lines = []
#             header_found = False 

#             for line in lines:
#                 if line.strip() == header_line:
#                     if not header_found:  # Add header only if it's the first one
#                         cleaned_lines.append(line)
#                         header_found = True
#                 else:
#                     cleaned_lines.append(line)

#             with open(filepath, 'w') as f:
#                 f.writelines(cleaned_lines)