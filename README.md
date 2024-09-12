# HPC-net-bench

The repository utilizes the cloud_noise and the interconnect benchmarks:

https://github.com/DanieleDeSensi/cloud_noise
https://github.com/HicrestLaboratory/interconnect-benchmark

---
#### Hardware Requirements:
- AMD Rome or Genoa CPUs, NVIDIA A100 or H100 GPUs.
- Infiniband/NVLinks for inter/intra-node communication.

#### Software Requirements:
- RHEL8, OpenMPI 4.1.5, CUDA 12.1, Driver version 545.
- Apptainer or Singularity for containerized environments.
- DCGM for power profiling.

---

## CPU Benchmarking

- Follow instructions from [de_sensi_noise_2022] to build the project

- Build the container with apptainer locally (with sudo) or on the cbuild partition (with --fakeroot)

```
apptainer build --fakeroot images/hybrid_image.sif definitions/definition_hybrid.def
apptainer build --fakeroot images/contained_image.sif definitions/definition_contained.def
```

- Run scripts from `scripts-native`, `scripts-hybrid`, and `scripts-spack` to gather data



---
## GPU Benchmarking

- Follow instructions from [interconnect-benchmark].
- Build GPU benchmarks using the following:
    ```bash
    make -f Makefile.SNELLIUS-POWER
    ```
- Build the container with apptainer locally (with sudo) or on the cbuild partition (with --fakeroot)
```
apptainer build --fakeroot containers/images/hybrid_image.sif containers/definitions/definition_hybrid.def
apptainer build --fakeroot containers/images/contained_image.sif containers/definitions/definition_contained.def
```

- Run experiments from the `sbatch` directory:
    ```bash
    ./sbatch/snellius-h100-hybrid/run-snellius-pp-all.sh
    ```
    Modify the `*all.sh` files to suit single-node or multi-node runs (halfnode.sh or wholenode.sh).

- The profiler files are located in ```include``` directory under *nvmlClass* and *dcgmiLogger*. Here the profiled fields can be modified. The ```-p 1``` flag can be used to collect NVML metrics, while ```-p 2``` is used to run DCGM profiler metrics. If the flag is ommited, no profiler is initiated. NOTE: these need to be modified in the sbatch files accordingly  


---

### Plotting Results

#### CPU:
- Use `plot.py` to generate latency and bandwidth plots by enabling/disabling specific methods at the bottom of the file.

#### GPU:
- Use `plotPowerDraw.py` for power metrics, and `plotBandwidthMultiPath.py` for bandwidth comparisons between native and hybrid runs. For example:
    ```python
    filename1 = "sout/native-h100"
    filename2 = "sout/hybrid-h100"
    ```
