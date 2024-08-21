#!/bin/bash
sbatch sbatch/snellius/run-snellius-mpp-Baseline-wholenode.sh
sbatch sbatch/snellius/run-snellius-mpp-CudaAware-wholenode.sh
sbatch sbatch/snellius/run-snellius-mpp-Nccl-wholenode.sh
sbatch sbatch/snellius/run-snellius-mpp-Aggregated-wholenode.sh
