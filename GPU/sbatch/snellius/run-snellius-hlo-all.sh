#!/bin/bash
sbatch sbatch/snellius/run-snellius-hlo-Baseline-halfnode.sh
sbatch sbatch/snellius/run-snellius-hlo-Baseline-wholenode.sh
sbatch sbatch/snellius/run-snellius-hlo-CudaAware-halfnode.sh
sbatch sbatch/snellius/run-snellius-hlo-CudaAware-wholenode.sh
sbatch sbatch/snellius/run-snellius-hlo-Nccl-halfnode.sh
sbatch sbatch/snellius/run-snellius-hlo-Nccl-wholenode.sh
