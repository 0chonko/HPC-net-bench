#!/bin/bash
sbatch sbatch/snellius-h100-hybrid/run-snellius-ar-Baseline-halfnode.sh
# sbatch --nodes=2  sbatch/snellius-h100-hybrid/run-snellius-ar-Baseline-wholenode.sh
sbatch sbatch/snellius-h100-hybrid/run-snellius-ar-CudaAware-halfnode.sh
# sbatch --nodes=2  sbatch/snellius-h100-hybrid/run-snellius-ar-CudaAware-wholenode.sh
sbatch sbatch/snellius-h100-hybrid/run-snellius-ar-Nccl-halfnode.sh
# sbatch --nodes=2 sbatch/snellius-h100-hybrid/run-snellius-ar-Nccl-wholenode.sh
sbatch sbatch/snellius-h100-hybrid/run-snellius-ar-Nvlink-halfnode.sh
# sbatch sbatch/snellius/run-snellius-ar-Nvlink-wholenode.sh
