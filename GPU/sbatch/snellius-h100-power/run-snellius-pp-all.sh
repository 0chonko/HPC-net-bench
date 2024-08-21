#!/bin/bash
sbatch sbatch/snellius-h100-power/run-snellius-pp-Baseline-halfnode.sh
#sbatch --nodes=2 sbatch/snellius-h100-power/run-snellius-pp-Baseline-wholenode.sh
sbatch sbatch/snellius-h100-power/run-snellius-pp-CudaAware-halfnode.sh
#sbatch --nodes=2 sbatch/snellius-h100-power/run-snellius-pp-CudaAware-wholenode.sh
sbatch sbatch/snellius-h100-power/run-snellius-pp-Nccl-halfnode.sh
#sbatch --nodes=2 sbatch/snellius-h100-power/run-snellius-pp-Nccl-wholenode.sh
sbatch sbatch/snellius-h100-power/run-snellius-pp-Nvlink-halfnode.sh
# sbatch sbatch/snellius-h100-power/run-snellius-pp-Nvlink-wholenode.sh
