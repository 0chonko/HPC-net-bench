#!/bin/bash
sbatch --nodes=2 sbatch/snellius-power/run-snellius-ar-Baseline-wholenode.sh
# sbatch sbatch/snellius-power/run-snellius-ar-Baseline-halfnode.sh
sbatch --nodes=2 sbatch/snellius-power/run-snellius-ar-CudaAware-wholenode.sh
# sbatch sbatch/snellius-power/run-snellius-ar-CudaAware-halfnode.sh
sbatch --nodes=2 sbatch/snellius-power/run-snellius-ar-Nccl-wholenode.sh
# sbatch sbatch/snellius-power/run-snellius-ar-Nccl-halfnode.sh
# sbatch sbatch/snellius-power/run-snellius-ar-Nvlink-halfnode.sh
# sbatch sbatch/snellius/run-snellius-ar-Nvlink-halfnode.sh
