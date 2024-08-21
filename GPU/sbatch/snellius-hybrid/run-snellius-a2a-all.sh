#!/bin/bash
# sbatch sbatch/snellius-hybrid/run-snellius-a2a-Baseline-halfnode.sh


sbatch --nodes=2 sbatch/snellius-hybrid/run-snellius-a2a-Baseline-wholenode.sh
# sbatch sbatch/snellius-hybrid/run-snellius-a2a-CudaAware-halfnode.sh
sbatch --nodes=2 sbatch/snellius-hybrid/run-snellius-a2a-CudaAware-wholenode.sh
# sbatch sbatch/snellius-hybrid/run-snellius-a2a-Nccl-halfnode.sh
sbatch --nodes=2 sbatch/snellius-hybrid/run-snellius-a2a-Nccl-wholenode.sh
# sbatch sbatch/snellius-hybrid/run-snellius-a2a-Nvlink-halfnode.sh
# sbatch sbatch/snellius-power/run-snellius-a2a-Nvlink-wholenode.sh

