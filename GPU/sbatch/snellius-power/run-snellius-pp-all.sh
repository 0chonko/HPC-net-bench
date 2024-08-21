#!/bin/bash
sbatch --nodes=2 sbatch/snellius-power/run-snellius-pp-Baseline-wholenode.sh
# sbatch  sbatch/snellius-power/run-snellius-pp-Baseline-halfnode.sh
sbatch --nodes=2 sbatch/snellius-power/run-snellius-pp-CudaAware-wholenode.sh
# sbatch  sbatch/snellius-power/run-snellius-pp-CudaAware-halfnode.sh
sbatch --nodes=2 sbatch/snellius-power/run-snellius-pp-Nccl-wholenode.sh
# sbatch  sbatch/snellius-power/run-snellius-pp-Nccl-halfnode.sh
# sbatch sbatch/snellius-power/run-snellius-pp-Nvlink-halfnode.sh
# sbatch sbatch/snellius/run-snellius-pp-Nvlink-wholenode.sh
