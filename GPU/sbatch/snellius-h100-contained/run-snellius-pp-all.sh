#!/bin/bash
sbatch sbatch/snellius-hybrid/run-snellius-pp-Baseline-halfnode.sh
# sbatch sbatcsnellius-hybrid-snellius-pp-Baseline-wholenode.sh
sbatch sbatch/snellius-hybrid/run-snellius-pp-CudaAware-halfnode.sh
# sbatch sbatcsnellius-hybrid-snellius-pp-CudaAware-wholenode.sh
sbatch sbatch/snellius-hybrid/run-snellius-pp-Nccl-halfnode.sh
# sbatch sbatcsnellius-hybrid-snellius-pp-Nccl-wholenode.sh
sbatch sbatch/snellius-hybrid/run-snellius-pp-Nvlink-halfnode.sh
# sbatch sbatch/snellius/run-snellius-pp-Nvlink-wholenode.sh
