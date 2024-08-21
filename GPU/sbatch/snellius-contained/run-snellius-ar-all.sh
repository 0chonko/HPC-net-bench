#!/bin/bash
sbatch sbatch/snellius-hybrid/run-snellius-ar-Baseline-halfnode.sh
# sbatch sbatcsnellius-hybrid-snellius-ar-Baseline-wholenode.sh
sbatch sbatch/snellius-hybrid/run-snellius-ar-CudaAware-halfnode.sh
# sbatch sbatcsnellius-hybrid-snellius-ar-CudaAware-wholenode.sh
sbatch sbatch/snellius-hybrid/run-snellius-ar-Nccl-halfnode.sh
# sbatch sbatcsnellius-hybrid-snellius-ar-Nccl-wholenode.sh
sbatch sbatch/snellius-hybrid/run-snellius-ar-Nvlink-halfnode.sh
# sbatch sbatch/snellius/run-snellius-ar-Nvlink-wholenode.sh
