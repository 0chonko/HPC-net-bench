#!/bin/bash

#SBATCH --job-name=pp_CudaAware_halfnode
#SBATCH --output=sout/snellius_pp_CudaAware_halfnode_%j.out
#SBATCH --error=sout/snellius_pp_CudaAware_halfnode_%j.err

#SBATCH --partition=gpu
#SBATCH --account=vusei7310
#SBATCH --time=00:05:00
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

#SBATCH --requeue

nvidia-smi topo -m > output.txt
