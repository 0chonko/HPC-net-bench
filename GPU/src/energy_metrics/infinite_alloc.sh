#!/bin/bash

#SBATCH --job-name=pp_CudaAware_halfnode
#SBATCH --output=sout/snellius_pp_CudaAware_halfnode_energy_inf%j.out
#SBATCH --error=sout/snellius_pp_CudaAware_halfnode__energy_inf%j.err

#SBATCH --partition=gpu
#SBATCH --account=vusei7310
#SBATCH --time=08:05:00
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

#SBATCH --requeue

sleep infinity