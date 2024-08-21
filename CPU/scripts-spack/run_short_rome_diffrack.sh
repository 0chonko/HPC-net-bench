#!/bin/bash
#SBATCH --job-name=run_short_rome_diffrack_spack
#SBATCH --output=run_short_rome_diffrack_spack.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --partition=rome
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --nodelist=tcn[469,433]


# export FI_LOG_LEVEL=info
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_plm=^slurm
export UCX_POSIX_USE_PROC_LINK=n


export OMPI_MCA_btl_tcp_if_include=ens4f0np0,ib0

cd ../ && ./run_spack.sh -k "snellius-short-rome,HPC,Different Racks,Day"
