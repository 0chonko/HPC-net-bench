#!/bin/bash
#SBATCH --job-name=run_short_genoa_diffrack_spack
#SBATCH --output=run_short_genoa_diffrack_spack.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --partition=genoa
#SBATCH --time=01:40:00
#SBATCH --exclusive
#SBATCH --nodelist=tcn[830,958]

export FI_LOG_LEVEL=info
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_plm=^slurm
export UCX_POSIX_USE_PROC_LINK=n


# export OMPI_MCA_ess=pmi2
# export LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/ucx-ohpc/1.11.2/lib:/opt/ohpc/pub/mpi/libfabric/1.13.0/lib:/opt/ohpc/pub/mpi/ucx-ohpc/1.11.2/lib:$LD_LIBRARY_PATH

export OMPI_MCA_btl_tcp_if_include=eno2np0,ib0

cd ../ && ./run_spack.sh -k "snellius-short-genoa-spack,HPC,Different Racks,Day" 