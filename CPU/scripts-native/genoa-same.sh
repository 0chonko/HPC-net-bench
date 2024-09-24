#!/bin/bash
#SBATCH --job-name=genoa-same
#SBATCH --output=genoa-same.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --partition=genoa
#SBATCH --time=02:30:00
#SBATCH --exclusive

export OMPI_DIR=/opt/ompi
export APPTAINER_OMPI_DIR=$OMPI_DIR
export APPTAINERENV_APPEND_PATH=$OMPI_DIR/bin
export APPTAINERENV_APPEND_LD_LIBRARY_PATH=$OMPI_DIR/lib
module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_btl_vader_single_copy_mechanism=none
OMPI_MCA_plm=^slurm
export FI_LOG_LEVEL=info

./benchmarks/run.sh -k "snellius-short-genoa-hybrid,HPC,Same Rack,Day"
