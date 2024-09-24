#!/bin/bash
#SBATCH --job-name=run_short_rome_diffrack
#SBATCH --output=run_short_rome_diffrack.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --partition=rome
#SBATCH --time=02:30:00
#SBATCH --exclusive
#SBATCH --nodelist=tcn[326,487]

export OMPI_DIR=/opt/ompi
export APPTAINER_OMPI_DIR=$OMPI_DIR
export APPTAINERENV_APPEND_PATH=$OMPI_DIR/bin
export APPTAINERENV_APPEND_LD_LIBRARY_PATH=$OMPI_DIR/lib
module load 2023
module load OpenMPI/4.1.5-GCC-12.3.0
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_btl_vader_single_copy_mechanism=none
OMPI_MCA_plm=^slurm

./benchmarks/run_hybrid.sh -k "snellius-short-rome-hybrid,HPC,Different Racks,Day" 

