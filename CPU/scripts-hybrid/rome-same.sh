    #!/bin/bash
    #SBATCH --job-name rome-hybrid
    #SBATCH --nodes=2 
    #SBATCH --ntasks-per-node=16
    #SBATCH --time=02:00:00
    #SBATCH --partition=rome
    #SBATCH --exclusive
    #SBATCH --output=rome-same-hybrid.txt




    export OMPI_DIR=/opt/ompi
    export APPTAINER_OMPI_DIR=$OMPI_DIR
    export APPTAINERENV_APPEND_PATH=$OMPI_DIR/bin
    export APPTAINERENV_APPEND_LD_LIBRARY_PATH=$OMPI_DIR/lib


    module load 2023
    module load OpenMPI/4.1.5-GCC-12.3.0
    module load UCX/1.14.1-GCCcore-12.3.0

    #export FI_LOG_LEVEL=info
    export OMPI_MCA_btl_openib_allow_ib=true
    export OMPI_MCA_btl_vader_single_copy_mechanism=none
    export OMPI_MCA_plm=^slurm
    export UCX_POSIX_USE_PROC_LINK=n


# export OMPI_MCA_ess=pmi2
#export LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/ucx-ohpc/1.11.2/lib:/opt/ohpc/pub/mpi/libfabric/1.13.0/lib:/opt/ohpc/pub/mpi/ucx-ohpc/1.11.2/lib:$LD_LIBRARY_PATH


# mpirun -np 32 echo "Hello, World! from processor $SLURM_PROCID" > output.txt
# singularity exec -B /home/gsavchenko/data/:/opt/cloud_noise/data/snellius/:rw bench_singularity.sif bash -c "cd /opt/cloud_noise/benchmarks/ && ./run.sh -s -k 'snellius-short-genoa,HPC,Same Rack,Day'" > output.txt
# singularity run -B /home/gsavchenko/data/:/opt/cloud_noise/data/:rw cloud_noise_2.sif bash -c "cd /opt/cloud_noise/benchmarks/ && ./run.sh -s -k 'snellius-short-genoa,HPC,Same Rack,Day'" > output.txt
# singularity run -B /home/gsavchenko/data/:/opt/cloud_noise/data/:rw \
#                 -B /usr/lib/x86_64-linux-gnu/slurm:/opt/slurm/:ro \
#                 -B /tmp:/tmp \
#                 --network=host \
#                 cloud_noise.sif bash -c "cd /opt/cloud_noise/benchmarks/ && mpirun -np 32 ./run.sh -s -k 'snellius-short-genoa,HPC,Same Rack,Day'" > output.txt

# singularity run -B /home/gsavchenko/data/:/opt/cloud_noise/data/:rw \
#                 -B /usr/lib/x86_64-linux-gnu/slurm:/opt/slurm/:ro \
#                 -B /tmp:/tmp \
#                 --network=host \ 
#                 cloud_noise.sif bash -c "cd /opt/cloud_noise/benchmarks/ && mpirun -np 32 ./


#export OMPI_MCA_btl_tcp_if_include=ens4f0np0,ib0
./benchmarks/run_hybrid.sh -k 'snellius-short-rome-hybrid,HPC,Same Rack,Day'

# cd .. && srun --mpi=pmi2 --output=slurm.out --error=slurm.out apptainer exec cloud_noise_spack2.sif /opt/benchmarks/mpi_tests/comm_test 32


# FI_LOG_LEVEL=info srun --partition=rome --mpi=cray_shasta --nodes 2 --tasks-per-node=16   --output=slurm.out --error=slurm.out apptainer exec cloud_noise_spack2.sif /opt/benchmarks/run.sh -s -k

# mpirun -n 32 singularity exec mpitest.sif /opt/mpitest

# check status of ucx library
# apptainer exec cloud_noise_2.sif ldd /opt/ompi/lib/openmpi/mca_pml_ucx.so
# mpirun -np 32 apptainer exec --fakeroot cloud_noise_2.sif /opt/cloud_noise/benchmarks/hello

