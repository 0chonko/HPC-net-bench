MPIRUN="srun" # Command for running MPI applications
MPIRUN_MAP_BY_NODE_FLAG="" # Flag to force ranks to be mapped by node (srun)
# MPIRUN_HOSTFILE_FLAG="--hostfile conf/snellius_hostfile" # Flag for specifying the hostfile
# MPIRUN_HOSTFILE_LONG_FLAG="--hostfile conf/snellius_hostfile_long" # Flag for specifying the hostfile (16 hosts)
MPIRUN_HOSTFILE_FLAG=""
MPIRUN_HOSTFILE_LONG_FLAG=""
MPIRUN_PINNING_FLAGS="" # Pinning flags
MPIRUN_ADDITIONAL_FLAGS="--mpi=pmi2"    # Any additional flag that must be used by mpirun
# INTERFACE_MASK="172.22.48.0/20" # Interface address + mask size of the two nodes
INTERFACE_MASK=""
RUN_IB=true # Shall we run IB tests?
RUN_IBV=true # Shall we run IBV tests?
NET_NOISE_CONC=1 # How many concurrent connections to run to measure bandwidth network noise
BW_SATURATING_SIZE=16777216 # How many bytes to send to get max bw
NG_CONFIGURE_FLAGS=""
MPI_COMPILER="mpicc"

# export LD_LIBRAY_PATH=$LD_LIBRARY_PATH:/usr/gsavchenko/amd-blis/lib/lp64/

