#!/bin/bash

#SBATCH --job-name=a2a_CudaAware_halfnode
#SBATCH --output=sout/contained/snellius_a2a_CudaAware_halfnode_contained_%j.out
#SBATCH --error=sout/contained/snellius_a2a_CudaAware_halfnode_contained_%j.err

#SBATCH --partition=gpu_h100
#SBATCH --account=vusei7310
#SBATCH --time=00:30:00
#SBATCH --qos=normal

#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

#SBATCH --requeue

echo "-------- Topology Info --------"
echo "Nnodes = $SLURM_NNODES"
srun -l bash -c 'if [[ "$SLURM_LOCALID" == "0" ]] ; then t="$SLURM_TOPOLOGY_ADDR" ; echo "Node: $SLURM_NODEID ---> $t" ; echo "$t" > tmp_${SLURM_NODEID}_${SLURM_JOB_ID}.txt ; fi'
echo "-------------------------------"
switchesPaths=()
for i in $(seq 0 $SLURM_NNODES)
do
        text=$(cat "tmp_${i}_${SLURM_JOB_ID}.txt")
        switchesPaths+=( "$text" )
        rm "tmp_${i}_${SLURM_JOB_ID}.txt"
done

echo "switchesPaths:"
for e in ${switchesPaths[@]}
do
        echo $e
done

echo "-------------------------------"
IFS='.' read -a zeroPath <<< "${switchesPaths[0]}"
# echo "zeroPath:"
# for e in ${zeroPath[@]}
# do
#         echo $e
# done

y="${#zeroPath[@]}"
zeroNode=${zeroPath[-1]}
maxDist="${#zeroPath[@]}"
for e in ${switchesPaths[@]}
do
        IFS='.' read -a tmpPath <<< "$e"
        tmpNode=${tmpPath[-1]}
        x="${#zeroPath[@]}"
        for j in ${!zeroPath[@]}
        do
                if [[ "${zeroPath[$j]}" != "${tmpPath[$j]}" && "$j" < "$x" ]]
                then
                        x="$j"
                        if [[ "$x" < "$y" ]]
                        then
                                y="$x"
                        fi
                fi
        done
        echo "$tmpNode ---> distance with node 0 ($zeroNode) = $(($maxDist - $x))"
done

echo "Max distance: $(($maxDist - $y))"
# if [[ "$(($maxDist - $y))" != "0" ]]
# then
#     echo "nodes are at the wrong distance ($(($maxDist - $y)) instead of 0); job requeued"
#     scontrol requeue ${SLURM_JOB_ID}
# fi

echo "-------------------------------"
echo ""
echo "-------------------------------"


# Add OpenMPI paths to environment variables
export PATH=/opt/software/linux-rocky8-zen/gcc-8.5.0/openmpi-4.1.5-ezcg3cwfrw2f6vrked6v7wz56vfupqs6/bin:$PATH
export LD_LIBRARY_PATH=/opt/software/linux-rocky8-zen/gcc-8.5.0/openmpi-4.1.5-ezcg3cwfrw2f6vrked6v7wz56vfupqs6/lib:$LD_LIBRARY_PATH
export CPATH=/opt/software/linux-rocky8-zen/gcc-8.5.0/openmpi-4.1.5-ezcg3cwfrw2f6vrked6v7wz56vfupqs6/include:$CPATH

# Add NCCL paths to environment variables
export PATH=/opt/software/linux-rocky8-zen/gcc-8.5.0/nccl-2.16.2-1-kwd5zgqdt7w3ioyyxbz6jb5sydloxguy/bin:$PATH
export LD_LIBRARY_PATH=/opt/software/linux-rocky8-zen/gcc-8.5.0/nccl-2.16.2-1-kwd5zgqdt7w3ioyyxbz6jb5sydloxguy/lib:$LD_LIBRARY_PATH
export CPATH=/opt/software/linux-rocky8-zen/gcc-8.5.0/nccl-2.16.2-1-kwd5zgqdt7w3ioyyxbz6jb5sydloxguy/include:$CPATH

# Add CUDA paths to environment variables
export PATH=/opt/software/linux-rocky8-zen/gcc-8.5.0/cuda-12.1.0-v6au6f6vdo7vhuf733jzbzbcu7h4ngpc/bin:$PATH
export LD_LIBRARY_PATH=/opt/software/linux-rocky8-zen/gcc-8.5.0/cuda-12.1.0-v6au6f6vdo7vhuf733jzbzbcu7h4ngpc/lib64:$LD_LIBRARY_PATH
export CPATH=/opt/software/linux-rocky8-zen/gcc-8.5.0/cuda-12.1.0-v6au6f6vdo7vhuf733jzbzbcu7h4ngpc/include:$CPATH


mkdir -p sout/contained
# cd $HOME/apptainer_interconnect/images && srun apptainer exec --nv spack.sif env UCX_TLS=sm,self UCX_MEMTYPE_CACHE=n PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" CPATH="$CPATH" bash -c "cd /opt/interconnect-benchmark-clean/src/energy_binary/ && ./a2a_CudaAware -p 1"
cd $HOME/apptainer_interconnect/images && export UCX_IB_SL=1 && apptainer exec --nv spack_maybe.sif /opt/interconnect-benchmark-clean/src/energy_binary/a2a_CudaAware -p 1
