#!/bin/bash

#SBATCH --job-name=ar_Baseline_halfnode
#SBATCH --output=sout/hybrid/snellius_ar_Baseline_halfnode_hybrid_%j.out
#SBATCH --error=sout/hybrid/snellius_ar_Baseline_halfnode_hybrid_%j.err

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



mkdir -p sout/hybrid
cd $HOME && export UCX_IB_SL=1 && srun apptainer exec --nv final_hybrid.sif env UCX_TLS=sm,self UCX_MEMTYPE_CACHE=n bash -c "cd /opt/interconnect-benchmark-clean/src/energy_binary/ && ./ar_Baseline -p 1"
