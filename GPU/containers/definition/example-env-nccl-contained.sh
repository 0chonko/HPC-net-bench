#!/bin/bash
#SBATCH --job-name=nccl-tests
#SBATCH --output=nccl-tests-%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=0:05:00

# SOURCE: https://www.together.ai/blog/a-practitioners-guide-to-testing-and-running-large-gpu-clusters-for-training-generative-ai-models
export LD_LIBRARY_PATH=/usr/lib:/usr/lib64
export NCCL_TESTS_HOME=nccl-tests
export NCCL_DEBUG=INFO
export NCCL_ALGO=RING
# export NCCL_DEBUG_SUBSYS=NET
export NCCL_IB_AR_THRESHOLD=0 
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_SPLIT_DATA_ON_QPS=0 
export NCCL_IB_QPS_PER_CONNECTION=2 
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_SOCKET_IFNAME=enp27s0np0
export NCCL_IGNORE_CPU_AFFINITY=1
. /opt/hpcx/hpcx-init.sh
hpcx_load
mpirun \
        --bind-to none \
        -mca btl tcp,self \
        -mca coll_hcoll_enable 0 \
        -mca btl_tcp_if_include enp27s0np0 \
        -x PATH \
        -x LD_LIBRARY_PATH \
        ${NCCL_TESTS_HOME}/build/all_reduce_perf -b 3G -e 24G -f 2 -g 8
