#!/bin/bash

#SBATCH --job-name=hlo_Baseline_halfnode
#SBATCH --output=sout/snellius_hlo_Baseline_halfnode_%j.out
#SBATCH --error=sout/snellius_hlo_Baseline_halfnode_%j.err

#SBATCH --partition=gpu
#SBATCH --account=vusei7310
#SBATCH --time=00:05:00
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

#SBATCH --requeue


# fi

echo "-------------------------------"
echo ""
echo "-------------------------------"

source exports/vars.sh



MODULE_PATH="moduleload/load_Baseline_modules.sh"
EXPORT_PATH="exportload/load_Baseline_halfnode_exports.sh"

mkdir -p sout
cat "${EXPORT_PATH}"
source ${MODULE_PATH} && source ${EXPORT_PATH} &&   srun bin/hlo_Baseline -pex 2 -pey 2 -pez 1
