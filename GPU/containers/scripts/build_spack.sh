#!/bin/bash
#SBATCH --job-name=build_spack_container
#SBATCH --output=build_log.txt
#SBATCH --partition=cbuild
#SBATCH --time=04:00:00

export APPTAINER_TMPDIR=$(mktemp -d /tmp/gsavchenkoXXXX)


cd ../definition-files/ && apptainer build --fakeroot --force spack_without_compile_flags.sif fully_contained.def 
