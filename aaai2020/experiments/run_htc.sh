#!/bin/bash

#SBATCH --account=ac_bnlp
#SBATCH --partition=savio2_htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=71:59:00
#SBATCH --mail-user=dfried@berkeley.edu
#SBATCH --mail-type=all
#SBATCH -o ./slurm_logs/slurm-%j-%x.out
#SBATCH -e ./slurm_logs/slurm-%j-%x.out

echo "job name:" $SLURM_JOB_NAME

source activate onecommon

echo "command: " $@

$@
