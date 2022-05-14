#!/bin/bash
#SBATCH -J binary_dots                       # Job name
#SBATCH -o slurm/binary_dots_%j.out          # output file (%j expands to jobID)
#SBATCH -e slurm/binary_dots_%j.err          # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH --mail-user=jtc257@cornell.edu       # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 2                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=16000                          # server memory requested (per node)
#SBATCH -t 2:00:00                           # Time limit (hh:mm:ss)
#SBATCH --nodelist=rush-compute-01 # Request partition
#SBATCH --partition=rush # Request partition
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed

source /home/jtc257/.bashrc
py111env
#source jc_run_train.sh
#indicator-confirm-mean 9
bash jc_run_test_static_plan.sh
