#!/bin/bash
#SBATCH -J hftrain                       # Job name
#SBATCH -o slurm/hftrain_%j.out          # output file (%j expands to jobID)
#SBATCH -e slurm/hftrain_%j.err          # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH --mail-user=jtc257@cornell.edu       # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 2                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=16000                          # server memory requested (per node)
#SBATCH -t 8:00:00                           # Time limit (hh:mm:ss)
##SBATCH --nodelist=rush-compute-01 # Request partition
##SBATCH --partition=rush # Request partition
##SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --gres=gpu:3090:1                  # Type/number of GPUs needed

. "/home/jtc257/anaconda3/etc/profile.d/conda.sh"
source /home/jtc257/.bashrc
source /home/jtc257/scripts/env.sh
py111env
#source jc_run_train.sh
#indicator-confirm-mean 9
#bash jc_run_test_static_plan.sh

#python hftrain.py --dataset text_given_plan_py_2py_2puy_en_sdn_psn_un --learning_rate 1e-5
python hftrain.py --dataset text_given_plan_py_2py_2puy_en_sdy_psn_un --learning_rate 1e-5
#python hftrain.py --dataset text_given_plan_py_2py_2puy_en_sdy_psy_un --learning_rate 1e-5
