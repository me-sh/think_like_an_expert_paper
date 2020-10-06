#! /bin/env bash

#SBATCH -J 'run_bet'
#SBATCH -o /usr/people/meshulam/slurm_logs/%j.out
#SBATCH -p all
#SBATCH -t 29

module load fsl

echo "input: $1"
echo "output 1 - reorient2std: $2"
echo "output 2- bet of reorient2std: $3"

fslreorient2std $1 $2
bet $2 $3 -R -B -f 0.3

echo "done"
