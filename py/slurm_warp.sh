#! /bin/env bash

#SBATCH -J 'run_warp'
#SBATCH -o /mnt/bucket/people/meshulam/slurm_logs/%j.out
#SBATCH -p all
#SBATCH -t 50

module load fsl

echo "Running warp"
echo "MNI brain ref file input: $1"
echo "Input file: $2"
echo "Output file: $3"
echo "Feat directory: $4"

applywarp --ref=$1 --in=$2 --out=$3 --warp=$4/reg/highres2standard_warp.nii.gz --premat=$4/reg/example_func2highres.mat
echo "completed"

