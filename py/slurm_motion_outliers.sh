#! /bin/env bash

#SBATCH -J 'run_fslMotionOutliers'
#SBATCH -o /mnt/bucket/people/meshulam/slurm_logs/%j.out
#SBATCH -p all
#SBATCH --cpus-per-task=1 
#SBATCH -t 50
#SBATCH --mem=32GB

module load fsl

echo "Running fslMotionOutliers"
echo "input .nii.gz: $1"
echo "binary output .txt: $2"
echo "metric filename .txt: $3"
echo "plot filename .png: $4"

fsl_motion_outliers -i $1 -o $2 --fd -s $3 -p $4
echo "completed"

