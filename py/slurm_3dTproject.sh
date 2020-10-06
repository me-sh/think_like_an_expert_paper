#! /bin/env bash

#SBATCH -J 'run_3dTproject'
#SBATCH -o /mnt/bucket/people/meshulam/slurm_logs/%j.out
#SBATCH -p all
#SBATCH -t 30



#!/bin/bash

# load latest afni module
module load afni/2018.10.29

# params
# $1: input_file
# $2: 1D file
# $3: brainmask
# $4: output_file

echo "Running 3dTproject"
echo "TR=2.0"
echo "project out 6 motion regressors"

echo ""
echo "input file: $1" #pre-processed without smoothing or highpass or denoising of motion regressors
echo "ort file (confounds): $2" #1D file
echo "brain mask: $3" # mask
echo "output file: $4" #nii.gz
echo ""

# Run AFNI's 3dTproject (good for whole brain spaces, T1w and MNI, not fsaverage) - with smoothing
3dTproject -polort 0 -TR 2.0 -input $1 -ort $2 -mask $3 -prefix $4

echo "completed"

