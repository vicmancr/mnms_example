#!/bin/bash
#SBATCH --job-name="unet-ds"
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-24:00 # Runtime: 24 hours
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -o /home/bsc39/bsc39304/logs/unet-ds-%j.out # File to which STDOUT will be written
#SBATCH -e /home/bsc39/bsc39304/logs/unet-ds-%j.err # File to which STDERR will be written

# Dependencies
module load SINGULARITY/3.5.2

cd /home/bsc39/bsc39304/mnms/
singularity exec --nv images/pollito.sif python segmentation_model/train.py unet2D_bn_modified_ds -d mnms -b 8 -n _ds
