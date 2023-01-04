#!/bin/bash
#SBATCH --job-name="unet-ds_rot_1vdr"
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-12:00 # Runtime: 12 hours
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -o /home/bsc39/bsc39304/logs/unet-ds_rot_1vdr-%j.out # File to which STDOUT will be written
#SBATCH -e /home/bsc39/bsc39304/logs/unet-ds_rot_1vdr-%j.err # File to which STDERR will be written

# Dependencies
module load SINGULARITY/3.5.2

cd /home/bsc39/bsc39304/mnms/
singularity exec --nv images/pollito.sif python segmentation_model/train.py unet2D_bn_modified_ds -d mnms_1vdr -b 8 -r -n _1vdr_ds_rot
