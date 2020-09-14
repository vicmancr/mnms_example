#!/bin/bash
#SBATCH --job-name="unet-ds_rot_1vdr"
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-48:00 # Runtime: 24 hours
#SBATCH -p high # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -o /homedtic/vcampello/logs/unet-ds_rot_1vdr-%j.out # File to which STDOUT will be written
#SBATCH -e /homedtic/vcampello/logs/unet-ds_rot_1vdr-%j.err # File to which STDERR will be written

# Dependencies
module load SINGULARITY/3.5.2

cd /home/bsc39/bsc39304/mnms/
singularity exec --nv images/pollito.sif python train.py unet2D_bn_modified_ds -d mnms_1vdr -b 8 -r -n ds_rot
