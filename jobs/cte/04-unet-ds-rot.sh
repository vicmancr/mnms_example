#!/bin/bash
#SBATCH --job-name="unet-ds_rot"
#SBATCH -n 40 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-12:00 # Runtime: 12 hours
#SBATCH --ntasks-per-node=40 # Necessary in BSC cluster
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -o /home/bsc39/bsc39304/logs/unet-ds_rot-%j.out # File to which STDOUT will be written
#SBATCH -e /home/bsc39/bsc39304/logs/unet-ds_rot-%j.err # File to which STDERR will be written

# Dependencies
module load ffmpeg/4.0.2
module load cudnn/7.1.3 atlas/3.10.3 scalapack/2.0.2 
module load fftw/3.3.7 szip/2.1.1 opencv/3.4.1
module load python/3.6.5_ML

cd /home/bsc39/bsc39304/mnms/
singularity exec --nv images/pollito.sif python segmentation_model/train.py unet2D_bn_modified_ds -d mnms -b 8 -r -n _ds_rot
