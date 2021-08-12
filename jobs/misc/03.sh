#!/bin/bash
#SBATCH --job-name="nnunet"
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-24:00 # Runtime: 24 hours
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH -o /home/bsc39/bsc39304/logs/nnunet-%j.out # File to which STDOUT will be written
#SBATCH -e /home/bsc39/bsc39304/logs/nnunet-%j.err # File to which STDERR will be written

# Dependencies
module load SINGULARITY/3.5.2

cd /home/bsc39/bsc39304/
singularity exec /gpfs/projects/bsc39/bsc39304/images/nnunet.sif sh command3.sh

