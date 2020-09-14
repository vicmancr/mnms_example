#!/bin/bash
#SBATCH --job-name="dugong"
#SBATCH --ntasks=1 # The number of processes to start
#SBATCH --cpus-per-task=2 # how many threads each process would open
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 15 # Runtime: 15 minutes
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -o /home/victor_campello/logs/dugong-%j.out # File to which STDOUT will be written
#SBATCH -e /home/victor_campello/logs/dugong-%j.err # File to which STDERR will be written

# dugong arguments must be substituted by proper ones

cd /home/victor_campello/mnms/

mkdir -p /home/victor_campello/mnms/results/dugong
singularity run --nv images/dugong.sif /home/victor_campello/mnms/validation /home/victor_campello/mnms/results/dugong
singularity exec --nv images/pollito.sif python metrics_mnms.py gt_validation/ results/dugong
