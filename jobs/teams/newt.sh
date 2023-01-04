#!/bin/bash
#SBATCH --job-name="newt"
#SBATCH --ntasks=1 # The number of processes to start
#SBATCH --cpus-per-task=2 # how many threads each process would open
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 24:00:00 # Runtime: 24 hours
#SBATCH --mem=12G
#SBATCH -o /home/victor_campello/logs/newt-%j.out # File to which STDOUT will be written
#SBATCH -e /home/victor_campello/logs/newt-%j.err # File to which STDERR will be written

# How to execute: run `sbatch example_job.sh <group_name>`

# Dependencies
module load singularity

cd /home/victor_campello/mnms/
singularity exec mega.sif mega-get -m Singularity/newt/newt.sif /home/victor_campello/mnms/images/newt.sif
mkdir /home/victor_campello/mnms/results/newt
singularity run images/newt.sif /home/victor_campello/mnms/validation /home/victor_campello/mnms/results/newt