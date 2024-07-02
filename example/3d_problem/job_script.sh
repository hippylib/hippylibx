#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Frontera CLX nodes
#
#   *** Serial Job in Small Queue***
# 
# Last revised: 22 June 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch clx.serial.slurm" on a Frontera login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J PACT_3D_problem           # Job name
#SBATCH -o PACT_3D_problem.o%j       # Name of stdout output file
#SBATCH -e PACT_3D_problem.e%j       # Name of stderr error file
#SBATCH -p development         # Queue (partition) name
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH -n 12               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:10:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A CDA23008       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=venu22@tacc.utexas.edu

# Any other commands must follow all #SBATCH directives...
pwd
date

# Launch code...
cd /work/09052/venurang/ls6/PACT_study/hippylibx/example
module load tacc-apptainer
ibrun apptainer exec main_image_latest.sif python3 -u 3d_problem_interpolate.py
