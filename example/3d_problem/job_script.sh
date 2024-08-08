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

#SBATCH -J PACT_3D_problem_factor_8           # Job name
#SBATCH -o PACT_3D_problem_factor_8.o%j       # Name of stdout output file
#SBATCH -e PACT_3D_problem_factor_8.e%j       # Name of stderr error file
#SBATCH -p development         # Queue (partition) name
#SBATCH -N 4               # Total # of nodes (must be 1 for serial)
#SBATCH -n 128               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A CDA23008       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=venu22@tacc.utexas.edu

# Any other commands must follow all #SBATCH directives...
pwd
date

# Launch code...
cd /work/09052/venurang/ls6/PACT_study/hippylibx/
module load tacc-apptainer/1.1.8
module load mvapich2/2.3.7
MV2_SMP_USE_CMA=0 MV2_USE_ALIGNED_ALLOC=1 ibrun apptainer run $SCRATCH/revised_dolfinx_image_tacc_v3.sif python -u example/3d_problem_interpolate.py