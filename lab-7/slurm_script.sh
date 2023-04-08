#!/bin/bash
#SBATCH --job-name=m21aie225_lab7 	# Job name
#SBATCH --partition=gpu2 	#Partition name can be test/small/medium/large/gpu #Partition “gpu” should be used only for gpu jobs
#SBATCH --nodes=1 			# Run all processes on a single node
#SBATCH --ntasks=1 			# Run a single task
#SBATCH --cpus-per-task=4 	# Number of CPU cores per task
#SBATCH --gres=gpu:1 		# Include gpu for the task (only for GPU jobs)
#SBATCH --mem=16gb 			# Total memory limit
#SBATCH --time=90:00:00 	# Time limit hrs:min:sec
#SBATCH --output=m21aie225_lab7_%j.log # Standard output and error log
#date;hostname;pwd


module load python/3.8

python3 M21AIE225_Lab_Assignment_7.py
