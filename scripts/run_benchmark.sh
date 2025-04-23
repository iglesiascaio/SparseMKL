#!/bin/bash

#SBATCH --cpus-per-task=8            # Number of CPUs per task
#SBATCH --mem=187G                    # Total memory per node



# Loading the required module
module load anaconda/Python-ML-2025a
module load julia/1.1.10.1

# Run the script
python run_benchmark.py