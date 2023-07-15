#!/bin/bash

#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --job-name=ppo_test
#SBATCH --output=output/ppo_test.out
#SBATCH --error=output/ppo_test.err

python urban_mobility_simulation/src/experiments/ppo_server_test.py

