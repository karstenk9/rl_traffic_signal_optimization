#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=20
#SBATCH --ntasks=1
#SBATCH --job-name=ppo_test
#SBATCH --output=ppo_test.out

cd urban_mobility_simulation/src/experiments/

python traci_get_output.py

