#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=20
#SBATCH --ntasks=1
#SBATCH --job-name=ppo_test
#SBATCH --output=ppo_test.out

./urban_mobility_simulation/src/experiments/ppo_server_test.py