#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=20
#SBATCH --ntasks=1
#SBATCH --job-name=actuated_TL
#SBATCH --output=output/actuated_TL.out
#SBATCH --error=output/actuated_TL.err
#SBATCH --gres=gpu:4

cd urban_mobility_simulation/src/experiments/

python traci_get_output_server.py

