#!/bin/bash

#SBATCH --time=00:40:00
#SBATCH --partition=multiple
#SBATCH --nodes=5
#SBATCH --job-name=convert_xml
#SBATCH --output=output/convert_xml.out
#SBATCH --error=output/convert_xml.err

python convert_xml_output.py

