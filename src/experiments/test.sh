#!/bin/bash

# Load necessary modules
module load SUMO/1.9.2-foss-2020a

# Run SUMO simulation
sumo-gui -c urban_mobility_simulation/models/20230502_SUMO_MA/osm.sumocfg \
    --tripinfo-output urban_mobility_simulation/src/data/tripinfo_actuated_TL.xml \
    --output-prefix TIME \
    --device.emissions.probability 1.0 \
    --emission-output urban_mobility_simulation/src/data/emission_info_actuated_TL.xml
