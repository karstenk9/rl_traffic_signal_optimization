import os
import sys

sys.path.append(os.path.join('c:', os.sep, '/opt/homebrew/opt/sumo/share/sumo/tools'))

sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "/Users/jenniferhahn/Library/CloudStorage/OneDrive-UniversitätMannheim/01_Master_Thesis/urban_mobility_simulation/models/Mannheim145_export06032023/osm.sumocfg"]


import traci

# start the TraCI server
traci.start(sumoCmd)

# run the simulation for 1000 time steps
for i in range(1000):
    # get the current simulation time
    #current_time = traci.simulation.getCurrentTime()

    # get the list of vehicles on the road network
    vehicle_list = traci.vehicle.getIDList()

    # print the number of vehicles on the road network
    #print("Current time: ", current_time)
    print("Number of vehicles: ", len(vehicle_list))

    # advance the simulation by one step
    traci.simulationStep()

# stop the TraCI server
traci.close()

