import os
import sys
from sumolib import checkBinary
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "/Users/jenniferhahn/Library/CloudStorage/OneDrive-UniversitätMannheim/01_Master_Thesis/urban_mobility_simulation/models/Mannheim145_export06032023/osm.sumocfg"]


# start the TraCI server
traci.start(sumoCmd)

# run the simulation for 1000 time steps
for i in range(1000):
    # get the current simulation time
    current_time = traci.simulation.getCurrentTime()

    # get the list of vehicles on the road network
    vehicle_list = traci.vehicle.getIDList()

    # print the number of vehicles on the road network
    print("Current time: ", current_time)
    print("Number of vehicles: ", len(vehicle_list))

    # advance the simulation by one step
    traci.simulationStep()

# stop the TraCI server
traci.close()

