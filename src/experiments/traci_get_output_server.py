
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random

import platform
# on the server (platform = Linux) we use libsumo and also don't need the tools in the path
if platform.system() != "Linux":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)  # we need to import python modules from the $SUMO_HOME/tools directory
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    import traci
else:
    import libsumo as traci

from sumolib import checkBinary


# we need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

# from sumolib import checkBinary  # noqa
# import traci  # noqa

def run():


    # run the simulation for 1000 time steps
    for i in range(1000):
        # get the current simulation time
        current_time = traci.simulation.getCurrentTime()

        # get the list of vehicles on the road network
        vehicle_list = traci.vehicle.getIDList()

        # print the number of vehicles on the road network
        # print("Current time: ", current_time)
        # print("Number of vehicles: ", len(vehicle_list))

        # advance the simulation by one step
        traci.simulationStep()

    # stop the TraCI server
    traci.close()


    #"""execute the TraCI control loop"""
    #step = 0
    # we start with phase 2 where EW has green
    #traci.trafficlight.setPhase("0", 2)
    #while traci.simulation.getMinExpectedNumber() > 0:
    #    traci.simulationStep()
    #    if traci.trafficlight.getPhase("0") == 2:
    #        # we are not already switching
    #        if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
    #            # there is a vehicle from the north, switch
    #            traci.trafficlight.setPhase("0", 3)
    #        else:
    #           # otherwise try to keep green for EW
    #            traci.trafficlight.setPhase("0", 2)
    #    step += 1
    #traci.close()
    #sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=True, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    #generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    

    traci.start(['sumo-gui', "-c", "/pfs/data5/home/ma/ma_ma/ma_jenhahn/masterarbeit_sumo/urban_mobility_simulation/models/20230502_SUMO_MA/osm.sumocfg",
                                "--tripinfo-output", "/pfs/data5/home/ma/ma_ma/ma_jenhahn/masterarbeit_sumo/urban_mobility_simulation/src/data/actuated_output/tripinfo.xml",
                                "--output-prefix", "TIME",
                                "--device.emissions.probability", "1.0",
                                "--emission-output", "/pfs/data5/home/ma/ma_ma/ma_jenhahn/masterarbeit_sumo/urban_mobility_simulation/src/data/actuated_output/emission_info.xml"])
    
    #run the simulation a few times to compare outputs

    run()
