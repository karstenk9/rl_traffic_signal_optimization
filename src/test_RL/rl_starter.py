from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import numpy as np

from sumolib import checkBinary
import traci 

# import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def rl_env():
    
    """
    Define reinforcement learning environment
    with states, actions, and rewards
    """

    n_actions = 2  # number of possible actions: stop or go (non-conflicting phases)
    n_states = 4  # number of possible states: North-South and East-West traffic light color
    Q = np.zeros([n_states, n_actions])  # Q-table for storing state-action values
    gamma = 0.8  # discount factor for future rewards
    alpha = 0.5  # learning rate for updating Q-values
    epsilon = 0.1  # exploration rate for choosing random actions
    
    return n_actions, n_states, Q, gamma, alpha, epsilon


def run(
    n_actions,
    n_states,
    Q,
    gamma,
    alpha,
    epsilon
    ):
    
    """
    Function to run the simulation for 1000 time steps, print the number of vehicles
    on the road network and run the reinforcement learning environment.
    """


    # define simulation loop
    for step in range(1000):

        # get the current simulation time
        current_time = traci.simulation.getCurrentTime()

        # get the list of vehicles on the road network
        vehicle_list = traci.vehicle.getIDList()
        
        # print the number of vehicles on the road network
        print("Current time: ", current_time)
        print("Number of vehicles: ", len(vehicle_list))
            
        traci.simulationStep()  # advance the simulation by one step
        
        # get current state
        north_south_color = traci.trafficlight.getRedYellowGreenState("n-s")
        east_west_color = traci.trafficlight.getRedYellowGreenState("e-w")
        if north_south_color == "G" and east_west_color == "r":
            state = 0  # North-South green, East-West red
        elif north_south_color == "r" and east_west_color == "G":
            state = 1  # North-South red, East-West green
        elif north_south_color == "y" and east_west_color == "r":
            state = 2  # North-South yellow, East-West red
        elif north_south_color == "r" and east_west_color == "y":
            state = 3  # North-South red, East-West yellow
        
        # choose action using epsilon-greedy strategy
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, n_actions-1)  # choose random action
        else:
            action = np.argmax(Q[state])  # choose best action based on current Q-values
            
        # take action and get reward
        if action == 0:
            traci.trafficlight.setRedYellowGreenState("n-s", "rrrGrrr")  # allow North-South traffic to move
            traci.trafficlight.setRedYellowGreenState("e-w", "rrrRrrr")  # stop East-West traffic
            reward = -1  # negative reward for stopping traffic
        else:
            traci.trafficlight.setRedYellowGreenState("n-s", "rrrRrrr")  # stop North-South traffic
            traci.trafficlight.setRedYellowGreenState("e-w", "rrrGrrr")  # allow East-West traffic to move
            reward = -1  # negative reward for stopping traffic
        
        # get new state and update Q-values
        north_south_color = traci.trafficlight.getRedYellowGreenState("n-s")
        east_west_color = traci.trafficlight.getRedYellowGreenState("e-w")
        if north_south_color == "G" and east_west_color == "r":
            new_state = 0  # North-South green, East-West red
        elif north_south_color == "r" and east_west_color == "G":
            new_state = 1
    
    # stop the TraCI server
    traci.close()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
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

    # start sumo as a subprocess and run the python script
    sumoCmd = [sumoBinary, "-c", "urban_mobility_simulation/models/Mannheim145_export06032023/osm.sumocfg"]
    traci.start(sumoCmd)
    
    traci.start([sumoBinary, "-c", "urban_mobility_simulation/models/Mannheim145_export06032023/osm.sumocfg",
                                "--tripinfo-output", "urban_mobility_simulation/src/data/tripinfo.xml",
                                "--output-prefix", "TIME",
                                "--device.emissions.probability", "1.0",
                                "--emission-output", "urban_mobility_simulation/src/data/emission_info.xml"])

    run()



############################################################################################################


# create SUMO simulation environment
sumoBinary = "/usr/bin/sumo-gui"  # path to SUMO binary
sumoCmd = [sumoBinary, "-c", "crossing.sumocfg"]  # path to SUMO configuration file
traci.start(sumoCmd)


# define simulation loop
for step in range(1000):
    traci.simulationStep()  # advance the simulation by one step
    
    # get current state
    north_south_color = traci.trafficlight.getRedYellowGreenState("n-s")
    east_west_color = traci.trafficlight.getRedYellowGreenState("e-w")
    if north_south_color == "G" and east_west_color == "r":
        state = 0  # North-South green, East-West red
    elif north_south_color == "r" and east_west_color == "G":
        state = 1  # North-South red, East-West green
    elif north_south_color == "y" and east_west_color == "r":
        state = 2  # North-South yellow, East-West red
    elif north_south_color == "r" and east_west_color == "y":
        state = 3  # North-South red, East-West yellow
    
    # choose action using epsilon-greedy strategy
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, n_actions-1)  # choose random action
    else:
        action = np.argmax(Q[state])  # choose best action based on current Q-values
        
    # take action and get reward
    if action == 0:
        traci.trafficlight.setRedYellowGreenState("n-s", "rrrGrrr")  # allow North-South traffic to move
        traci.trafficlight.setRedYellowGreenState("e-w", "rrrRrrr")  # stop East-West traffic
        reward = -1  # negative reward for stopping traffic
    else:
        traci.trafficlight.setRedYellowGreenState("n-s", "rrrRrrr")  # stop North-South traffic
        traci.trafficlight.setRedYellowGreenState("e-w", "rrrGrrr")  # allow East-West traffic to move
        reward = -1  # negative reward for stopping traffic
    
    # get new state and update Q-values
    north_south_color = traci.trafficlight.getRedYellowGreenState("n-s")
    east_west_color = traci.trafficlight.getRedYellowGreenState("e-w")
    if north_south_color == "G" and east_west_color == "r":
        new_state = 0  # North-South green, East-West red
    elif north_south_color == "r" and east_west_color == "G":
        new_state = 1
