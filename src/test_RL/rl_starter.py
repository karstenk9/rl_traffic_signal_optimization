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

def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 3600  # number of time steps
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 11
    pNS = 1. / 30
    with open("urban_mobility_simulation/src/test_RL/data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" \
guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>


def run():
    
    """
    Function to run the simulation for 1000 time steps, print the number of vehicles
    on the road network and run the reinforcement learning environment.
    """

    # define the states, actions, and Q-table
    states = ['GG', 'GY', 'YG', 'YY']
    actions = ['GGGrrr', 'YYYrrr', 'rrrGGG', 'rrrYYY']
    q_table = np.zeros([len(states), len(actions)])


    # define simulation loop
    for step in range(1000):

        # get the current simulation time
        current_time = traci.simulation.getCurrentTime()

        # get the list of vehicles on the road network
        vehicle_list = traci.vehicle.getIDList()
        
        # print the number of vehicles on the road network
        print("Current time: ", current_time)
        print("Number of vehicles: ", len(vehicle_list))
            
        #traci.simulationStep()  # advance the simulation by one step
        
        # get current state
        # input
        # traci.trafficlight.getRedYellowGreenState(self, tlsID)
        
        
        # choose the current state based on the current traffic light phase
        current_phase = traci.trafficlight.getPhase('1')
        if current_phase == 0:
            current_state = 'GG'
        elif current_phase == 1:
            current_state = 'YG'
        elif current_phase == 2:
            current_state = 'YY'
        elif current_phase == 3:
            current_state = 'GY'
        
        # choose the next action based on the current state and the Q-table
        epsilon = 0.1
        if np.random.uniform() < epsilon:
            # choose a random action with probability epsilon
            action_index = np.random.choice(len(actions))
        else:
            # choose the best action based on the Q-table with probability 1-epsilon
            action_index = np.argmax(q_table[current_state_index,:])
        next_action = actions[action_index]
        
        # perform the chosen action and observe the resulting state and reward
        traci.trafficlight.setPhase('1', actions.index(next_action))
        traci.simulationStep()
        next_phase = traci.trafficlight.getPhase('1')
        if next_phase == 0:
            next_state = 'GG'
        elif next_phase == 1:
            next_state = 'YG'
        elif next_phase == 2:
            next_state = 'YY'
        elif next_phase == 3:
            next_state = 'GY'
        reward = traci.edge.getLastStepVehicleNumber('1') + traci.edge.getLastStepVehicleNumber('3') - traci.edge.getLastStepVehicleNumber('2') - traci.edge.getLastStepVehicleNumber('4')
        
        # update the Q-table
        alpha = 0.5
        gamma = 0.9
        current_state_index = states.index(current_state)
        next_state_index = states.index(next_state)
        q_table[current_state_index, action_index] = (1 - alpha) * q_table[current_state_index, action_index] + alpha * (reward + gamma * np.max(q_table[next_state_index,:]))
        
        # increment the step counter
        step += 1

    # end the simulation
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
    sumoCmd = [sumoBinary, "-c", "urban_mobility_simulation/src/test_RL/data/cross.sumocfg"]
    traci.start(sumoCmd)
    
    # traci.start([sumoBinary, "-c", "urban_mobility_simulation/models/Mannheim145_export06032023/osm.sumocfg",
    #                             "--tripinfo-output", "urban_mobility_simulation/src/data/tripinfo.xml",
    #                             "--output-prefix", "TIME",
    #                             "--device.emissions.probability", "1.0",
    #                             "--emission-output", "urban_mobility_simulation/src/data/emission_info.xml"])

    #n_actions, n_states, Q, gamma, alpha, epsilon = rl_env()
    run()



############################################################################################################


# create SUMO simulation environment
sumoBinary = "/usr/bin/sumo-gui"  # path to SUMO binary
sumoCmd = [sumoBinary, "-c", "crossing.sumocfg"]  # path to SUMO configuration file
traci.start(sumoCmd)


# # define simulation loop
# for step in range(1000):
#     traci.simulationStep()  # advance the simulation by one step
    
#     # get current state
#     north_south_color = traci.trafficlight.getRedYellowGreenState("n-s")
#     east_west_color = traci.trafficlight.getRedYellowGreenState("e-w")
#     if north_south_color == "G" and east_west_color == "r":
#         state = 0  # North-South green, East-West red
#     elif north_south_color == "r" and east_west_color == "G":
#         state = 1  # North-South red, East-West green
#     elif north_south_color == "y" and east_west_color == "r":
#         state = 2  # North-South yellow, East-West red
#     elif north_south_color == "r" and east_west_color == "y":
#         state = 3  # North-South red, East-West yellow
    
#     # choose action using epsilon-greedy strategy
#     if random.uniform(0, 1) < epsilon:
#         action = random.randint(0, n_actions-1)  # choose random action
#     else:
#         action = np.argmax(Q[state])  # choose best action based on current Q-values
        
#     # take action and get reward
#     if action == 0:
#         traci.trafficlight.setRedYellowGreenState("n-s", "rrrGrrr")  # allow North-South traffic to move
#         traci.trafficlight.setRedYellowGreenState("e-w", "rrrRrrr")  # stop East-West traffic
#         reward = -1  # negative reward for stopping traffic
#     else:
#         traci.trafficlight.setRedYellowGreenState("n-s", "rrrRrrr")  # stop North-South traffic
#         traci.trafficlight.setRedYellowGreenState("e-w", "rrrGrrr")  # allow East-West traffic to move
#         reward = -1  # negative reward for stopping traffic
    
#     # get new state and update Q-values
#     north_south_color = traci.trafficlight.getRedYellowGreenState("n-s")
#     east_west_color = traci.trafficlight.getRedYellowGreenState("e-w")
#     if north_south_color == "G" and east_west_color == "r":
#         new_state = 0  # North-South green, East-West red
#     elif north_south_color == "r" and east_west_color == "G":
#         new_state = 1


############################################################################################################


# Q-Learning algorithm

# import traci
# import numpy as np

# # Initialize Q-values for each state-action pair
# Q = np.zeros((num_states, num_actions))

# # Set hyperparameters
# gamma = 0.9
# alpha = 0.1
# epsilon = 0.1

# # Main loop
# while traci.simulation.getMinExpectedNumber() > 0:
    
#     # Get current state
#     state = get_state()
    
#     # Choose action using epsilon-greedy policy
#     if np.random.rand() < epsilon:
#         action = np.random.choice(num_actions)
#     else:
#         action = np.argmax(Q[state])
    
#     # Take action and observe reward and next state
#     reward = take_action(action)
#     next_state = get_state()
    
#     # Update Q-value for current state-action pair
#     Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))
    
#     # Update traffic signal timing based on Q-values
#     timings = get_timings_from_Q(Q[state])
#     traci.trafficlight.setCompleteRedYellowGreenDefinition('TL', timings)
    
#     # Advance simulation by one step
#     traci.simulationStep()


