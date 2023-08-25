import traci
import pandas as pd
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
import ma_environment.custom_envs as custom_env
import supersuit as ss
import numpy as np

# IMPORTANT! # make sure tls logic for selected traffic lights is set to 'static' in the sumo config (net) file
# and run in debug mode


# initialize SUMO environment using eval environment (collects different data during compute info)
env = custom_env.MA_grid_eval(use_gui=False,
                            reward_fn = 'queue',
                            traffic_lights= ['tls_159','tls_160', 'tls_161'],
                            out_csv_name='/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/evaluation/queue_200_1800steps',
                            begin_time=25200,
                            num_seconds=9000,
                            time_to_teleport=300)

# wrap observation space to have one common observation space for all agents
env = ss.pad_observations_v0(env)

# wrap action space to have one common action space for all agents (based on largest action space)
env = ss.pad_action_space_v0(env)

# wrap pettingzoo env 
env = ss.pettingzoo_env_to_vec_env_v1(env)

# oncatenate envs
env = ss.concat_vec_envs_v1(env, 1, num_cpus=4, base_class="stable_baselines3")

# wrap with monitor wrapper
env = VecMonitor(env)


# Load specific trained model
model = PPO.load('/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/logs/queue_200.zip')

# Get state of environment
observation = env.reset()

# Initialize SUMO simulation
traci.start(["sumo", "-c", "urban_mobility_simulation/models/20230718_sumo_ma/osm.sumocfg"])

# Initialize the lists to hold data
data = []

# Traffic lights to monitor / get controlled lanes from
tls = ['tls_159','tls_160', 'tls_161']

# Run the simulation for step from 25200 to 34200
for step in range(1800):
    # Advance simulation step
    traci.simulationStep()
    
    controlled_lanes = list(set(item for sublist in (traci.trafficlight.getControlledLanes(ts) for ts in tls) for item in sublist))

    # Let model decide based on the current environment state
    actions, _ = model.predict(observation, state=None, deterministic=True)

    # Apply the model's action to simulation
    observation, reward, done, information = env.step(actions) # step takes 5 seconds for one simulation step --> 1800 steps = 9000 seconds = 2.5 hours

    # Collect your required data
    local_vehicles = [item for sublist in (traci.lane.getLastStepVehicleIDs(lane_id) for lane_id in controlled_lanes) for item in sublist]
    
    CO2_emissions = sum([traci.vehicle.getCO2Emission(vehicle_id) for vehicle_id in local_vehicles])
    CO_emissions = sum([traci.vehicle.getCOEmission(vehicle_id) for vehicle_id in local_vehicles])
    HC_emissions = sum([traci.vehicle.getHCEmission(vehicle_id) for vehicle_id in local_vehicles])
    PMx_emissions = sum([traci.vehicle.getPMxEmission(vehicle_id) for vehicle_id in local_vehicles])
    NOx_emissions = sum([traci.vehicle.getNOxEmission(vehicle_id) for vehicle_id in local_vehicles])
    waiting_time = sum([traci.vehicle.getWaitingTime(vehicle_id) for vehicle_id in local_vehicles])
    total_num_stops = sum([traci.vehicle.getStopState(vehicle_id) for vehicle_id in local_vehicles])
    current_reward = np.mean(reward)
        
    # Append to data list
    data.append([step, CO2_emissions, CO_emissions, HC_emissions, PMx_emissions, NOx_emissions, waiting_time, total_num_stops, current_reward])

# Close the TraCI connection
traci.close()

columns = ['step', 'CO2_emissions', 'CO_emissions', 'HC_emissions', 'PMx_emissions', 'NOx_emissions', 'waiting_time', 'total_num_stops', 'current_reward']

df = pd.DataFrame(data, columns=columns)
df.to_csv('urban_mobility_simulation/src/data/evaluation/queue_df.csv', index=False)

