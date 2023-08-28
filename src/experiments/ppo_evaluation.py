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
                            out_csv_name='/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/evaluation/queue_200',
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
model = PPO.load('/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/logs/queue_200.zip', env=env)

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

    # Apply the model's action to simulation and get new observation state
    observation, reward, done, information = env.step(actions) # step takes 5 seconds for one simulation step --> 1800 steps = 9000 seconds = 2.5 hours

    # Get lanes and vehicles on lanes controlled by traffic lights
    controlled_lanes = list(set(item for sublist in (traci.trafficlight.getControlledLanes(ts) for ts in tls) for item in sublist))
    local_vehicle_ids = [item for sublist in (traci.lane.getLastStepVehicleIDs(lane_id) for lane_id in controlled_lanes) for item in sublist]

    # Get # of vehicles on the lanes
    num_vehicles = len(local_vehicle_ids) 
   
    # Get vehicle types
    vehicle_types = [] if num_vehicles == 0 else [traci.vehicle.getTypeID(vehicle_id) for vehicle_id in local_vehicle_ids] if num_vehicles > 0 else []

    # Get average speed of vehicles on the lanes
    avg_speed = 0.0 if num_vehicles == 0 else np.mean([traci.vehicle.getSpeed(vehicle_id) for vehicle_id in local_vehicle_ids])

    # Get local emission and further measures
    local_CO2_emission = sum(traci.lane.getCO2Emission(lane_id) for lane_id in controlled_lanes)
    local_CO_emission = sum(traci.lane.getCOEmission(lane_id) for lane_id in controlled_lanes)
    local_HC_emission = sum(traci.lane.getHCEmission(lane_id) for lane_id in controlled_lanes)
    local_PMx_emission = sum(traci.lane.getPMxEmission(lane_id) for lane_id in controlled_lanes)
    local_NOx_emission = sum(traci.lane.getNOxEmission(lane_id) for lane_id in controlled_lanes)
    local_fuel_consumption = sum(traci.lane.getFuelConsumption(lane_id) for lane_id in controlled_lanes)
    local_noise_emission = sum(traci.lane.getNoiseEmission(lane_id) for lane_id in controlled_lanes)
    local_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in controlled_lanes)
    local_stopped_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in controlled_lanes)
    
    
    # Get TLS Info (to compare phases and transitions)
    tls159_phase = traci.trafficlight.getPhase(tls[0])
    tls159_phase_duration = traci.trafficlight.getPhaseDuration(tls[0])
    tls159_state = traci.trafficlight.getRedYellowGreenState(tls[0])
    tls160_phase = traci.trafficlight.getPhase(tls[1])
    tls160_phase_duration = traci.trafficlight.getPhaseDuration(tls[1])
    tls160_state = traci.trafficlight.getRedYellowGreenState(tls[1])
    tls161_phase = traci.trafficlight.getPhase(tls[2])
    tls161_phase_duration = traci.trafficlight.getPhaseDuration(tls[2])
    tls161_state = traci.trafficlight.getRedYellowGreenState(tls[2])
    

    # Append data to list for dataframe
    data.append([step, num_vehicles, vehicle_types, avg_speed, local_CO2_emission, local_CO_emission, local_HC_emission,
             local_PMx_emission, local_NOx_emission, local_fuel_consumption, local_noise_emission, 
             local_waiting_time, local_stopped_vehicles, 
             tls159_phase, tls159_phase_duration, tls159_state,
             tls160_phase, tls160_phase_duration, tls160_state,
             tls161_phase, tls161_phase_duration, tls161_state])

# Close the TraCI connection
traci.close()

# Create a DataFrame from the data
columns = ['Step', 'num_vehicles', 'vehicle_types', 'avg_speed', 'localCO2Emission', 'localCOEmission', 'localHCEmission',
           'localPMxEmission', 'localNOxEmission', 'local_fuel_consumption','localNoiseEmission',
           'localWaitingTime', 'localStoppedVehicles',
           'tls159_phase', 'tls159_phase_duration', 'tls159_state',
           'tls160_phase', 'tls160_phase_duration', 'tls160_state',
           'tls161_phase', 'tls161_phase_duration', 'tls161_state']

df = pd.DataFrame(data, columns=columns)
df.to_csv('urban_mobility_simulation/src/data/evaluation/queue_df.csv', index=False)

