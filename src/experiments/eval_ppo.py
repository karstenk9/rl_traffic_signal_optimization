import numpy as np
import pandas as pd
import supersuit as ss
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import sys
from pathlib import Path

import ma_environment.custom_envs as custom_env

assert len(sys.argv) == 2, "Filename of the trained model must be passed."
model_path = Path(sys.argv[1])
assert model_path.exists(), f"File {model_path} does not exist."
name = model_path.stem


env = custom_env.MA_grid_eval(use_gui=False,
                            reward_fn = 'diff-waiting-time',
                            traffic_lights= ['tls_159','tls_160', 'tls_161'],
                            out_csv_name= name + "_eval1",
                            begin_time=25200,
                            num_seconds=9000,
                            time_to_teleport=300)


max_time = env.unwrapped.env.sim_max_time
delta_time = env.unwrapped.env.delta_time

#wrap observation space to have one common observation space for all agents
env = ss.pad_observations_v0(env)

#wrap action space to have one common action space for all agents (based on largest action space)
env = ss.pad_action_space_v0(env)

#wrap pettingzoo env
env = ss.pettingzoo_env_to_vec_env_v1(env)

#concatenate envs
env = ss.concat_vec_envs_v1(env, 1, num_cpus=4, base_class="stable_baselines3")

env = VecMonitor(env)

model = PPO.load(model_path, env=env)

obs = env.reset()

# Traffic lights to monitor / get controlled lanes from
tls = ['tls_159','tls_160', 'tls_161']
controlled_lanes = list(set(item for sublist in (traci.trafficlight.getControlledLanes(ts) for ts in tls) for item in sublist))

data = []

for t in range(25200, 34200, delta_time):
    
    actions, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(actions)
    
    print(actions)
    
    # Get # of vehicles on the lanes
    local_vehicle_ids = [item for sublist in (traci.lane.getLastStepVehicleIDs(lane_id) for lane_id in controlled_lanes) for item in sublist]
    num_vehicles = len(local_vehicle_ids) 
   
    # Get vehicle types
    vehicle_types = [] if num_vehicles == 0 else [traci.vehicle.getTypeID(vehicle_id) for vehicle_id in local_vehicle_ids] if num_vehicles > 0 else []

    # Get average speed of vehicles on the lanes
    avg_speed = 0.0 if num_vehicles == 0 else np.mean([traci.vehicle.getSpeed(vehicle_id) for vehicle_id in local_vehicle_ids])

    # get CO2 emissions for controlled lanes
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
    
    data.append([num_vehicles, vehicle_types, avg_speed, local_CO2_emission, local_CO_emission, local_HC_emission,
             local_PMx_emission, local_NOx_emission, local_fuel_consumption, local_noise_emission, 
             local_waiting_time, local_stopped_vehicles, actions,
             tls159_phase, tls159_phase_duration, tls159_state,
             tls160_phase, tls160_phase_duration, tls160_state,
             tls161_phase, tls161_phase_duration, tls161_state])
    # with open('<emission_output_path>/<emission_output_file_name>.csv', 'a') as f:
    #     f.write('{},{},{:.3f}\n'.format(t, 'controlled_lanes', local_CO2_emission))
    # # get lane data
    # lane_data = traci.lanearea.getLastStepOccupancy(lane_area_ids=controlled_lanes)
    # # save to file
    # with open('<lane_output_path>/<lane_output_file_name>.csv', 'a') as f:
    #     f.write('{},{},{},{},{},{},{}\n'.format(t, lane_data['tls_159_0'], lane_data['tls_159_1'], lane_data['tls_160_0'], lane_data['tls_160_1'], lane_data['tls_161_0'], lane_data['tls_161_1']))

columns = ['num_vehicles', 'vehicle_types', 'avg_speed', 'localCO2Emission', 'localCOEmission', 'localHCEmission',
           'localPMxEmission', 'localNOxEmission', 'local_fuel_consumption','localNoiseEmission',
           'localWaitingTime', 'localStoppedVehicles', 'actions',
           'tls159_phase', 'tls159_phase_duration', 'tls159_state',
           'tls160_phase', 'tls160_phase_duration', 'tls160_state',
           'tls161_phase', 'tls161_phase_duration', 'tls161_state']

df = pd.DataFrame(data, columns=columns)
df.to_csv( name + "_eval_df.csv", index=False)

env.close()