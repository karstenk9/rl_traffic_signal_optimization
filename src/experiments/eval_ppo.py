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
                            out_csv_name= name + "_eval",
                            begin_time=25200,
                            num_seconds=9000,
                            time_to_teleport=300,
                            )


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
controlled_vehicles = set()
vehicle_departures = dict()
vehicle_arrivals = dict()
vehicle_waiting_times = dict()

data = []

for t in range(25200, 34200, delta_time):

    for vehicle in traci.simulation.getDepartedIDList():
        vehicle_departures[vehicle] = t

    for vehicle in traci.simulation.getArrivedIDList():
        vehicle_arrivals[vehicle] = t

    for vehicle in traci.vehicle.getIDList():
        vehicle_waiting_times[vehicle] = traci.vehicle.getAccumulatedWaitingTime(vehicle)


    actions, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(actions)

    print(actions)

    # Get # of vehicles on the lanes
    local_vehicle_ids = [item for sublist in (traci.lane.getLastStepVehicleIDs(lane_id) for lane_id in controlled_lanes) for item in sublist]
    controlled_vehicles.update(local_vehicle_ids)
    num_vehicles = len(local_vehicle_ids)

    # Get vehicle types
    vehicle_types = [] if num_vehicles == 0 else [traci.vehicle.getTypeID(vehicle_id) for vehicle_id in local_vehicle_ids] if num_vehicles > 0 else []

    # Get average speed of vehicles on the lanes
    avg_speed = 0.0 if num_vehicles == 0 else np.mean([traci.vehicle.getSpeed(vehicle_id) for vehicle_id in local_vehicle_ids])

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

    data.append([num_vehicles, vehicle_types, avg_speed,
             local_waiting_time, local_stopped_vehicles, actions,
             tls159_phase, tls159_phase_duration, tls159_state,
             tls160_phase, tls160_phase_duration, tls160_state,
             tls161_phase, tls161_phase_duration, tls161_state])

columns = ['num_vehicles', 'vehicle_types', 'avg_speed',
           'localWaitingTime', 'localStoppedVehicles', 'actions',
           'tls159_phase', 'tls159_phase_duration', 'tls159_state',
           'tls160_phase', 'tls160_phase_duration', 'tls160_state',
           'tls161_phase', 'tls161_phase_duration', 'tls161_state']

df = pd.DataFrame(data, columns=columns)
df.to_csv(name + "-eval-df.csv", index=False)

vehicle_times = pd.DataFrame({
    "depart_time": vehicle_departures,
    "arrive_time": vehicle_arrivals,
    "waiting_time": vehicle_waiting_times
}).reset_index().rename({"index": "vehicle_id"}, axis=1)
# vehicle_missing_times = vehicle_times["vehicle_id"][vehicle_times.isna().any(axis=1)].tolist()
# if vehicle_missing_times:
#     print(f"Departure / arrival / waiting time missing for vehicles: {vehicle_missing_times}. Ignoring them.")
#     vehicle_times.dropna(inplace=True)
# vehicle_times["depart_time"] = vehicle_times["depart_time"].astype(int)
# vehicle_times["arrive_time"] = vehicle_times["arrive_time"].astype(int)
# vehicle_times["waiting_time"] = vehicle_times["waiting_time"].astype(int)
vehicle_times["is_controlled_vehicle"] = vehicle_times["vehicle_id"].apply(lambda veh: veh in controlled_vehicles)
vehicle_times.to_csv(name + "-vehicle-times.csv", index=False)

env.close()