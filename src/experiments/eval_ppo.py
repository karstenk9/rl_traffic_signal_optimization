import os

import numpy as np
import pandas as pd
import supersuit as ss
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import sys
from pathlib import Path
import json

import ma_environment.custom_envs as custom_env
from _utils import change_sumo_config_status, VehicleTimesListener

N_ITERATIONS = 6
TLS_IDS = ['tls_159','tls_160', 'tls_161']
BEGIN_TIME = 25200
NUM_SECONDS = 9000
DELTA_TIME = 5
TLS_CONFIG_PATH = "../../models/20230718_sumo_ma/additional_tls.xml"

assert len(sys.argv) == 2, "Name of the trained model (subdirectory of outputs) or 'baseline' must be passed."
name = sys.argv[1]

for i in range(N_ITERATIONS):
    print(f"ITERATION {i}")

    if name == "baseline":
        print("Evaluating the baseline - all traffic lights will be set to actuated.")
        evaluate_model = False
        change_sumo_config_status(TLS_CONFIG_PATH, TLS_IDS, "actuated")
        traci.start(['sumo', "-c", "../../models/20230718_sumo_ma/osm.sumocfg", "--time-to-teleport", "300"])
        change_sumo_config_status(TLS_CONFIG_PATH, TLS_IDS, "static")
    else:
        change_sumo_config_status(TLS_CONFIG_PATH, TLS_IDS, "static")
        model_path = Path(f"../../outputs/{name}/model.zip")
        assert model_path.exists(), f"File {model_path} does not exist."
        print(f"Evaluating model {name}")
        evaluate_model = True
        env = custom_env.MA_grid_eval(use_gui=False,
                                    reward_fn = 'diff-waiting-time',
                                    traffic_lights= ['tls_159','tls_160', 'tls_161'],
                                    # out_csv_name= name + "_eval",
                                    begin_time=BEGIN_TIME,
                                    num_seconds=NUM_SECONDS,
                                    time_to_teleport=300,
                                    )
        assert DELTA_TIME == env.unwrapped.env.delta_time
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 1, num_cpus=4, base_class="stable_baselines3")
        env = VecMonitor(env)
        model = PPO.load(model_path, env=env)
        obs = env.reset()

    output_path = f"../../outputs/{name}/evaluation"
    os.makedirs(output_path, exist_ok=True)

    controlled_lanes = list(set(item for sublist in (traci.trafficlight.getControlledLanes(ts) for ts in TLS_IDS) for item in sublist))
    data = []

    vehicle_times_listener = VehicleTimesListener(controlled_lanes)
    traci.addStepListener(vehicle_times_listener)

    for t in range(BEGIN_TIME, BEGIN_TIME+NUM_SECONDS, DELTA_TIME):

        if evaluate_model:
            actions, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(actions)
            print(actions)
        else:
            for _ in range(DELTA_TIME):
                traci.simulationStep()
            actions = []

        # Get # of vehicles on the lanes
        local_vehicle_ids = [item for sublist in (traci.lane.getLastStepVehicleIDs(lane_id) for lane_id in controlled_lanes) for item in sublist]
        num_vehicles = len(local_vehicle_ids)

        # Get vehicle types
        vehicle_types = [] if num_vehicles == 0 else [traci.vehicle.getTypeID(vehicle_id) for vehicle_id in local_vehicle_ids] if num_vehicles > 0 else []

        # Get average speed of vehicles on the lanes
        avg_speed = 0.0 if num_vehicles == 0 else np.mean([traci.vehicle.getSpeed(vehicle_id) for vehicle_id in local_vehicle_ids])

        local_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in controlled_lanes)
        local_stopped_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in controlled_lanes)

        # Get TLS Info (to compare phases and transitions)
        tls159_phase = traci.trafficlight.getPhase(TLS_IDS[0])
        tls159_phase_duration = traci.trafficlight.getPhaseDuration(TLS_IDS[0])
        tls159_state = traci.trafficlight.getRedYellowGreenState(TLS_IDS[0])
        tls160_phase = traci.trafficlight.getPhase(TLS_IDS[1])
        tls160_phase_duration = traci.trafficlight.getPhaseDuration(TLS_IDS[1])
        tls160_state = traci.trafficlight.getRedYellowGreenState(TLS_IDS[1])
        tls161_phase = traci.trafficlight.getPhase(TLS_IDS[2])
        tls161_phase_duration = traci.trafficlight.getPhaseDuration(TLS_IDS[2])
        tls161_state = traci.trafficlight.getRedYellowGreenState(TLS_IDS[2])

        data.append([num_vehicles, vehicle_types, avg_speed,
                 local_waiting_time, local_stopped_vehicles, actions,
                 tls159_phase, tls159_phase_duration, tls159_state,
                 tls160_phase, tls160_phase_duration, tls160_state,
                 tls161_phase, tls161_phase_duration, tls161_state])

    if evaluate_model:
        env.close()
    else:
        traci.close()

    columns = ['num_vehicles', 'vehicle_types', 'avg_speed',
               'localWaitingTime', 'localStoppedVehicles', 'actions',
               'tls159_phase', 'tls159_phase_duration', 'tls159_state',
               'tls160_phase', 'tls160_phase_duration', 'tls160_state',
               'tls161_phase', 'tls161_phase_duration', 'tls161_state']

    simulation_states = pd.DataFrame(data, columns=columns)   # for each timestep one row
    simulation_states.to_csv(output_path + f"/simulation-states_{i}.csv", index=False)

    vehicle_times = pd.DataFrame({
        "depart_time": vehicle_times_listener.vehicle_departures,
        "arrive_time": vehicle_times_listener.vehicle_arrivals,
    }).reset_index().rename({"index": "vehicle_id"}, axis=1)
    vehicle_times["travel_time"] = vehicle_times["arrive_time"] - vehicle_times["depart_time"]
    vehicle_times["is_controlled_vehicle"] = vehicle_times["vehicle_id"].apply(lambda veh: veh in vehicle_times_listener.controlled_vehicles)
    vehicle_times["is_teleported_vehicle"] = vehicle_times["vehicle_id"].apply(lambda veh: veh in vehicle_times_listener.teleported_vehicles)
    vehicle_times.to_csv(output_path + f"/vehicle-times_{i}.csv", index=False)

    n_veh = vehicle_times["travel_time"].count()
    vehicle_avg_travel_time = vehicle_times["travel_time"].mean()
    vehicle_times_controlled = vehicle_times[vehicle_times["is_controlled_vehicle"]]
    n_veh_controlled = vehicle_times_controlled["travel_time"].count()
    vehicle_avg_travel_time_controlled = vehicle_times_controlled["travel_time"].mean()
    lane_avg_waiting_time = simulation_states['localWaitingTime'].mean()

    eval_results = {
        "vehicle_travel_time": {
            "all": {"number_vehicles": int(n_veh), "avg": vehicle_avg_travel_time},
            "tls_controlled": {"number_vehicles": int(n_veh_controlled), "avg": vehicle_avg_travel_time_controlled},
        },
        "lane_waiting_time": {"avg": lane_avg_waiting_time}
    }
    with open(output_path + f"/eval_results_{i}.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
