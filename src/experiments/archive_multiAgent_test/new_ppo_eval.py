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

### adjust path for model to evaluate
model_path = '/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/logs/average-speed_200.zip'

### adjust output-path for specific model
output_path = '/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/evaluation/average-speed_new_eval.csv'

### adjust reward function
reward_fn = 'average-speed'

### set traffic lights to monitor / get controlled lanes from
tls = ['tls_159','tls_160', 'tls_161']

# initialize SUMO environment using eval environment (collects different data during compute info)
env = custom_env.MA_grid_eval(use_gui=True,
                            reward_fn = reward_fn,
                            traffic_lights= ['tls_159','tls_160', 'tls_161'],
                            out_csv_name='/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/evaluation/average-speed_200_new',
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

# load the PPO model
model = PPO.load(model_path, env=env)

# initialize SUMO simulation
traci.start(["sumo", "-c", "urban_mobility_simulation/models/20230718_sumo_ma/osm.sumocfg"]) # sumocfg specified runtime from 25200 to 34200

# Create an empty DataFrame to store the data
columns = [
    "episode",
    "num_vehicles",
    "vehicle_types",
    "avg_speed",
    "local_CO2_emission",
    "local_CO_emission",
    "local_HC_emission",
    "local_PMx_emission",
    "local_NOx_emission",
    "local_fuel_consumption",
    "local_noise_emission",
    "local_waiting_time",
    "local_stopped_vehicles",
    "tls159_phase",
    "tls159_phase_duration",
    "tls159_state",
    "tls160_phase",
    "tls160_phase_duration",
    "tls160_state",
    "tls161_phase",
    "tls161_phase_duration",
    "tls161_state",
    "reward",
]

data = pd.DataFrame(columns=columns)

# evaluate the model 
num_episodes = 1

for episode in range(num_episodes):
    
    # # get state of the environment
    # obs = env.reset()
    # done = False
    # episode_reward = 0
    # episode_emission = 0
    # while not done:
    #     action, _ = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     episode_reward += reward[0]
        
    
    obs = env.reset()
    done = [False for _ in range(len(tls))]
    episode_reward = [0 for _ in range(len(tls))]
    for step in range(1800):
        
        # Access data through TraCI
        traci.simulationStep()
        
        actions, _ = model.predict(obs, state=None, deterministic=True)
        obs, rewards, done, info = env.step(actions)
        episode_reward += rewards

        # ---- Get local traffic statistics ----
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

        # Append the data to the DataFrame
        row_data = {
            "episode": episode,
            "num_vehicles": num_vehicles,
            "vehicle_types": vehicle_types,
            "avg_speed": avg_speed,
            "local_CO2_emission": local_CO2_emission,
            "local_CO_emission": local_CO_emission,
            "local_HC_emission": local_HC_emission,
            "local_PMx_emission": local_PMx_emission,
            "local_NOx_emission": local_NOx_emission,
            "local_fuel_consumption": local_fuel_consumption,
            "local_noise_emission": local_noise_emission,
            "local_waiting_time": local_waiting_time,
            "local_stopped_vehicles": local_stopped_vehicles,
            "tls159_phase": tls159_phase,
            "tls159_phase_duration": tls159_phase_duration,
            "tls159_state": tls159_state,
            "tls160_phase": tls160_phase,
            "tls160_phase_duration": tls160_phase_duration,
            "tls160_state": tls160_state,
            "tls161_phase": tls161_phase,
            "tls161_phase_duration": tls161_phase_duration,
            "tls161_state": tls161_state,
            "reward": episode_reward
        }
        data = data.append(row_data, ignore_index=True)

    print(f"Episode {episode} reward: {episode_reward}")

print(f"Average reward over {num_episodes} episodes: {np.mean(data['reward'])}")


# Save the DataFrame to a CSV file
data.to_csv(output_path, index=False)