import os
import sys

import gym


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import traci
from stable_baselines3.ppo.ppo import PPO

import ma_environment.env as custom_env
import supersuit as ss


env = custom_env.parallel_env(
    net_file='urban_mobility_simulation/models/20230718_sumo_ma/osm.net.xml, \
            urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
            urban_mobility_simulation/models/20230718_sumo_ma/pt/stops.add.xml, \
            urban_mobility_simulation/models/20230718_sumo_ma/osm.poly.xml, \
            urban_mobility_simulation/models/20230718_sumo_ma/pt/vtypes.xml',
    route_file='urban_mobility_simulation/models/20230718_sumo_ma/veh_routes.xml, \
                urban_mobility_simulation/models/20230718_sumo_ma/bicycle_routes.xml,\
                urban_mobility_simulation/models/20230718_sumo_ma/motorcycle_routes.xml,\
                urban_mobility_simulation/models/20230718_sumo_ma/truck_routes.xml, \
                urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.xml',\
    out_csv_name="urban_mobility_simulation/src/data/model_outputs/1108_ppo_multi_20000",
    use_gui=False,
    num_seconds=30000,
    yellow_time=4,
    min_green=5,
    max_green=60,
    time_to_teleport=300,
    fixed_ts=False,
    begin_time=19800,
    sumo_warnings=False,
    additional_sumo_cmd="--emission-output urban_mobility_simulation/src/data/emission_output/emission_info_ppo_multi.xml",
)

# env = sumo_rl.MA_grid(net_file='urban_mobility_simulation/models/20230718_sumo_ma/osm.net.xml,',
#                   route_file='urban_mobility_simulation/models/20230718_sumo_ma/routes_nm.xml,',
#                   use_gui=True,
#                   num_seconds=30000,
#                   begin_time=19800,
#                   time_to_teleport=300)
# observations = env.reset()

#env = ss.concat_vec_envs_v1(env, 3, num_cpus=1, base_class="stable_baselines3")
#env = VecMonitor(env)

import sumo_rl

# env = sumo_rl.parallel_env(
#     net_file='urban_mobility_simulation/models/20230718_sumo_ma/osm.net.xml, \
#             urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
#             urban_mobility_simulation/models/20230718_sumo_ma/pt/stops.add.xml, \
#             urban_mobility_simulation/models/20230718_sumo_ma/osm.poly.xml, \
#             urban_mobility_simulation/models/20230718_sumo_ma/pt/vtypes.xml',
#     route_file='urban_mobility_simulation/models/20230718_sumo_ma/veh_routes.xml, \
#                 urban_mobility_simulation/models/20230718_sumo_ma/bicycle_routes.xml,\
#                 urban_mobility_simulation/models/20230718_sumo_ma/motorcycle_routes.xml,\
#                 urban_mobility_simulation/models/20230718_sumo_ma/truck_routes.xml, \
#                 urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.xml',\
#     out_csv_name="urban_mobility_simulation/src/data/model_outputs/ppo_multi_parallel_10000",
#     use_gui=True,
#     num_seconds=22000,
#     yellow_time=4,
#     min_green=5,
#     max_green=60,
#     time_to_teleport=300,
#     begin_time=19800,
#     fixed_ts=False,
# )

rollout_fragment_length = 150
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=4, base_class='stable_baselines3')

save_path = f"urban_mobility_simulation/src/data/logs/PPO_{train_timesteps}"
log_path = f"urban_mobility_simulation/src/data/logs/PPO_logs"

model = PPO('MlpPolicy', 
            env, 
            tensorboard_log=log_path, 
            verbose=1, 
            gamma=0.95, 
            ent_coef=0.01, 
            n_steps=rollout_fragment_length,
            learning_rate=2e-5, 
            vf_coef=1, 
            max_grad_norm=0.9, 
            gae_lambda=1.0, 
            n_epochs=30, 
            clip_range=0.3,
            batch_size=128,
            device='auto'
            )
train_timesteps = 200000


model.learn(total_timesteps=train_timesteps,
            tb_log_name=f"PPO_{train_timesteps}"
            )
model.save(f"MA_test_{train_timesteps}")
