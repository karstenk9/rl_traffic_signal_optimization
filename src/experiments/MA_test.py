# import os
# import sys

import gym


# if "SUMO_HOME" in os.environ:
#     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
#     sys.path.append(tools)
# else:
#     sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import traci
from stable_baselines3.ppo.ppo import PPO

# import environment.env as custom_env


# env = custom_env.parallel_env(
#     net_file='urban_mobility_simulation/models/20230718_sumo_ma/osm.net.xml, \
#             urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
#             urban_mobility_simulation/models/20230718_sumo_ma/additional_tls.xml, \
#             urban_mobility_simulation/models/20230718_sumo_ma/pt/stops.add.xml, \
#             urban_mobility_simulation/models/20230718_sumo_ma/osm.poly.xml, \
#             urban_mobility_simulation/models/20230718_sumo_ma/pt/vtypes.xml',
#     route_file='urban_mobility_simulation/models/20230718_sumo_ma/routes_nm.xml, \
#                 urban_mobility_simulation/models/20230718_sumo_ma/bicycle_routes.xml,\
#                 urban_mobility_simulation/models/20230718_sumo_ma/motorcycle_routes.xml,\
#                 urban_mobility_simulation/models/20230718_sumo_ma/trucks_routes.xml, \
#                 urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.add.xml',\
#     out_csv_name="urban_mobility_simulation/src/data/model_outputs/ppo_multi_10000",
#     single_agent=True,
#     use_gui=True,
#     num_seconds=43200,
#     yellow_time=4,
#     min_green=5,
#     max_green=60,
#     time_to_teleport=300,
#     fixed_ts=False,
# )
# model = PPO(
#     env=env,
#     policy="MlpPolicy",
#     learning_rate=3e-4,
#     verbose=1,
# )

# model.learn(total_timesteps=10000)


# import sumo_rl
# import supersuit as ss
# from stable_baselines3.common.vec_env import VecMonitor
# from pettingzoo.utils import conversions

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
import supersuit as ss

# env = sumo_rl.MA_grid(net_file='urban_mobility_simulation/models/20230718_sumo_ma/osm.net.xml,',
#                   route_file='urban_mobility_simulation/models/20230718_sumo_ma/routes_nm.xml,',
#                   use_gui=True,
#                   num_seconds=30000,
#                   begin_time=19800,
#                   time_to_teleport=300,
#                   additional_sumo_cmd="--scale 0.5")

env = sumo_rl.parallel_env(
    net_file='urban_mobility_simulation/models/20230718_sumo_ma/osm.net.xml, \
            urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
            urban_mobility_simulation/models/20230718_sumo_ma/additional_tls.xml, \
            urban_mobility_simulation/models/20230718_sumo_ma/pt/stops.add.xml, \
            urban_mobility_simulation/models/20230718_sumo_ma/osm.poly.xml, \
            urban_mobility_simulation/models/20230718_sumo_ma/pt/vtypes.xml',
    route_file='urban_mobility_simulation/models/20230718_sumo_ma/routes_nm.xml, \
                urban_mobility_simulation/models/20230718_sumo_ma/bicycle_routes.xml,\
                urban_mobility_simulation/models/20230718_sumo_ma/motorcycle_routes.xml,\
                urban_mobility_simulation/models/20230718_sumo_ma/trucks_routes.xml, \
                urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.add.xml',\
    out_csv_name="urban_mobility_simulation/src/data/model_outputs/ppo_multi_parallel_10000",
    use_gui=True,
    num_seconds=43200,
    yellow_time=4,
    min_green=5,
    max_green=60,
    time_to_teleport=300,
    begin_time=19800,
    fixed_ts=False,
    additional_sumo_cmd="--scale 0.25"
)

env_steps = 1000  # 2 * env.width * env.height  # Code uses 1.5 to calculate max_steps
rollout_fragment_length = 50
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

model = PPO('MlpPolicy', 
            env, 
            tensorboard_log=f"/tmp/MA_test", 
            verbose=3, 
            gamma=0.95, 
            n_steps=rollout_fragment_length, 
            ent_coef=0.01, 
            learning_rate=2e-5, 
            vf_coef=1, 
            max_grad_norm=0.9, 
            gae_lambda=1.0, 
            n_epochs=30, 
            clip_range=0.3,
            batch_size=150)
train_timesteps = 100000
model.learn(total_timesteps=train_timesteps)
model.save(f"MA_test_{train_timesteps}")
