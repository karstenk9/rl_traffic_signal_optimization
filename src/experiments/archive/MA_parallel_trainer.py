import os
import sys

import platform
if platform.system() != "Linux":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)  # we need to import python modules from the $SUMO_HOME/tools directory
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    import traci
else:
    import libsumo as traci
    
import numpy as np
import pandas as pd
import ray
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import supersuit as ss

#import sumo_rl
import ma_environment.custom_envs as custom_env

# if __name__ == "__main__":
#     ray.init()

#     env_name = "MA_grid"

#     register_env(
#         env_name,
#         lambda _: ParallelPettingZooEnv(
#             custom_env.parallel_env(
#                 net_file='/home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/osm.net.xml, \
#                         /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
#                         /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/additional_tls.xml, \
#                         /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/stops.add.xml, \
#                         /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/osm.poly.xml, \
#                         /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/vtypes.xml',
#                 route_file='/home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/routes_veh.xml, \
#                             /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/bicycle_routes.xml,\
#                             /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/motorcycle_routes.xml,\
#                             /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/trucks_routes.xml, \
#                             /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.add.xml',
#                 out_csv_name='/home/inestp01/rl_traffic_signal_optimization/src/data/model_outputs/',
#                 use_gui=True,
#                 num_seconds=80000,
#                 begin_time=19800,
#                 time_to_teleport=300,
#                 additional_sumo_cmd="--scale 0.5"
#             )
#         ),
#     )

env_name = "MA_grid"

def env_creator(args):
    env = custom_env.MA_grid_new(
                net_file = "/home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/osm.net.xml, \
                            /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
                            /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/stops.add.xml, \
                            /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/vtypes.xml, \
                            /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/osm.poly.xml",
                route_file ="/home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/veh_routes.xml, \
                            /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.xml, \
                            /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/truck_routes.xml, \
                            /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/bicycle_routes.xml, \
                            /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/motorcycle_routes.xml",
                out_csv_name='/home/inestp01/rl_traffic_signal_optimization/src/data/model_outputs/MA_grid_combinedemissionAllTest',
                use_gui=False,
                num_seconds=30000,
                begin_time=19800,
                time_to_teleport=300,
                reward_fn='combined_emission',
                sumo_warnings=False)
    return env

register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
env = ParallelPettingZooEnv(env_creator({}))
obs_space = env.observation_space
act_space = env.action_space
print('Obersation space:', obs_space)
print('Action space:', act_space)

def gen_policy(i):
    config = {
        "gamma": 0.95,
    }
    return (None, obs_space, act_space, config)

policies = {"policy_0": gen_policy(0)}

policy_ids = list(policies.keys())

config = (
    PPOConfig()
    .environment(env=env_name, disable_env_checking=True)
    .rollouts(num_rollout_workers=3, rollout_fragment_length='auto')
    .training(
        train_batch_size=1024,
        lr=2e-5,
        gamma=0.95,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        sgd_minibatch_size=64,
        num_sgd_iter=10,
    )
#    .debugging(log_level="ERROR")
    .framework(framework="torch")
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "4")))
)

ppo = config.build()

for i in range(100):
    
    #print("Training iteration {}...".format(i))
    result = ppo.train()
    print("Training iteration {}...".format(i))
    print(pretty_print(result))

ppo.save("/home/inestp01/rl_traffic_signal_optimization/src/data/model_outputs/ppo_rllib_AllEmission")

#result =  
tune.run(
    "PPO",
    name="PPO",
    stop={"timesteps_total": 100000},
    checkpoint_freq=10,
    local_dir="~/Documents/ray_results/" + env_name,
    config=config.to_dict(),
   )

#print(pretty_print(result))
