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
from ray import air, tune
#from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import (
    PPOConfig,
    PPOTF1Policy,
    PPOTF2Policy,
    PPOTorchPolicy,
)
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv, PettingZooEnv
from ray.rllib.algorithms.registry import get_trainer_class
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

import ma_environment.custom_envs as custom_env
import supersuit as ss


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
                    out_csv_name='/home/inestp01/rl_traffic_signal_optimization/src/data/model_outputs/MA_grid_emissionAllTest',
                    use_gui=False,
                    num_seconds=30000,
                    begin_time=19800,
                    time_to_teleport=300,
                    reward_fn='combined_emission',
                    sumo_warnings=False)
        return env
    
    
def select_policy(framework):
        if framework == "torch":
            return PPOTorchPolicy
        elif framework == "tf":
            return PPOTF1Policy
        else:
            return PPOTF2Policy
        
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "ppo_policy"


if __name__ == "__main__":
    ray.init(num_cpus=4, num_gpus=2)
   
    env_name = "MA_grid_new"

    # register env
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    # create env
    env = ParallelPettingZooEnv(env_creator({}))
    #get obs and action space
    #obs_space = env.observation_space
    #act_space = env.action_space
    
    algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=2)
    .environment(env=env_name, disable_env_checking=True)
    .framework(framework="torch")
    .build()
)

    for i in range(10):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")

    
    config = (
        PPOConfig()
        .environment(env=env_name, 
                     disable_env_checking=True
                     )
        .rollouts(num_rollout_workers=2, 
                  rollout_fragment_length='auto',
                  num_envs_per_worker=1)
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
    
        # .multiagent(
        #     policies=env.get_agent_ids(),
        #     policy_mapping_fn=policy_mapping_fn#(lambda agent_id, *args, **kwargs: agent_id)
        # )
        # .multiagent(
        #     policies={id: (PPOTF1Policy, env.observation_spaces[id], env.action_spaces[id], {}) for id in env.agents},
        #     policy_mapping_fn= (lambda id: id) 
        #)
        .debugging(log_level="ERROR")
        .framework(framework="tf2")
        .resources(num_gpus=2)
        .evaluation(evaluation_num_workers=1, 
                    evaluation_parallel_to_training=True, 
                    evaluation_interval=1)
    )
    
    algo = config.build()

    for _ in range(100):
        
        print('Training iteration: ', _)
        print(algo.train())

    print('Evaluating...')
    algo.evaluate() 
    
    # tune.Tuner(
    #     "PPO",
    #     #name="PPO",
    #     stop={"timesteps_total": 100000},
    #     checkpoint_freq=10,
    #     local_dir="~/ray_results/" + env_name,
    #     config=config.to_dict(),
    # ).fit()
    
    
    # policies = {'ppo_policy':(
    #             select_policy('torch'),
    #             obs_space,
    #             act_space,
    #             config,
    #         )}
    
    
    # config.multi_agent(
    #         policies=policies,
    #         policy_mapping_fn=policy_mapping_fn,
    #         policies_to_train=["ppo_policy"]
    # )
    
    #ppo = config.build()
    
    # tune.run(
    #     "PPO",
    #     name="PPO",
    #     stop={"timesteps_total": 100000},
    #     checkpoint_freq=10,
    #     local_dir="~/ray_results/" + env_name,
    #     config=config.to_dict()
    # ).fit()
    
    #ä######## here
    
    # results = tune.Tuner(
    # args.run,
    # run_config=air.RunConfig(
    #     stop=stop,
    # ),
    # param_space=config,
    # ).fit()

    # if not results:
    #     raise ValueError(
    #         "No results returned from tune.run(). Something must have gone wrong."
    #     )
    
    # ray.shutdown()







