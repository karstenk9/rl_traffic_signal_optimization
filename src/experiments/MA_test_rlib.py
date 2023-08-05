import os
import sys


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import pandas as pd
import ray
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import supersuit as ss

import sumo_rl

def env_creator():
    env = sumo_rl.MA_grid(net_file='urban_mobility_simulation/models/20230718_sumo_ma/osm.net.xml,',
                  route_file='urban_mobility_simulation/models/20230718_sumo_ma/routes_nm.xml,',
                  use_gui=True,
                  num_seconds=30000,
                  begin_time=19800,
                  time_to_teleport=300,
                  additional_sumo_cmd="--scale 0.5")
    return env


if __name__ == "__main__":
    env_name = "MA_grid"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    test_env = ParallelPettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy(i):
        config = {
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)

    policies = {"policy_0": gen_policy(0)}

    policy_ids = list(policies.keys())

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="~/ray_results/"+env_name,
        config={
            # Environment specific
            "env": env_name,
            # https://github.com/ray-project/ray/issues/10761
            "no_done_at_end": True,
            # "soft_horizon" : True,
            "num_gpus": 0,
            "num_workers": 2,
            "num_envs_per_worker": 1,
            "compress_observations": False,
            "batch_mode": 'truncate_episodes',
            "clip_rewards": False,
            "vf_clip_param": 500.0,
            "entropy_coeff": 0.01,
            # effective batch_size: train_batch_size * num_agents_in_each_environment [5, 10]
            # see https://github.com/ray-project/ray/issues/4628
            "train_batch_size": 1000,  # 5000
            "rollout_fragment_length": 50,  # 100
            "sgd_minibatch_size": 100,  # 500
            "vf_share_layers": False
            },
    )