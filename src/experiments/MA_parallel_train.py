import os
import shutil
import subprocess

import numpy as np
import supersuit as ss
import traci
from pyvirtualdisplay.smartdisplay import SmartDisplay
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from tqdm import trange

import environment.MA_parallel_env as environment


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

import environment.MA_parallel_env as environment


if __name__ == "__main__":
    ray.init()

    env_name = "MannheimNetwork"

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(
            environment.parallel_env(
                net_file='/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/models/20230718_sumo_ma/osm.net_1.xml, \
                        /Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
                        /Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/models/20230718_sumo_ma/pt/stops.add.xml, \
                        /Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/models/20230718_sumo_ma/osm.poly.xml, \
                        /Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/models/20230718_sumo_ma/pt/vtypes.xml',
                route_file='/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/models/20230718_sumo_ma/routes_nm.xml, \
                            /Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/models/20230718_sumo_ma/bicycle_routes.xml,\
                            /Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/models/20230718_sumo_ma/motorcycle_routes.xml,\
                            /Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/models/20230718_sumo_ma/trucks_routes.xml, \
                            /Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.add.xml',
                out_csv_name='/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/model_outputs/',
                use_gui=True,
                num_seconds=80000,
                begin_time=19800,
                time_to_teleport=300,
                additional_sumo_cmd="--scale 0.75"
            )
        ),
    )

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
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
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    print("Starting training")
    
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
