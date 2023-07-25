import os
import sys

import gym

import platform

if platform.system() != "Linux":
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")
    import traci
else:
    import libsumo as traci
    
import numpy as np
#import traci
from stable_baselines3.ppo.ppo import PPO

from environment.env import SumoEnvironment


env = SumoEnvironment(
    net_file="urban_mobility_simulation/models/20230718_sumo_ma/osm.net_1.xml, \
              urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
              urban_mobility_simulation/models/20230718_sumo_ma/pt/stops.add.xml, \
              urban_mobility_simulation/models/20230718_sumo_ma/osm.poly.xml, \
              urban_mobility_simulation/models/20230718_sumo_ma/pt/vtypes.xml",
            # urban_mobility_simulation/models/20230718_sumo_ma/additional_tls.xml, \
    single_agent=True,
    route_file="urban_mobility_simulation/models/20230718_sumo_ma/routes_nm.xml, \
                urban_mobility_simulation/models/20230718_sumo_ma/bicycle_routes.xml,\
                urban_mobility_simulation/models/20230718_sumo_ma/motorcycle_routes.xml,\
                urban_mobility_simulation/models/20230718_sumo_ma/trucks_routes.xml, \
                urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.add.xml",
    out_csv_name="urban_mobility_simulation/src/data/model_outputs/ppo_new_local_test",
    use_gui=False,
    num_seconds=43200,
    yellow_time=4,
    min_green=5,
    max_green=60,
    time_to_teleport=300,
    fixed_ts=False,
    additional_sumo_cmd="--scale 0.5",
    begin_time=10000,
)
model = PPO(
    env=env,
    policy="MlpPolicy",
    learning_rate=3e-4,
    verbose=1,
    gamma=0.95,
    gae_lambda=0.9,
    ent_coef=0.1,
    vf_coef=0.25,
    batch_size=64
)

model.learn(total_timesteps=43200)
