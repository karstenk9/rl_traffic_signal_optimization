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

from environment.env import SumoEnvironment


env = SumoEnvironment(
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
    out_csv_name="urban_mobility_simulation/src/data/model_outputs/ppo_multi_test",
    single_agent=False,
    use_gui=True,
    num_seconds=3600,
    yellow_time=4,
    min_green=5,
    max_green=60,
    time_to_teleport=300,
    fixed_ts=False,
)

model = PPO(
    env=env,
    policy="MlpPolicy",
    learning_rate=3e-4,
    verbose=1,
    n_steps=50
)

model.learn(total_timesteps=10000)
