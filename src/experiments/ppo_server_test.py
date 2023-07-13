import os
import sys

import gym
import numpy as np

import platform
import libsumo as traci

from sumolib import checkBinary

from stable_baselines3.ppo.ppo import PPO
from environment.env import SumoEnvironment

env = SumoEnvironment(
    net_file="urban_mobility_simulation/models/20230502_SUMO_MA/osm.net.xml, \
            urban_mobility_simulation/models/20230502_SUMO_MA/pt/stops.add.xml, \
            urban_mobility_simulation/models/20230502_SUMO_MA/osm.poly.xml",
    single_agent=True,
    route_file="urban_mobility_simulation/models/20230502_SUMO_MA/routes.xml, \
                urban_mobility_simulation/models/20230502_SUMO_MA/osm.bicycle.trips.xml,\
                urban_mobility_simulation/models/20230502_SUMO_MA/osm.motorcycle.trips.xml,\
                urban_mobility_simulation/models/20230502_SUMO_MA/osm.truck.trips.xml, \
                urban_mobility_simulation/models/20230502_SUMO_MA/pt/ptflows.rou.xml, \
                urban_mobility_simulation/models/20230502_SUMO_MA/osm.passenger.trips.xml", 
    out_csv_name="urban_mobility_simulation/src/data/model_outputs/ppo_withPT",
    use_gui=False,
    num_seconds=5000,
    yellow_time=4,
    min_green=5,
    max_green=60,
)

model = PPO(
    env=env,
    policy="MlpPolicy",
    learning_rate=3e-4,
    verbose=1
)

model.learn(total_timesteps=5000)
