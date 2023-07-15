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
                #urban_mobility_simulation/models/20230502_SUMO_MA/osm.pedestrip.trips.xml",
                #urban_mobility_simulation/models/20230502_SUMO_MA/osm.pedestrian.trips.xml", \
    out_csv_name="urban_mobility_simulation/src/data/model_outputs/ppo_withPT_10000",
    use_gui=True,
    num_seconds=10000,
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
    

)

model.learn(total_timesteps=10000)
