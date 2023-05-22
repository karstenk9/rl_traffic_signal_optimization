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
# from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.ppo.ppo import PPO

from sumo_rl import SumoEnvironment


env = SumoEnvironment(
    #net_file="sumo-rl/nets/big-intersection/big-intersection.net.xml",
    net_file="urban_mobility_simulation/models/20230502_SUMO_MA/osm.net.xml",
    #net_file="urban_mobility_simulation/src/team_rl_test/data/cross.net.xml",
    single_agent=True,
    #route_file="sumo-rl/nets/big-intersection/routes.rou.xml",
    route_file="urban_mobility_simulation/models/20230502_SUMO_MA/routes.xml, urban_mobility_simulation/models/20230502_SUMO_MA/osm.bicycle.trips.xml,urban_mobility_simulation/models/20230502_SUMO_MA/osm.motorcycle.trips.xml,urban_mobility_simulation/models/20230502_SUMO_MA/osm.truck.trips.xml",
    #route_file="urban_mobility_simulation/src/team_rl_test/data/cross.rou.xml",
    out_csv_name="sumo-rl/outputs/big-intersection/dqn_woPT",
    use_gui=True,
    num_seconds=5000,
    #num_seconds=5400,
    yellow_time=4,
    min_green=5,
    max_green=60,
)
model = PPO(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    verbose=1,
    
    
# model = DQN(
#     env=env,
#     policy="MlpPolicy",
#     learning_rate=1e-3,
#     learning_starts=0,
#     buffer_size=50000,
#     train_freq=1,
#     target_update_interval=500,
#     exploration_fraction=0.05,
#     exploration_final_eps=0.01,
#     verbose=1,
)

model.learn(total_timesteps=5000)
