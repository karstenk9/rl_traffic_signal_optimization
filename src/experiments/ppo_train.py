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

import ma_environment.custom_envs as custom_env


env = custom_env.MA_grid_train(use_gui=False,
                            reward_fn = 'diff-waiting-time',
                            traffic_lights= ['tls_159','tls_160', 'tls_161'], 
                            sumo_warnings=False,
                            begin_time=25200,
                            num_seconds=4500, # sim_max_time = begin_time + num_seconds
                            out_csv_name='/Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/model_outputs/diff-waiting-time_200000',
                            additional_sumo_cmd="--emission-output /Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/model_outputs/emission_diff-waiting-time.xml, \
                                                --lanedata-output /Users/jenniferhahn/Documents/GitHub/urban_mobility_simulation/src/data/model_outputs/lane_diff-waiting-time.xml",
                            )
max_time = env.unwrapped.env.sim_max_time
delta_time = env.unwrapped.env.delta_time

#wrap observation space to have one common observation space for all agents
env = ss.pad_observations_v0(env)

#wrap action space to have one common action space for all agents (based on largest action space)
env = ss.pad_action_space_v0(env)

#wrap pettingzoo env
env = ss.pettingzoo_env_to_vec_env_v1(env)

#concatenate envs
env = ss.concat_vec_envs_v1(env, 1, num_cpus=4, base_class="stable_baselines3")

env = VecMonitor(env)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=3,
    gamma=0.95,
    n_steps=256,
    ent_coef=0.01,
    learning_rate=0.00025,
    vf_coef=0.05,
    max_grad_norm=0.9,
    gae_lambda=0.95,
    n_epochs=10,
    clip_range=0.3,
    batch_size=64,
    tensorboard_log="./logs/MA_grid/waitingTime",
    device='auto' # use 'auto' for cpu & mps for GPU
)


print("Starting training")
model.learn(total_timesteps=200000)

model.save('urban_mobility_simulation/src/data/logs/waitingTime_200')

print("Training finished. Starting evaluation")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

print('Mean Reward: ', mean_reward)

env.close()