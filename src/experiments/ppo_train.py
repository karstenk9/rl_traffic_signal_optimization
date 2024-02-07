import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.logger import configure
import sys
import os

import ma_environment.custom_envs as custom_env


n_steps = 200000
if len(sys.argv) > 1:
    try:
        n_steps = int(sys.argv[1])
    except ValueError:
        pass

reward_fn = "diff-waiting-time"
name = reward_fn + str(n_steps)
output_path = f"../../outputs/{name}"
os.makedirs(output_path, exist_ok=True)


env = custom_env.MA_grid_train(use_gui=False,
                            reward_fn = reward_fn,
                            traffic_lights= ['tls_159','tls_160', 'tls_161'], 
                            sumo_warnings=False,
                            begin_time=25200,
                            num_seconds=4500, # sim_max_time = begin_time + num_seconds
                            out_csv_name='/home/inestp01/rl_traffic_signal_optimization/src/data/model_outputs/'+name,
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
    device='auto' # use 'auto' for cpu & mps for GPU
)

logger = configure("log", ["stdout", "csv"])
model.set_logger(logger)

print("Starting training")
model.learn(total_timesteps=n_steps)

model.save(output_path + "/model.zip")

print("Training finished. Starting evaluation")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

print('Mean Reward: ', mean_reward)

env.close()