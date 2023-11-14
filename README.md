MARL for Traffic Signal Optimisation using SUMO
==============================

**Multi Agent Reinforcement Learning with PPO for Traffic Signal Optimization using SUMO**

As part of my Master Thesis at the University of Mannheim, the project investigates intelligent traffic signal control using multi-agent reinforcement learning (MARL) to identify measures that can reduce pollutant and noise emissions within cities. 
Hereby, SUMO (Simulation of Urban Mobility) is used for a microscopic traffic simulation and analysis within the inner city of Mannheim.

How to run this repository
------------

## Prerequisites:

#### Install [SUMO](https://eclipse.dev/sumo/) latest version:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc 
```
Important: Set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```
#### Install (OpenAI) [Gymnasium](https://gymnasium.farama.org):

```bash
pip install gymnasium
```
#### Install [PettingZoo](https://pettingzoo.farama.org):

```bash
pip install pettingzoo
```

#### Install [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3):

```bash
pip install stable-baselines3[extra]
```


### Run Multi-Agent RL with Stable Baseline3 PPO

The folder [src/experiments](https://github.com/JenniferHahn/urban_mobility_simulation/tree/master/src/experiments) contains the relevant training [ppo_train.py](https://github.com/JenniferHahn/rl_traffic_signal_optimization/blob/master/src/experiments/ppo_train.py) as well as evaluation files ([eval_ppo.py](https://github.com/JenniferHahn/rl_traffic_signal_optimization/blob/master/src/experiments/eval_ppo.py) & [5episode_eval_ppo.py](https://github.com/JenniferHahn/rl_traffic_signal_optimization/blob/master/src/experiments/5episode_eval_ppo.py)). Another folder - [ma_environment](https://github.com/JenniferHahn/urban_mobility_simulation/tree/master/src/experiments/ma_environment) - specifies the network, environment, traffic signal actions and rewards, actions and observations for the multi-agent setting.

The network of the inner city of Mannheim used can be found in [models/20230717_sumo_ma](https://github.com/JenniferHahn/urban_mobility_simulation/tree/master/models/20230718_sumo_ma).

All models trained for different reward functions (defined in traffic_signal.py) are stored in [src/data/logs](https://github.com/JenniferHahn/rl_traffic_signal_optimization/tree/master/src/data/logs).

### Credits for SUMO-RL

The RL environments [ma_environment](https://github.com/JenniferHahn/urban_mobility_simulation/tree/master/src/experiments/ma_environment) and [environment](https://github.com/JenniferHahn/urban_mobility_simulation/tree/master/src/experiments/environment) are based on Lucas Alegre's [sumo-rl environment](https://github.com/LucasAlegre/sumo-rl/tree/main/sumo_rl/environment) which make use of Gymnasium and Stable Baselines3.


**Important Notes:**

- The obs_as_tensor function in stable_baselines3/common/util.py was updated to run the code with local GPU (mps).



Project Organization
------------



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
