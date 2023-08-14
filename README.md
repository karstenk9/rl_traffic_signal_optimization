Urban Mobility Simulation 
==============================

**Multi Agent RL (PPO) for Traffic Signal Optimization using SUMO*
The project investigates traffic polciies, and more specifically intelligent traffic signal control, based on reinforcement learning to identify measures that can reduce pollutant emissions within cities. SUMO (Simulation of Urban Mobility) is used for a microscopic traffic simulation and analysis within the inner city of Mannheim.

How to run this repository
------------

## Install

### Install SUMO latest version:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc 
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```
Important: for a huge performance boost (~8x) with Libsumo, you can declare the variable:
```bash
export LIBSUMO_AS_TRACI=1
```
Notice that you will not be able to run with sumo-gui or with multiple simulations in parallel if this is active ([more details](https://sumo.dlr.de/docs/Libsumo.html)).

### Install SUMO-RL

Stable release version is available through pip
```bash
pip install sumo-rl
```

Alternatively you can install using the latest (unreleased) version
```bash
git clone https://github.com/LucasAlegre/sumo-rl
cd sumo-rl
pip install -e .
```

### Run Multi-Agent RL with Stable Baseline3 PPO
The folder [src/experiments](https://github.com/JenniferHahn/urban_mobility_simulation/tree/master/src/experiments) contains the relevant training file and another folder [ma_environment](https://github.com/JenniferHahn/urban_mobility_simulation/tree/master/src/experiments/ma_environment) that specifies the network, environment, traffic signal actions and rewards, actions and observations for the multi-agent setting.

The network of the inner city of Mannheim used can be found in [models/20230717_sumo_ma](https://github.com/JenniferHahn/urban_mobility_simulation/tree/master/models/20230718_sumo_ma).

The Environment including agent, action and observation spaces as well as inital reward function and metrics for a baseline are based on [SUMO-RL](https://github.com/LucasAlegre/sumo-rl/). 


**Important Notes:**

- The obs_as_tensor function in stable_baselines3/common/util.py was updated to run the code with local GPU (mps).
- In case there are any issues running ray, try pip install -U pyarrow and install numpy version 1.19
- In case there are any issues with the reset function within pettingzoo_env, exchange the original ray/rllib/env/wrappers/pettingzoo_env.py with the following script/commit
https://github.com/ray-project/ray/blob/f24f94ca36a79366e935d842d657a25470216faa/rllib/env/wrappers/pettingzoo_env.py




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
