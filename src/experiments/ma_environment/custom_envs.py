"""Environments from RESCO: https://github.com/jault/RESCO, paper https://people.engr.tamu.edu/guni/Papers/NeurIPS-signals.pdf ."""
import os
from ma_environment.env import env, parallel_env
from ma_environment.env_eval import parallel_env as parallel_env_eval


def MA_grid_train(parallel=True, **kwargs):
    """Mannheim Simulaton Network.

    Number of agents = 3
    Number of actions = variable (wrapped so that largest action space is taken)
    """
    kwargs.update(
        {
            "net_file": "/home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/osm.net.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/stops.add.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/vtypes.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/osm.poly.xml",
            "route_file": "/home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/veh_routes.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/truck_routes.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/bicycle_routes.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/motorcycle_routes.xml",
        "num_seconds": 4500,
        "begin_time": 25200,
        "time_to_teleport": 300,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)
    

def MA_grid_eval(parallel=True, **kwargs):
    """Mannheim Simulaton Network.

    Number of agents = 3
    Number of actions = variable (wrapped so that largest action space is taken)
    """
    kwargs.update(
        {
            "net_file": "/home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/osm.net.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/stops.add.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/vtypes.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/osm.poly.xml",
            "route_file": "/home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/veh_routes.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/truck_routes.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/bicycle_routes.xml, \
                        /home/inestp01/rl_traffic_signal_optimization/models/20230718_sumo_ma/motorcycle_routes.xml",
        "num_seconds": 9000,
        "begin_time": 25200,
        "time_to_teleport": 300,
        }
    )
    if parallel:
        return parallel_env_eval(**kwargs)
    else:
        return env(**kwargs)
    

def MA_grid_old(parallel=False, **kwargs):
    """Previous Mannheim Simulaton Network.

    Number of agents = 1
    Number of actions = variable (wrapped so that largest action space is taken)
    """
    kwargs.update(
        {
            "net_file": "urban_mobility_simulation/models/20230502_SUMO_MA/osm.net.xml, \
                        urban_mobility_simulation/models/20230502_SUMO_MA/pt/stops.add.xml, \
                        urban_mobility_simulation/models/20230502_SUMO_MA/osm.poly.xml",
            "single_agent":True,
            "route_file":"urban_mobility_simulation/models/20230502_SUMO_MA/routes.xml, \
                        urban_mobility_simulation/models/20230502_SUMO_MA/osm.bicycle.trips.xml,\
                        urban_mobility_simulation/models/20230502_SUMO_MA/osm.motorcycle.trips.xml,\
                        urban_mobility_simulation/models/20230502_SUMO_MA/osm.truck.trips.xml, \
                        urban_mobility_simulation/models/20230502_SUMO_MA/pt/ptflows.rou.xml, \
                        urban_mobility_simulation/models/20230502_SUMO_MA/osm.passenger.trips.xml",
                        #urban_mobility_simulation/models/20230502_SUMO_MA/osm.pedestrip.trips.xml",
                        #urban_mobility_simulation/models/20230502_SUMO_MA/osm.pedestrian.trips.xml", \
            "out_csv_name":"urban_mobility_simulation/src/data/model_outputs/ppo_withPT_10000",
        "num_seconds": 28800,
        "begin_time": 25200,
        "time_to_teleport": 300,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)