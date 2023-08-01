"""Environments from RESCO: https://github.com/jault/RESCO, paper https://people.engr.tamu.edu/guni/Papers/NeurIPS-signals.pdf ."""
import os

import sumo_rl
from sumo_rl import env, parallel_env


PATH = os.path.dirname(sumo_rl.__file__)


def Mannheim_Network(parallel=True, **kwargs):
    """Grid 4x4 network.

    Number of agents = 3
    Number of actions = 4
    Agents have the same obsevation and action space
    """
    kwargs.update(
        {
            'net_file': 'urban_mobility_simulation/models/20230718_sumo_ma/osm.net_1.xml, \
                        urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_stops.add.xml, \
                        urban_mobility_simulation/models/20230718_sumo_ma/pt/stops.add.xml, \
                        urban_mobility_simulation/models/20230718_sumo_ma/osm.poly.xml, \
                        urban_mobility_simulation/models/20230718_sumo_ma/pt/vtypes.xml',
            'route_file': 'urban_mobility_simulation/models/20230718_sumo_ma/routes_nm.xml, \
                        urban_mobility_simulation/models/20230718_sumo_ma/bicycle_routes.xml,\
                        urban_mobility_simulation/models/20230718_sumo_ma/motorcycle_routes.xml,\
                        urban_mobility_simulation/models/20230718_sumo_ma/trucks_routes.xml, \
                        urban_mobility_simulation/models/20230718_sumo_ma/pt/gtfs_pt_vehicles.add.xml',
            'begin_time'
            'num_seconds': 3600,
        }
    )
    if parallel:
        return parallel_env(**kwargs)
    else:
        return env(**kwargs)
