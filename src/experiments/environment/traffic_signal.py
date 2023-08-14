"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
from typing import Callable, List, Union

# if "SUMO_HOME" in os.environ:
#     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
#     sys.path.append(tools)
# else:
#     raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import platform

if platform.system() != "Linux":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)  # we need to import python modules from the $SUMO_HOME/tools directory
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    import traci
else:
    import libsumo as traci
import numpy as np
#from gymnasium import spaces
from gym import spaces


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        begin_time: int,
        reward_fn: Union[str, Callable],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.sumo = sumo

        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficSignal.reward_fns.keys():
                self.reward_fn = TrafficSignal.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_lenght = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        # print('Logic Phases', logic.phases)
        # print('Len Log Phases', len(logic.phases))
        # print('current Index', logic.currentPhaseIndex)
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase) ## dropped comment here 
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase) ## dropped comment here 
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    ### REWARDS ###

    def compute_reward(self):
        """Computes the reward of the traffic signal."""
        self.last_reward = self.reward_fn(self)
        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward
    
    def _average_emission_reward(self):
        return -self.get_average_emission.total_emission_avg()
    
    def _CO2_emission_reward(self):
        return - self.get_total_CO2emission()
    
    def _ts_emission_reward(self):
        # emission for all lanes controlled by all choen traffic signals
        # get respective list element for type of emission 
        # [CO2_emission,CO_emission, HC_emission,Mx_emission,NOx_emission, emission_combined,fuel_consumption]
        return -self.get_emission_for_controlled_lanes()[0]
    
    def _noise_emission_reward(self):
        return -self.get_average_noise_emission
    
    def _brake_reward(self):
        return self.get_total_braking()

    def _acceleration_reward(self):
        return self.get_total_acceleration()
    
    ### COMPUTE REWARD COMPONENTS ###
    
    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_total_CO2emission(self) -> float:
        '''
        Return the total CO2 pollutant emission of all vehicles in a simulation.
        '''
        CO2emission = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 0.0
        CO2emission = sum(self.sumo.vehicle.getCO2Emission(veh) for veh in vehs)
        return CO2emission
    
    def get_average_emission(self) -> float:
        '''
        Return the average pollutant emission of alle vehicles in a simulation separated into CO, CO2, HC, NOx, PMx, and fuel consumption.
        '''
        CO_emission = 0.0
        CO2_emission = 0.0
        HC_emission = 0.0
        PMx_emission = 0.0
        NOx_emission = 0.0
        fuel_consumption = 0.0
        
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 0.0
        
        CO_emission = sum(self.sumo.vehicle.getCOEmission(veh) for veh in vehs)
        CO2_emission = sum(self.sumo.vehicle.getCO2Emission(veh) for veh in vehs)
        HC_emission = sum(self.sumo.vehicle.getHCEmission(veh) for veh in vehs)
        PMx_emission = sum(self.sumo.vehicle.getPMxEmission(veh) for veh in vehs)
        NOx_emission = sum(self.sumo.vehicle.getNOxEmission(veh) for veh in vehs)
        fuel_consumption = sum(self.sumo.vehicle.getFuelConsumption(veh) for veh in vehs)
        
        total_emission_avg = (CO_emission + CO2_emission + HC_emission + PMx_emission + NOx_emission) / len(vehs)
        CO_avg = CO_emission / len(vehs)
        CO2_avg = CO2_emission / len(vehs)
        HC_avg = HC_emission / len(vehs)
        PMx_avg = PMx_emission / len(vehs)
        NOx_avg = NOx_emission / len(vehs)
        fuel_avg = fuel_consumption / len(vehs)
        
        return total_emission_avg, CO_avg, CO2_avg, HC_avg, PMx_avg, NOx_avg, fuel_avg
    

    def get_emission_per_lane(self) -> List[List[float]]:
        '''
        Function to get average emissions for all lanes, storing different emission values in a list element for each lane.
        
        Returns:
            List[List[float]]: List of lists containing the average emissions per lane.
            [CO2_emission,CO_emission, HC_emission,Mx_emission,NOx_emission, emission_combined,fuel_consumption]
        '''
        
        emission_per_lane = []
        
        #for every lane in the simulation
        for lane in self.lanes:
            #get all vehicles in the lane
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            CO_emission = 0.0
            CO2_emission = 0.0
            HC_emission = 0.0
            PMx_emission = 0.0
            NOx_emission = 0.0
            fuel_consumption = 0.0
            #for every vehicle in the lane
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                # get CO2 emission
                CO2 = self.sumo.vehicle.getCO2Emission(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: CO2}
                else:
                    self.env.vehicles[veh][veh_lane] = CO2 - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                CO2_emission += self.env.vehicles[veh][veh_lane]
                # get CO emission
                CO = self.sumo.vehicle.getCOEmission(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: CO}
                else:
                    self.env.vehicles[veh][veh_lane] = CO - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                CO_emission += self.env.vehicles[veh][veh_lane]
                # get HC emission
                HC = self.sumo.vehicle.getHCEmission(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: HC}
                else:
                    self.env.vehicles[veh][veh_lane] = HC - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                HC_emission += self.env.vehicles[veh][veh_lane]
                # get PMx emission
                PMx = self.sumo.vehicle.getPMxEmission(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: PMx}
                else:
                    self.env.vehicles[veh][veh_lane] = PMx - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                PMx_emission += self.env.vehicles[veh][veh_lane]
                # get NOx emission
                NOx = self.sumo.vehicle.getNOxEmission(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: NOx}
                else:
                    self.env.vehicles[veh][veh_lane] = NOx - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                NOx_emission += self.env.vehicles[veh][veh_lane]
                
                # get fuel consumption
                fuel = self.sumo.vehicle.getFuelConsumption(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: fuel}
                else:
                    self.env.vehicles[veh][veh_lane] = fuel - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                fuel_consumption += self.env.vehicles[veh][veh_lane]
            
                emission_combined = (CO_emission + CO2_emission + HC_emission + PMx_emission + NOx_emission) / 5
                
                
            emission_per_lane.append([CO2_emission,
                                    CO_emission,
                                    HC_emission,
                                    PMx_emission,
                                    NOx_emission,
                                    emission_combined,
                                    fuel_consumption
                                    ])
            
        return emission_per_lane
    
    
    def get_emission_for_controlled_lanes(self) -> List[float]:
        '''
        Function to get average emissions for all relevant lanes, storing different emission values in a list element for each lane.
        Relevant lanes = Lanes controlled by the chosen traffic signal
        
        Returns:
            List[float]: List containing the total emissions per lane.
            [CO2_emission,CO_emission, HC_emission,Px_emission,NOx_emission, emission_combined,fuel_consumption]
        '''
    
        emission_on_lanes = []
        lanes = []
            
        #ger all lanes that are controlled by all traffic lights in self.ts_ids
        for ts in self.ts_ids:
            lanes.append(self.sumo.trafficlight.getControlledLanes(ts))
            #get all vehicles in the lane
            veh_list = []
            for lane in lanes:
                veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
            CO_emission = 0.0
            CO2_emission = 0.0
            HC_emission = 0.0
            PMx_emission = 0.0
            NOx_emission = 0.0
            fuel_consumption = 0.0
            #for every vehicle located on the lanes
            for veh in veh_list:
                # get CO2 emission
                CO2 = self.sumo.vehicle.getCO2Emission(veh)
                CO2_emission += CO2
                # get CO emission
                CO = self.sumo.vehicle.getCOEmission(veh)
                CO_emission += CO
                # get HC emission
                HC = self.sumo.vehicle.getHCEmission(veh)
                HC_emission += HC
                # get PMx emission
                PMx = self.sumo.vehicle.getPMxEmission(veh)
                PMx_emission += PMx
                # get NOx emission
                NOx = self.sumo.vehicle.getNOxEmission(veh)
                NOx_emission += NOx
                # get fuel consumption
                fuel = self.sumo.vehicle.getFuelConsumption(veh)
                fuel_consumption += fuel
            
                emission_combined = (CO_emission + CO2_emission + HC_emission + PMx_emission + NOx_emission) / 5
                
            emission_on_lanes.append(CO2_emission,
                                    CO_emission,
                                    HC_emission,
                                    PMx_emission,
                                    NOx_emission,
                                    emission_combined,
                                    fuel_consumption
                                    )
            
        return emission_on_lanes
    
    def get_ts_emissions(self, ts_id):
        '''
        Function to get emissions for all lanes that the traffic light controls.
        
        Returns:
            Dict: Containing total emissions for all controlled lanes (per emission type).
            {CO2_emission,CO_emission, HC_emission,Mx_emission,NOx_emission, emission_combined,fuel_consumption}
        '''
        
        ts_lane_emissions = []
        
        #get all lanes that are controlled by the traffic light
        lanes = self.sumo.trafficlight.getControlledLanes(ts_id)
        
        #get all vehicles on the lanes
        veh_list = []
        for lane in lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
            
        for veh in veh_list:
            #veh_lane = self.sumo.vehicle.getLaneID(veh)
            # get CO2 emission
            CO2 = self.sumo.vehicle.getCO2Emission(veh)
            CO2_emission += CO2
            # get CO emission
            CO = self.sumo.vehicle.getCOEmission(veh)
            CO_emission += CO
            # get HC emission
            HC = self.sumo.vehicle.getHCEmission(veh)
            HC_emission += HC
            # get PMx emission
            PMx = self.sumo.vehicle.getPMxEmission(veh)
            PMx_emission += PMx
            # get NOx emission
            NOx = self.sumo.vehicle.getNOxEmission(veh)
            NOx_emission += NOx
            # get fuel consumption
            fuel = self.sumo.vehicle.getFuelConsumption(veh)
            fuel_consumption += fuel
            
            emission_combined = (CO_emission + CO2_emission + HC_emission + PMx_emission + NOx_emission) / 5
            
        ts_lane_emissions = {'CO2_emission': CO2_emission,
                             'CO_emission': CO_emission,
                             'HC_emission': HC_emission,
                             'PMx_emission': PMx_emission,
                             'NOx_emission': NOx_emission,
                             'emission_combined': emission_combined,
                             'fuel_consumption': fuel_consumption}
        
        return ts_lane_emissions
            
    
    # def get_emission_per_eclass(self):
    #     '''
    #     Get current emission per vehicle class.
        
    #     Returns:
    #         Dict(str: float): Dictionary containing the current emission per emission class.
    #     '''
    #     emission_per_type = {}
    #     emission_classes = self._get_vehicle_eclasses()
    #     for emission_class in emission_classes:
    #         emission_per_type[emission_class] = 0.0
    #     if emission_classes is None:
    #         emission_per_type = {'None' : 0.0}
    #         return emission_per_type
    #     vehs = self._get_veh_list()
    #     if len(vehs) == 0:
    #         return 0.0
    #     for veh in vehs:
    #         eclass = self._get_vehicle_type(veh)
    #         emission_per_type[eclass] += self.sumo.vehicle.getCO2Emission(veh)
        
    #     for emission_class in emission_classes:
    #         emission_per_type[emission_class] /= len(vehs)
        
    #     return emission_per_type

    
    def get_total_noise_emission(self) -> float:
        '''
        Return the total noise emission of all vehicles in a simulation.
        '''
        noise_emission = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 0.0
        noise_emission = sum(self.sumo.vehicle.getNoiseEmission(veh) for veh in vehs)
            
        return noise_emission
    
    def get_total_braking(self) -> float:
        '''
        Returns the total braking of all vehicles in a simulation.
        '''
        vehs = self._get_veh_list()
        accelerations = np.array([self.sumo.vehicle.getAcceleration(veh) for veh in vehs])
        brake = np.sum(accelerations[accelerations < 0])
        return brake


    def get_total_acceleration(self) -> float:
        '''
        Returns the total acceleration of all vehicles in a simulation.
        '''
        vehs = self._get_veh_list()
        accelerations = np.array([self.sumo.vehicle.getAcceleration(veh) for veh in vehs])
        accel = np.sum(accelerations[accelerations > 0])
        return accel
    

    def get_average_noise_emission(self) -> float:
        '''
        Returns the average noise emission of all vehicles in a simulation.
        '''
        noise_emission = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 0.0
        for veh in vehs:
            noise_emission += self.sumo.vehicle.getNoiseEmission(veh)
            
        return noise_emission / len(vehs)

    def get_noise_emission_lane(self) -> List[float]:
        '''
        Calculates the average noise emission for all lanes.
        
        Returns:
            List[float]: List of average noise emission of each intersection lane.
        '''
        noise_emission_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            noise_emission = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                noise = self.sumo.vehicle.getNoiseEmission(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: noise}
                else:
                    self.env.vehicles[veh][veh_lane] = noise - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                noise_emission += self.env.vehicles[veh][veh_lane]
            noise_emission_per_lane.append(noise_emission)
        return noise_emission_per_lane
    
    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        )

    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list
    
    def _get_vehicle_eclasses(self) -> List[str]:
        '''
        Returns a list of emission classes present in the current simulation.
        Returns:
            str: Vehicle type of the vehicle.
        '''
        veh_eclasses = []
        for veh in self._get_veh_list():
            eclass = self.sumo.vehicle.getEmissionClass(veh)
            if eclass not in veh_eclasses:
                veh_eclasses.append(eclass)
        
        return veh_eclasses


    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
        "CO2_emission": _CO2_emission_reward,
        "combined_emission": _average_emission_reward,
        "noise_emission": _noise_emission_reward,
        "local_emission": _ts_emission_reward,
    }

