import xml.etree.ElementTree as ET
import traci


def change_sumo_config_status(file_path, tls_ids, status):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for element in root.iter('tlLogic'):
        if element.get('id') in tls_ids:
            element.set("type", status)

    tree.write(file_path)


class VehicleTimesListener(traci.StepListener):
    """
    This listener can be added to TRACI to read out the vehicles departure and arrival times of vehicles.
    The step function is called every time when traci.simulationStep is called.
    """
    def __init__(self, controlled_lanes):
        self.controlled_lanes = controlled_lanes
        self.teleported_vehicles = set()
        self.vehicle_departures = dict()
        self.vehicle_arrivals = dict()
        self.controlled_vehicles = set()

    def step(self, t):
        sim_time = traci.simulation.getTime()
        for vehicle in traci.simulation.getDepartedIDList():
            self.vehicle_departures[vehicle] = sim_time
        for vehicle in traci.simulation.getArrivedIDList():
            self.vehicle_arrivals[vehicle] = sim_time
        for lane in self.controlled_lanes:
            self.controlled_vehicles.update(traci.lane.getLastStepVehicleIDs(lane))
        self.teleported_vehicles.update(traci.simulation.getStartingTeleportIDList())

        # indicate that the step listener should stay active in the next step
        return True