# Import libraries
import traci
import pandas as pd
import numpy as np

# IMPORTANT! # make sure to set all traffic lights logics to actuated in the sumo config (net) file!


# Connect to SUMO
traci.start(['sumo', "-c", "urban_mobility_simulation/models/20230718_sumo_ma/osm.sumocfg",
                                    "--time-to-teleport", "300",
                                    "--tripinfo-output", "urban_mobility_simulation/src/data/actuated_output/tripinfo_actuatedTL_9000.xml",
                                    "--output-prefix", "TIME",
                                    #"--device.emissions.probability", "1.0",
                                    "--emission-output", "urban_mobility_simulation/src/data/actuated_output/emission_actuatedTL_9000.xml"
                                    ])

# Initialize a list to hold data
data = []

# Traffic lights to monitor / get controlled lanes from
tls = ['tls_159','tls_160', 'tls_161']
             
# Loop through the simulation steps
for step in range(25200, 34200):  # == 2,5h simulation time
    traci.simulationStep() 
    
    all_vehicles = traci.vehicle.getIDList()
    
    # Get lanes and vehicles on lanes controlled by traffic lights
    controlled_lanes = list(set(item for sublist in (traci.trafficlight.getControlledLanes(ts) for ts in tls) for item in sublist))
    local_vehicle_ids = [item for sublist in (traci.lane.getLastStepVehicleIDs(lane_id) for lane_id in controlled_lanes) for item in sublist]

    # Get # of vehicles on the lanes
    num_vehicles = len(local_vehicle_ids) 
   
    # Get vehicle types
    vehicle_types = [] if num_vehicles == 0 else [traci.vehicle.getTypeID(vehicle_id) for vehicle_id in local_vehicle_ids] if num_vehicles > 0 else []

    # Get average speed of vehicles on the lanes
    avg_speed = 0.0 if num_vehicles == 0 else np.mean([traci.vehicle.getSpeed(vehicle_id) for vehicle_id in local_vehicle_ids])

    # Get local emission and further measures
    local_CO2_emission = sum(traci.lane.getCO2Emission(lane_id) for lane_id in controlled_lanes)
    local_CO_emission = sum(traci.lane.getCOEmission(lane_id) for lane_id in controlled_lanes)
    local_HC_emission = sum(traci.lane.getHCEmission(lane_id) for lane_id in controlled_lanes)
    local_PMx_emission = sum(traci.lane.getPMxEmission(lane_id) for lane_id in controlled_lanes)
    local_NOx_emission = sum(traci.lane.getNOxEmission(lane_id) for lane_id in controlled_lanes)
    local_fuel_consumption = sum(traci.lane.getFuelConsumption(lane_id) for lane_id in controlled_lanes)
    local_noise_emission = sum(traci.lane.getNoiseEmission(lane_id) for lane_id in controlled_lanes)
    local_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in controlled_lanes)
    local_stopped_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in controlled_lanes)
    
    # Get system total info
    total_CO2_emission = sum([traci.vehicle.getCO2Emission(vehicle) for vehicle in all_vehicles])
    total_CO_emission = sum([traci.vehicle.getCOEmission(vehicle) for vehicle in all_vehicles])
    total_HC_emission = sum([traci.vehicle.getHCEmission(vehicle) for vehicle in all_vehicles])
    total_PMx_emission = sum([traci.vehicle.getPMxEmission(vehicle) for vehicle in all_vehicles])
    total_NOx_emission = sum([traci.vehicle.getNOxEmission(vehicle) for vehicle in all_vehicles])
    total_fuel_consumption = sum([traci.vehicle.getFuelConsumption(vehicle) for vehicle in all_vehicles])
    total_noise_emission = sum([traci.vehicle.getNoiseEmission(vehicle) for vehicle in all_vehicles])
    total_waiting_time =  sum([traci.vehicle.getWaitingTime(vehicle) for vehicle in all_vehicles])
    total_stopped_vehicles = sum(int(speed < 0.1) for speed in [traci.vehicle.getSpeed(vehicle) for vehicle in all_vehicles])
    
    # Get TLS Info (to compare phases and transitions)
    tls159_phase = traci.trafficlight.getPhase(tls[0])
    tls159_phase_duration = traci.trafficlight.getPhaseDuration(tls[0])
    tls159_state = traci.trafficlight.getRedYellowGreenState(tls[0])
    tls160_phase = traci.trafficlight.getPhase(tls[1])
    tls160_phase_duration = traci.trafficlight.getPhaseDuration(tls[1])
    tls160_state = traci.trafficlight.getRedYellowGreenState(tls[1])
    tls161_phase = traci.trafficlight.getPhase(tls[2])
    tls161_phase_duration = traci.trafficlight.getPhaseDuration(tls[2])
    tls161_state = traci.trafficlight.getRedYellowGreenState(tls[2])
    

    # Append data to list for dataframe
    data.append([step, num_vehicles, vehicle_types, avg_speed, local_CO2_emission, local_CO_emission, local_HC_emission,
             local_PMx_emission, local_NOx_emission, local_fuel_consumption, local_noise_emission, 
             local_waiting_time, local_stopped_vehicles, 
             total_CO2_emission, total_CO_emission, total_HC_emission, total_PMx_emission, total_NOx_emission, total_fuel_consumption,
             total_noise_emission, total_waiting_time, total_stopped_vehicles,
             tls159_phase, tls159_phase_duration, tls159_state,
             tls160_phase, tls160_phase_duration, tls160_state,
             tls161_phase, tls161_phase_duration, tls161_state])

# Close the TraCI connection
traci.close()

# Create a DataFrame from the data
columns = ['Step', 'num_vehicles', 'vehicle_types', 'avg_speed', 'localCO2Emission', 'localCOEmission', 'localHCEmission',
           'localPMxEmission', 'localNOxEmission', 'local_fuel_consumption','localNoiseEmission',
           'localWaitingTime', 'localStoppedVehicles',
           'totalCO2Emission', 'totalCOEmission', 'totalHCEmission', 'totalPMxEmission', 'totalNOxEmission', 'totalFuelConsumption',
           'totalNoiseEmission', 'totalWaitingTime', 'totalStoppedVehicles',
           'tls159_phase', 'tls159_phase_duration', 'tls159_state',
           'tls160_phase', 'tls160_phase_duration', 'tls160_state',
           'tls161_phase', 'tls161_phase_duration', 'tls161_state']

df = pd.DataFrame(data, columns=columns)

# Write the DataFrame to a csv file
df.to_csv('urban_mobility_simulation/src/data/actuated_output/actuated_output_9000steps_moreInfo.csv', index=False)