# Import libraries
import traci
import pandas as pd
import numpy as np

# IMPORTANT! # make sure to set all traffic lights logics to actuated in the sumo config (net) file!

# List of lanes to monitor (lanes controlled by TL agents later on)
# lane_ids = ['251161865#1_2', '189068695#4_0', '276412658_1', ':cluster_1743822458_1743822558_1743822643_1743822689_1743822737_8039877991_cluster_1120310798_1634545540_1665161322_1665161338_1665161344_1743822496_1743822510_1743822551_1743822648_1743822650_1743822666_1743822667_1743822676_1743822687_1754245066_1756301705_1949670169_2004844603_297701075_412123597_412123598_412123601_412181181_w2_0', \
#             '96747828#0_0', '276412658_4', ':cluster_cluster_1109568391_cluster_1109568409_1109568428_25422076_834022925_834023464_cluster_1364262300_cluster_1109568414_1364262303_1743822792_297701095_4586935111_w3_0', '96747828#0_2', '96747828#0_1', ':cluster_1743822458_1743822558_1743822643_1743822689_1743822737_8039877991_cluster_1120310798_1634545540_1665161322_1665161338_1665161344_1743822496_1743822510_1743822551_1743822648_1743822650_1743822666_1743822667_1743822676_1743822687_1754245066_1756301705_1949670169_2004844603_297701075_412123597_412123598_412123601_412181181_w0_0',\
#             ':cluster_cluster_1743822127_4889543898_cluster_1756301698_25113885_271075996_cluster_1743822142_cluster_1743822133_1743822191_1743822205_1756301686_1756301687_1756301692_1756301694_2003762999_248783350_2574373755_2574373756_2574373757_2574373758_266980758_26936934_419622507_4890135930_6838389013_6838389025_6840644265_705047176_w2_0', '251157594_1', \
#             ':cluster_1743822458_1743822558_1743822643_1743822689_1743822737_8039877991_cluster_1120310798_1634545540_1665161322_1665161338_1665161344_1743822496_1743822510_1743822551_1743822648_1743822650_1743822666_1743822667_1743822676_1743822687_1754245066_1756301705_1949670169_2004844603_297701075_412123597_412123598_412123601_412181181_w8_0', \
#             ':cluster_cluster_1743822127_4889543898_cluster_1756301698_25113885_271075996_cluster_1743822142_cluster_1743822133_1743822191_1743822205_1756301686_1756301687_1756301692_1756301694_2003762999_248783350_2574373755_2574373756_2574373757_2574373758_266980758_26936934_419622507_4890135930_6838389013_6838389025_6840644265_705047176_w1_0', \
#             ':cluster_cluster_1743822127_4889543898_cluster_1756301698_25113885_271075996_cluster_1743822142_cluster_1743822133_1743822191_1743822205_1756301686_1756301687_1756301692_1756301694_2003762999_248783350_2574373755_2574373756_2574373757_2574373758_266980758_26936934_419622507_4890135930_6838389013_6838389025_6840644265_705047176_w4_0', \
#             '189877634#0_3',':cluster_1743822458_1743822558_1743822643_1743822689_1743822737_8039877991_cluster_1120310798_1634545540_1665161322_1665161338_1665161344_1743822496_1743822510_1743822551_1743822648_1743822650_1743822666_1743822667_1743822676_1743822687_1754245066_1756301705_1949670169_2004844603_297701075_412123597_412123598_412123601_412181181_w6_0', 
#             ':cluster_1743822458_1743822558_1743822643_1743822689_1743822737_8039877991_cluster_1120310798_1634545540_1665161322_1665161338_1665161344_1743822496_1743822510_1743822551_1743822648_1743822650_1743822666_1743822667_1743822676_1743822687_1754245066_1756301705_1949670169_2004844603_297701075_412123597_412123598_412123601_412181181_w5_0', \
#             ':cluster_cluster_1743822127_4889543898_cluster_1756301698_25113885_271075996_cluster_1743822142_cluster_1743822133_1743822191_1743822205_1756301686_1756301687_1756301692_1756301694_2003762999_248783350_2574373755_2574373756_2574373757_2574373758_266980758_26936934_419622507_4890135930_6838389013_6838389025_6840644265_705047176_w5_0', '189877634#0_0', \
#             '189877634#0_2', '548514763#0_3', '251161865#1_1', '778989039#3_0', '96747818#3_0', '276412658_2', '276412658_3', ':cluster_cluster_1109568391_cluster_1109568409_1109568428_25422076_834022925_834023464_cluster_1364262300_cluster_1109568414_1364262303_1743822792_297701095_4586935111_w2_0', '548514763#0_1', ':cluster_1743822458_1743822558_1743822643_1743822689_1743822737_8039877991_cluster_1120310798_1634545540_1665161322_1665161338_1665161344_1743822496_1743822510_1743822551_1743822648_1743822650_1743822666_1743822667_1743822676_1743822687_1754245066_1756301705_1949670169_2004844603_297701075_412123597_412123598_412123601_412181181_w7_0', \
#             '548514763#0_2', ':cluster_cluster_1109568391_cluster_1109568409_1109568428_25422076_834022925_834023464_cluster_1364262300_cluster_1109568414_1364262303_1743822792_297701095_4586935111_w1_0', '251161865#1_3', '96851712#0_0', ':cluster_1743822458_1743822558_1743822643_1743822689_1743822737_8039877991_cluster_1120310798_1634545540_1665161322_1665161338_1665161344_1743822496_1743822510_1743822551_1743822648_1743822650_1743822666_1743822667_1743822676_1743822687_1754245066_1756301705_1949670169_2004844603_297701075_412123597_412123598_412123601_412181181_w3_0',\
#             '548514763#0_0', '189877848_0', '251157594_2', '251157594_4', '251157594_3', '189877634#0_1', '278441980#3_0', '189877848_1']


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
             local_PMx_emission, local_NOx_emission, local_noise_emission, 
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
           'localPMxEmission', 'localNOxEmission', 'localNoiseEmission',
           'localWaitingTime', 'localStoppedVehicles',
           'totalCO2Emission', 'totalCOEmission', 'totalHCEmission', 'totalPMxEmission', 'totalNOxEmission', 'totalFuelConsumption',
           'totalNoiseEmission', 'totalWaitingTime', 'totalStoppedVehicles',
           'tls159_phase', 'tls159_phase_duration', 'tls159_state',
           'tls160_phase', 'tls160_phase_duration', 'tls160_state',
           'tls161_phase', 'tls161_phase_duration', 'tls161_state']

df = pd.DataFrame(data, columns=columns)

# Write the DataFrame to a csv file
df.to_csv('urban_mobility_simulation/src/data/actuated_output/actuated_output_9000steps_moreInfo.csv', index=False)