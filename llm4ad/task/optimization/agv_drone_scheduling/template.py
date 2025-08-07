task_description = """
# AGV and Drone Hybrid Scheduling Problem

## Objective
Design an efficient algorithm to schedule AGVs and drones to process all parts while minimizing the total completion time. Score = negative value of the average completion time of successful instances (higher is better).

## Environment Setup
- 3 locations: A (loading area), B (processing area), C (buffer area)
- AGVs and drones are responsible for transporting parts and trays
- Parts arrive at A, need to be processed at B, and trays return to A
- All vehicles need to manage battery levels

## Key Parameters
- part_num = 50, tray_num = 10, AGV_num = 4, drone_num = 2
- B_location_num = 5, C_location_num = 5
- t_B_processing = 600 (processing time)
- Base travel times: t_AB_base = 311, t_AC_base = 201, t_BC_base = 141
- t_A_part_interval = 50 (part arrival interval)
- Drone speed factor: drone_speed_factor = 0.6 (drones are 1.67 times faster than AGVs)
- Traffic jam probability: traffic_jam_prob = 0.2
- AGV traffic jam impact factor: traffic_jam_factor_agv = 2.0
- Drone traffic jam impact factor: traffic_jam_factor_drone = 1.2
- Processing failure probability: processing_failure_prob = 0.1
- Battery capacity: battery_capacity = 100
- Battery consumption per distance unit: battery_consumption_per_distance = 0.05
- Critical battery level: battery_critical_level = 20
- Charging station is located at point A

## State Information
- A_part_info[i,0]: 1 means part exists, 0 means does not exist
- A_part_info[i,1]: Part arrival time
- A_tray[i]: 1 means tray is available, 0 means unavailable
- B_info[i,0]: Part ID (0 means none)
- B_info[i,1]: Tray ID (0 means none)
- B_info[i,2]: Processing end time
- B_info[i,3]: Processing status (0=unprocessed, 1=normal processing, 2=idle run)
- C_info[i,0]: Part ID (0 means none)
- C_info[i,1]: Tray ID (0 means none)
- C_info[i,2]: 1 means location is reserved, 0 means not reserved
- vehicle_info[i,0]: Position (1=A, 2=B, 3=C)
- vehicle_info[i,1]: Carried part ID (0 means none)
- vehicle_info[i,2]: Carried tray ID (0 means none)
- vehicle_info[i,3]: Time to complete current action
- vehicle_info[i,4]: Vehicle type (0=AGV, 1=drone)
- vehicle_battery[i]: Vehicle battery level (0-100)
- vehicle_timer[i]: Current time for each vehicle

## Available Actions
- 0: Take part and tray from A to B
- 1: Collect tray from B back to A
- 2: Take part and tray from A to C
- 3: Take part and tray from C to B
- 4: Go to A for charging
- Return -1 indicates no available action

## Validation Functions
- check_action0_valide(vehicle_index): A has parts, A has trays, B has space, battery sufficient
- check_action1_valide(vehicle_index): B has completed parts/empty trays, battery sufficient
- check_action2_valide(vehicle_index): A has parts, A has trays, C has unreserved space, battery sufficient
- check_action3_valide(vehicle_index): C has parts, B has space, battery sufficient
- check_action4_valide(vehicle_index): Check if charging is needed

## Helper Functions
- return_nearest_A_part_tray_index(): Get index of nearest part/tray at A
- return_nearest_available_pos(position): Get nearest available position at B/C
- return_near_finished_part_pos_in_B(): Get index of part with earliest end time in B
- check_battery_sufficient(vehicle_index, start, end): Check if battery is sufficient to complete task
- need_charging(vehicle_index): Check if charging is needed
- calculate_travel_time(vehicle_index, start, end): Calculate actual travel time considering traffic jams

## Key Considerations
- Prevent deadlocks: Ensure trays are recycled back to A
- Balance usage of buffer area C
- All parts must be processed to succeed
- Handle processing failures: Failed parts don't produce finished products, just recycle the tray
- Battery management: Need to charge on time to prevent battery depletion
- Reasonable allocation of AGVs and drones: Drones are faster and less affected by congestion, suitable for long distances and congested routes
- Completion time = maximum time of all vehicles
"""

template_program = '''
import numpy as np

def select_next_action(env, vehicle_index, current_node):
    """
    Design an innovative algorithm to select the next action for the given vehicle.
    
    Parameters:
        env: Vehicle environment containing the current state
        vehicle_index: Index of the vehicle to be scheduled
        current_node: Current position of the vehicle ('A', 'B', or 'C')
        
    Returns:
        Action index (0, 1, 2, 3, 4) or -1 indicating no available action
        
        Action 0: Take part and tray from A to B
        Action 1: Collect tray from B back to A
        Action 2: Take part and tray from A to C
        Action 3: Take part and tray from C to B
        Action 4: Go to A for charging
    """
    # First check if charging is needed
    if env.check_action4_valide(vehicle_index):
        return 4
        
    # Determine vehicle type
    is_drone = (env.vehicle_info[vehicle_index, 4] == 1)
    
    # Basic implementation
    if env.check_action0_valide(vehicle_index):
        return 0
    elif env.check_action1_valide(vehicle_index):
        return 1
    elif env.check_action2_valide(vehicle_index):
        return 2
    elif env.check_action3_valide(vehicle_index):
        return 3
    else:
        return -1
''' 