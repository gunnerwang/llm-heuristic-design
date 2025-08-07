task_description = """
# AGV Scheduling Problem

In a factory automation setting, multiple AGVs (Automated Guided Vehicles) need to transport materials between different locations:
- Location A: Material loading area with parts and trays
- Location B: Processing area where parts are processed
- Location C: Buffer/storage area

The goal is to minimize the total completion time for processing all parts.

## The Environment

- Multiple AGVs can operate simultaneously
- Parts arrive at location A with trays
- Parts need to be processed at location B
- Location C can be used as a buffer
- AGVs can pick up parts and trays from A, and transport them to B or C
- AGVs can move parts from C to B
- After processing at B, trays need to be returned to A

## Environment State and Functions
The environment (env) provides several attributes and functions:

### Important State Attributes:
- A_part_info: 2D array (part_max_num × 2) for parts at location A
  - First column (A_part_info[:,0]): 1 if part is present, 0 if not
  - Second column (A_part_info[:,1]): Arrival time of the part
  
- A_tray: 1D array (tray_max_num) for trays at location A
  - Value 1: Tray is available
  - Value 0: Tray is not available (in use)
  
- B_info: 2D array (B_location_num × 3) for parts and trays at location B
  - First column (B_info[:,0]): Part ID (0 if no part)
  - Second column (B_info[:,1]): Tray ID (0 if no tray)
  - Third column (B_info[:,2]): Processing end time (when part will be finished)
  
- C_info: 2D array (C_location_num × 3) for parts and trays at location C
  - First column (C_info[:,0]): Part ID (0 if no part)
  - Second column (C_info[:,1]): Tray ID (0 if no tray)
  - Third column (C_info[:,2]): Reserved flag (1 if position is reserved)
  
- AGV_info: 2D array (AGV_max_num × 4) for AGV state information
  - First column (AGV_info[:,0]): Current location ID (1 for A, 2 for B, 3 for C)
  - Second column (AGV_info[:,1]): Part ID being carried (0 if none)
  - Third column (AGV_info[:,2]): Tray ID being carried (0 if none)
  - Fourth column (AGV_info[:,3]): Time until AGV completes current action
  
- AGV_timer: 1D array (AGV_num) tracking current time for each AGV

### Available Functions:
- check_action0_valide(): Check if taking part from A to B is valid
  - Returns True if: A has parts, A has trays, and B has space
  
- check_action1_valide(): Check if recycling tray from B to A is valid
  - Returns True if: B has a part that finished processing or an empty tray
  
- check_action2_valide(): Check if taking part from A to C is valid
  - Returns True if: A has parts, A has trays, and C has unreserved space
  
- check_action3_valide(): Check if carrying part from C to B is valid
  - Returns True if: C has parts and B has space

- return_nearest_A_part_tray_index(): Returns the indices of the nearest part and tray at A
- return_nearest_available_pos(position): Returns the index of the nearest available position at B or C
- return_near_finished_part_pos_in_B(): Returns the index of a part in B that has finished processing

## Available Actions

- Action 0: Take part and tray from A to B
- Action 1: Recycle tray from B to A
- Action 2: Take part and tray from A to C
- Action 3: Carry part and tray from C to B

## Your Task

Design an algorithm to efficiently schedule AGVs by selecting the best action for each AGV at any given time.
"""

template_program = '''
import numpy as np

def select_next_action(env, agv_index, current_node):
    """
    Select the next action for the given AGV.
    
    Args:
        env: The AGV environment containing the current state
        agv_index: Index of the AGV to schedule
        current_node: Current location of the AGV ('A', 'B', or 'C')
        
    Returns:
        Action index (0, 1, 2, or 3) or -1 if no valid action is available
        
        Action 0: Take part and tray from A to B
        Action 1: Recycle tray from B to A
        Action 2: Take part and tray from A to C
        Action 3: Carry part and tray from C to B
    """
    reward_list = []
    if env.check_action0_valide():
        if current_node == 'A':
            extra_t = 0
        elif current_node == 'B':
            extra_t = env.t_AB
        elif current_node == 'C':
            extra_t = env.t_AC
        A_part_index, A_tray_index = env.return_nearest_A_part_tray_index()
        reward = env.distance_fac*extra_t + env.time_fac*max(env.A_part_info[A_part_index-1,1] - extra_t - env.AGV_timer[agv_index], 0)
        reward_list.append(reward)
    else:
        reward_list.append(10e5)
    if env.check_action1_valide():
        if current_node == 'A':
            extra_t = env.t_AB
        elif current_node == 'B':
            extra_t = 0
        elif current_node == 'C':
            extra_t = env.t_BC
        tray_index = env.return_near_finished_part_pos_in_B()
        reward = env.distance_fac*extra_t + env.time_fac*max(0, env.B_info[tray_index, 2]-extra_t-env.AGV_timer[agv_index])
        reward_list.append(reward)
    else:
        reward_list.append(10e5)
    if env.check_action2_valide():
        if current_node == 'A':
            extra_t = 0
        elif current_node == 'B':
            extra_t = env.t_AB
        elif current_node == 'C':
            extra_t = env.t_AC
        A_part_index, A_tray_index = env.return_nearest_A_part_tray_index()
        reward = env.distance_fac*extra_t + env.time_fac*max(env.A_part_info[A_part_index-1,1] - extra_t - env.AGV_timer[agv_index], 0)
        reward_list.append(reward)
    else:
        reward_list.append(10e5)
    if env.check_action3_valide():
        if current_node == 'C':
            extra_t = 0
        elif current_node == 'B':
            extra_t = env.t_BC
        else:
            extra_t = env.t_AC
        reward = extra_t*env.distance_fac
        reward_list.append(reward)
    else:
        reward_list.append(10e5)
    if min(reward_list) == 10e5:
        return -1
    else:
        return np.argmin(reward_list)
'''
