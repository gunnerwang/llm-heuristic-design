task_description = """
# AGV Scheduling Problem

## Objective
Design an algorithm to efficiently schedule AGVs in a factory setting. Process ALL parts while minimizing total completion time. Score = negative average completion time across successful instances (higher is better).

## Environment Setup
- 3 Locations: A (loading), B (processing), C (buffer)
- AGVs transport parts with trays between locations
- Parts arrive at A, need processing at B, and trays return to A

## Key Constants
- part_num = 50, tray_num = 10, AGV_num = 6
- B_location_num = 5, C_location_num = 5
- t_B_processing = 600 (processing time)
- Travel times: t_AB = 311, t_AC = 201, t_BC = 141
- t_A_part_interval = 50 (part arrival interval)

## State Information
- A_part_info[i,0]: 1 if part present, 0 if not
- A_part_info[i,1]: Part arrival time
- A_tray[i]: 1 if tray available, 0 if not
- B_info[i,0]: Part ID (0 if none)
- B_info[i,1]: Tray ID (0 if none)
- B_info[i,2]: Processing end time
- C_info[i,0]: Part ID (0 if none)
- C_info[i,1]: Tray ID (0 if none)
- C_info[i,2]: 1 if position reserved, 0 if not
- AGV_info[i,0]: Location (1=A, 2=B, 3=C)
- AGV_info[i,1]: Part ID carried (0 if none)
- AGV_info[i,2]: Tray ID carried (0 if none)
- AGV_info[i,3]: Time to complete current action
- AGV_timer[i]: Current time for each AGV

## Available Actions
- 0: Take part+tray from A to B
- 1: Recycle tray from B to A
- 2: Take part+tray from A to C
- 3: Carry part+tray from C to B
- Return -1 if no valid action available

## Validation Functions
- check_action0_valide(): True if A has parts, A has trays, B has space
- check_action1_valide(): True if B has finished parts/empty trays
- check_action2_valide(): True if A has parts, A has trays, C has unreserved space
- check_action3_valide(): True if C has parts, B has space

## Helper Functions
- return_nearest_A_part_tray_index(): Get indices of nearest part/tray at A
- return_nearest_available_pos(position): Get nearest available position at B/C
- return_near_finished_part_pos_in_B(): Get index of part in B with earliest finish time

## Critical Considerations
- Prevent deadlocks: ensure trays cycle back to A
- Balance buffer usage at location C
- All parts must be processed for success
- Completion time = maximum time across all AGVs
"""

template_program = '''
import numpy as np

def select_next_action(env, agv_index, current_node):
    """
    Design a novel algorithm to select the next action for the given AGV.
    
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
    # Basic placeholder implementation
    if env.check_action0_valide():
        return 0
    elif env.check_action1_valide():
        return 1
    elif env.check_action2_valide():
        return 2
    elif env.check_action3_valide():
        return 3
    else:
        return -1
'''
