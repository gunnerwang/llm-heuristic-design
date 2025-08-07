import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from multi_vehicle_env import multiVehicleEnv
import argparse
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Multi-Vehicle Testing with different selection methods')
parser.add_argument('--selection', type=int, default=1, choices=[0, 1, 2],
                    help='Selection method: 0=DQN, 1=Greedy, 2=Shortest Distance')
args = parser.parse_args()

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Detect and use appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if hasattr(torch, "mps") and torch.backends.mps.is_available() else 
                      "cpu")
print(f"Using device: {device}")

env = multiVehicleEnv()
# Get number of actions from action space
n_actions = len(env.action_list)
# Get the number of state observations
env.reset_test()
state_list = env.return_state_info()

choice = args.selection
print(f"Using selection method: {['DQN', 'Greedy', 'Shortest Distance'][choice]}")

# DQN model class would go here, or could be imported from a separate file
# For now we'll skip implementing a DQN since it's not the main focus

steps_done = 0
not_valid_time = 0

# This function would need to be implemented for a DQN approach
def select_action_dqn(state_list, policy_net):
    # Placeholder function since we're not implementing DQN
    print("DQN not implemented in this version")
    return -1

# Greedy action selection
def select_action_greedy(env, vehicle_index, node):
    # Prioritize battery charging if needed
    if env.check_action4_valide(vehicle_index):
        return 4
        
    # Check vehicle type
    is_drone = (env.vehicle_info[vehicle_index, 4] == 1)
    
    # Current node position affects valid actions
    current_position = node
    
    # Simple greedy strategy:
    # First priority: Recycle trays from B to keep the system running
    if env.check_action1_valide(vehicle_index):
        return 1
        
    # Second priority: Move parts from C to B if possible
    if env.check_action3_valide(vehicle_index):
        return 3
        
    # Third priority: Move parts directly from A to B
    if env.check_action0_valide(vehicle_index):
        return 0
        
    # Fourth priority: Move parts from A to C as buffer
    if env.check_action2_valide(vehicle_index):
        return 2
    
    # If current position prevents direct action, move to appropriate node
    if current_position == 'B' and (max(env.A_part_info[:,0]) == 1):
        # Go to A to pick up parts
        return 1  # This will move vehicle to A by recycling trays
        
    if current_position == 'A' and (max(env.B_info[:,0]) >= 1):
        # No valid action at A, but parts need to be processed at B
        return 0  # This will move vehicle to B with parts
    
    if current_position == 'C' and env.check_action1_valide(vehicle_index) == False:
        # Move to another node if nothing to do at C
        return 3  # Try to move parts from C to B if possible
        
    # No valid action
    return -1

# Shortest distance strategy
def select_action_shortest_distance(env, vehicle_index, node):
    # Prioritize battery charging if needed
    if env.check_action4_valide(vehicle_index):
        return 4
        
    # Check vehicle type
    is_drone = (env.vehicle_info[vehicle_index, 4] == 1)
    valid_actions = []
    
    # Check all valid actions
    if env.check_action0_valide(vehicle_index):
        # A to B
        if node == 'A':
            # Already at A, just need to go to B
            distance = env.t_AB_base
        elif node == 'B':
            # Already at B, need to go to A then B
            distance = env.t_AB_base * 2
        else:  # At C
            # Need to go to A then B
            distance = env.t_AC_base + env.t_AB_base
        valid_actions.append((0, distance))
        
    if env.check_action1_valide(vehicle_index):
        # B to A
        if node == 'B':
            # Already at B, just need to go to A
            distance = env.t_AB_base
        elif node == 'A':
            # Already at A, need to go to B then A
            distance = env.t_AB_base * 2
        else:  # At C
            # Need to go to B then A
            distance = env.t_BC_base + env.t_AB_base
        valid_actions.append((1, distance))
        
    if env.check_action2_valide(vehicle_index):
        # A to C
        if node == 'A':
            # Already at A, just need to go to C
            distance = env.t_AC_base
        elif node == 'B':
            # At B, need to go to A then C
            distance = env.t_AB_base + env.t_AC_base
        else:  # At C
            # Already at C, need to go to A then C
            distance = env.t_AC_base * 2
        valid_actions.append((2, distance))
        
    if env.check_action3_valide(vehicle_index):
        # C to B
        if node == 'C':
            # Already at C, just need to go to B
            distance = env.t_BC_base
        elif node == 'B':
            # At B, need to go to C then B
            distance = env.t_BC_base * 2
        else:  # At A
            # At A, need to go to C then B
            distance = env.t_AC_base + env.t_BC_base
        valid_actions.append((3, distance))
    
    # If we have valid actions, choose the one with shortest distance
    if valid_actions:
        # Sort by distance (second element of tuple)
        valid_actions.sort(key=lambda x: x[1])
        return valid_actions[0][0]
    
    # If no valid actions, suggest relocating based on current position
    if node == 'B' and (max(env.A_part_info[:,0]) == 1):
        # Go to A to pick up parts
        return 1  # This will move vehicle to A by recycling trays
        
    if node == 'A' and (max(env.B_info[:,0]) >= 1):
        # No valid action at A, but parts need to be processed at B
        return 0  # This will move vehicle to B with parts
    
    if node == 'C' and (max(env.B_info[:,0]) >= 1):
        # Move to another node if nothing to do at C
        return 3  # Try to move parts from C to B if possible
    
    return -1

final_reward = 0

env.reset_test()
state_list = env.return_state_info()

# Initialize vehicle states
A_vec = torch.tensor(state_list[0], dtype=torch.float32, device=device).unsqueeze(0)
B_vec = torch.tensor(state_list[1], dtype=torch.float32, device=device).unsqueeze(0)
C_vec = torch.tensor(state_list[2], dtype=torch.float32, device=device).unsqueeze(0)
vehicle_vec = torch.tensor(state_list[3], dtype=torch.float32, device=device).unsqueeze(0)

terminated = False

# Initial actions for all vehicles
for i in range(env.vehicle_num):
    vehicle_type = "Drone" if env.vehicle_info[i, 4] == 1 else "AGV"
    print(f"Initializing {vehicle_type} {i}")
    
    if choice == 0:
        # Placeholder for DQN
        action = -1
    elif choice == 1:
        action = select_action_greedy(env, i, 'A')
    else:
        action = select_action_shortest_distance(env, i, 'A')
    
    print(f"{vehicle_type} {i} at A, action: {action}")
    
    if action == 0: 
        reward = env.take_part_from_A(i, 'A', 'B')
    elif action == 1: 
        reward = env.recycle_tray_from_B_to_A(i, 'A')
    elif action == 2: 
        reward = env.take_part_from_A(i, 'A', 'C')
    elif action == 3: 
        reward = env.carry_partandtray_from_C_to_B(i, 'A')
    elif action == 4:
        reward = env.charge_vehicle(i, 'A')
    else:
        print(f"No valid action for {vehicle_type} {i}")
        reward = 0
        
    final_reward += reward

# Update state information
state_list = env.return_state_info()
A_vec = torch.tensor(state_list[0], dtype=torch.float32, device=device).unsqueeze(0)
B_vec = torch.tensor(state_list[1], dtype=torch.float32, device=device).unsqueeze(0)
C_vec = torch.tensor(state_list[2], dtype=torch.float32, device=device).unsqueeze(0)
vehicle_vec = torch.tensor(state_list[3], dtype=torch.float32, device=device).unsqueeze(0)

print("\nStarting simulation...")
start = time.time() 

# Main simulation loop
for t in range(1000):
    # Check if simulation is done
    if len(env.time_queue) == 0:
        print("Time queue empty, simulation finished")
        break
        
    # Sort and pop the earliest event
    env.time_queue.sort()
    current_time = env.time_queue.pop(0)
    
    # Get action data
    action_data = env.vehicle_action_buffer[current_time]
    message = action_data['action']
    node = action_data['node']
    vehicle_index = action_data['vehicle_index']
    action_num = action_data['action_num']
    
    print(f"Time {current_time:.1f}: {message}")
    
    # Execute the action's effects
    if action_num == 0 or action_num == 3:
        env.put_part_in_B(vehicle_index)
    elif action_num == 1:
        env.put_tray_in_A(vehicle_index)
    elif action_num == 2:
        env.put_part_in_C(vehicle_index)
    # No effect needed for action 4 (charging) as it's handled in charge_vehicle
    
    # Remove from buffer
    del env.vehicle_action_buffer[current_time]
    
    # Choose next action for this vehicle
    if choice == 0:
        # Placeholder for DQN
        action = -1
    elif choice == 1:
        action = select_action_greedy(env, vehicle_index, node)
    else:
        action = select_action_shortest_distance(env, vehicle_index, node)
    
    vehicle_type = "Drone" if env.vehicle_info[vehicle_index, 4] == 1 else "AGV"
    print(f"  Next action for {vehicle_type} {vehicle_index} at {node}: {action}")
    
    # Execute the chosen action
    if action == 0:
        reward = env.take_part_from_A(vehicle_index, node, 'B')
    elif action == 1:
        reward = env.recycle_tray_from_B_to_A(vehicle_index, node)
    elif action == 2:
        reward = env.take_part_from_A(vehicle_index, node, 'C')
    elif action == 3:
        reward = env.carry_partandtray_from_C_to_B(vehicle_index, node)
    elif action == 4:
        reward = env.charge_vehicle(vehicle_index, node)
    else:
        print(f"  No valid action for {vehicle_type} {vehicle_index}")
        reward = 0
        terminated = True
        
    final_reward += reward
    
    # Update state information
    if not terminated:
        next_state_list = env.return_state_info()
        A_vec = torch.tensor(next_state_list[0], dtype=torch.float32, device=device).unsqueeze(0)
        B_vec = torch.tensor(next_state_list[1], dtype=torch.float32, device=device).unsqueeze(0)
        C_vec = torch.tensor(next_state_list[2], dtype=torch.float32, device=device).unsqueeze(0)
        vehicle_vec = torch.tensor(next_state_list[3], dtype=torch.float32, device=device).unsqueeze(0)
    
    # Check if simulation has terminated
    if terminated:
        print("\nSimulation terminated early")
        break

# Print final statistics
print("\n----- Final Statistics -----")
print("A_part_info (first 10 rows):")
print(env.A_part_info[:10])
print("\nA_tray:")
print(env.A_tray)
print("\nB_info:")
print(env.B_info)
print("\nC_info:")
print(env.C_info)
print("\nvehicle_info:")
print(env.vehicle_info[:env.vehicle_num])
print("\nvehicle_battery:")
print(env.vehicle_battery[:env.vehicle_num])
print("\nvehicle_timer:")
print(env.vehicle_timer[:env.vehicle_num])

# Calculate throughput
processed_parts = env.part_num - np.sum(env.A_part_info[:, 0])
max_completion_time = np.max(env.vehicle_timer) if env.vehicle_timer.size > 0 else 0

print(f"\nProcessed {processed_parts}/{env.part_num} parts")
print(f"Final reward: {final_reward}")
print(f"Reward per part: {final_reward/env.part_num:.4f}")
print(f"Maximum completion time: {max_completion_time}")

# Print any remaining actions in the queue
if env.time_queue:
    print("\nRemaining actions in queue:")
    env.time_queue.sort()
    for time_point in env.time_queue:
        if time_point in env.vehicle_action_buffer:
            message = env.vehicle_action_buffer[time_point]['action']
            print(f"Time {time_point:.1f}: {message}")

# Print execution time
end = time.time()
elapsed_time = end - start
print(f"\nSimulation elapsed time: {elapsed_time:.4f} seconds") 