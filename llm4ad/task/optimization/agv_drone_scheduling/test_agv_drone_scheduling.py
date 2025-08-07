#!/usr/bin/env python3
"""
Test file for the AGV and drone scheduling environment.
This file runs simulations with different scheduling strategies and compares their performance.

Usage:
    python test_agv_drone_scheduling.py [--verbose] [--instances NUM] [--seed SEED]
"""

import argparse
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from multi_vehicle_env import multiVehicleEnv

def smart_scheduler(env, vehicle_index, current_node):
    """
    A more advanced scheduler that:
    1. Prioritizes battery charging
    2. Assigns different roles to AGVs and drones
    3. Considers traffic congestion
    4. Handles processing failures
    
    Args:
        env: The environment object
        vehicle_index: Index of the vehicle to schedule
        current_node: Current location of the vehicle ('A', 'B', or 'C')
        
    Returns:
        Action index (0-4) or -1 if no valid action
    """
    # First check if vehicle needs charging (critical)
    if env.need_charging(vehicle_index):
        if env.check_action4_valide(vehicle_index):
            return 4
            
    # Check vehicle type - use drones and AGVs differently
    is_drone = (env.vehicle_info[vehicle_index, 4] == 1)
    
    # Battery status - be cautious if battery is getting low but not critical
    battery_level = env.vehicle_battery[vehicle_index]
    battery_cautious = battery_level < env.battery_capacity * 0.35  # 35% threshold for caution
    
    if is_drone:
        # Drones are faster and less affected by traffic congestion
        # Focus on longer routes and time-critical tasks
        
        # First priority: B→A (recycle trays to keep production flowing)
        if env.check_action1_valide(vehicle_index):
            # Only do this if at B or battery is sufficient for trip
            if current_node == 'B' or (not battery_cautious):
                return 1
                
        # Second priority: C→B (move parts from buffer to production)
        if env.check_action3_valide(vehicle_index):
            if current_node == 'C' or (not battery_cautious):
                return 3
                
        # Third priority: A→B direct route (if battery good)
        if env.check_action0_valide(vehicle_index):
            if not battery_cautious:
                return 0
                
        # Fourth priority: A→C (buffer parts in C)
        if env.check_action2_valide(vehicle_index):
            if not battery_cautious:
                return 2
    else:
        # AGVs - more battery capacity but slower and affected by congestion
        # Focus on routine tasks and battery management
        
        # First priority: B→A (recycle trays)
        if env.check_action1_valide(vehicle_index) and current_node == 'B':
            # Prioritize when already at location B
            return 1
            
        # Second priority: A→C (short distance for AGVs)
        if env.check_action2_valide(vehicle_index) and current_node == 'A':
            return 2
            
        # Third priority: C→B (move from buffer when AGV is already at C)
        if env.check_action3_valide(vehicle_index) and current_node == 'C':
            return 3
            
        # Fourth priority: A→B (direct production route)
        if env.check_action0_valide(vehicle_index) and (current_node == 'A'):
            return 0
            
        # Check all other valid actions if no optimized path available
        if env.check_action1_valide(vehicle_index):
            return 1
        if env.check_action3_valide(vehicle_index):
            return 3
        if env.check_action0_valide(vehicle_index):
            return 0
        if env.check_action2_valide(vehicle_index):
            return 2
    
    # If battery is cautious and not at charging location, go charge
    if battery_cautious and env.check_action4_valide(vehicle_index):
        return 4
        
    # If no valid action or all actions rejected due to battery concerns
    # Try the basic valid actions as fallback
    if env.check_action1_valide(vehicle_index):
        return 1
    if env.check_action3_valide(vehicle_index):
        return 3
    if env.check_action0_valide(vehicle_index):
        return 0
    if env.check_action2_valide(vehicle_index):
        return 2
        
    # Last resort - try charging if at all possible
    if env.check_action4_valide(vehicle_index):
        return 4
    
    # No valid action found
    return -1

def greedy_scheduler(env, vehicle_index, current_node):
    """
    A greedy scheduler that prioritizes:
    1. Critical battery charging
    2. Actions that minimize travel time from current location
    3. Actions with highest efficiency (value/time ratio)
    
    Args:
        env: The environment object
        vehicle_index: Index of the vehicle to schedule
        current_node: Current location of the vehicle ('A', 'B', or 'C')
        
    Returns:
        Action index (0-4) or -1 if no valid action
    """
    # Critical battery check - only charge when absolutely necessary
    if env.need_charging(vehicle_index):
        if env.check_action4_valide(vehicle_index):
            return 4
    
    # Create a list of potential actions with travel time-based priority scores
    actions = []
    is_drone = (env.vehicle_info[vehicle_index, 4] == 1)
    
    # Calculate base times according to vehicle type
    speed_factor = env.drone_speed_factor if is_drone else 1.0
    time_AB = env.t_AB_base * speed_factor
    time_AC = env.t_AC_base * speed_factor
    time_BC = env.t_BC_base * speed_factor
    
    # Helper function to estimate travel time from current node to target node
    def calc_travel_time(target):
        if current_node == target:
            return 0
        elif (current_node == 'A' and target == 'B') or (current_node == 'B' and target == 'A'):
            return time_AB
        elif (current_node == 'A' and target == 'C') or (current_node == 'C' and target == 'A'):
            return time_AC
        elif (current_node == 'B' and target == 'C') or (current_node == 'C' and target == 'B'):
            return time_BC
        return 0
    
    # Check action 0: A→B (direct production)
    if env.check_action0_valide(vehicle_index):
        # Calculate time to pickup at A plus time to deliver to B
        time_to_A = calc_travel_time('A')
        total_time = time_to_A + time_AB
        # Higher value for production actions
        value = 100
        # Lower score means higher priority (less time)
        priority = total_time / value
        actions.append((0, priority))
        
    # Check action 1: B→A (recycle trays)
    if env.check_action1_valide(vehicle_index):
        # Calculate time to pickup at B plus time to deliver to A
        time_to_B = calc_travel_time('B')
        total_time = time_to_B + time_AB
        # High value for recycling to keep production going
        value = 90
        priority = total_time / value
        actions.append((1, priority))
        
    # Check action 2: A→C (buffering)
    if env.check_action2_valide(vehicle_index):
        # Calculate time to pickup at A plus time to deliver to C
        time_to_A = calc_travel_time('A')
        total_time = time_to_A + time_AC
        # Lower value for buffering
        value = 60
        priority = total_time / value
        actions.append((2, priority))
        
    # Check action 3: C→B (from buffer to production)
    if env.check_action3_valide(vehicle_index):
        # Calculate time to pickup at C plus time to deliver to B
        time_to_C = calc_travel_time('C')
        total_time = time_to_C + time_BC
        # High value for moving from buffer to production
        value = 95
        priority = total_time / value
        actions.append((3, priority))
    
    # Add charging as a low-priority option if battery is below 30%
    battery_level = env.vehicle_battery[vehicle_index]
    if battery_level < env.battery_capacity * 0.3 and env.check_action4_valide(vehicle_index):
        # Time to reach charging station at A
        time_to_A = calc_travel_time('A')
        # Priority increases as battery decreases
        battery_factor = (env.battery_capacity * 0.3 - battery_level) / (env.battery_capacity * 0.3)
        priority = time_to_A * (1 - battery_factor)  # Lower priority value for more urgent charging
        actions.append((4, priority))
    
    # Sort actions by priority (lowest time-to-value ratio first)
    actions.sort(key=lambda x: x[1])
    
    # Take the highest priority action
    if actions:
        return actions[0][0]
    
    # If no valid prioritized actions, try charging as last resort
    if env.check_action4_valide(vehicle_index):
        return 4
    
    # No valid action found
    return -1

def balanced_scheduler(env, vehicle_index, current_node):
    """
    A balanced scheduler that assigns vehicles to tasks based on their current location
    and capabilities, with efficient battery management
    
    Args:
        env: The environment object
        vehicle_index: Index of the vehicle to schedule
        current_node: Current location of the vehicle ('A', 'B', or 'C')
        
    Returns:
        Action index (0-4) or -1 if no valid action
    """
    # Critical battery check
    if env.check_action4_valide(vehicle_index):
        return 4
        
    # Get vehicle type and battery status
    is_drone = (env.vehicle_info[vehicle_index, 4] == 1)
    battery_level = env.vehicle_battery[vehicle_index]
    
    # Location-based priorities
    if current_node == 'A':
        # At location A, check in order: A→B, A→C
        if env.check_action0_valide(vehicle_index):
            return 0
        if env.check_action2_valide(vehicle_index):
            return 2
            
    elif current_node == 'B':
        # At location B, prioritize recycling trays
        if env.check_action1_valide(vehicle_index):
            return 1
            
    elif current_node == 'C':
        # At location C, prioritize moving parts to B
        if env.check_action3_valide(vehicle_index):
            return 3
    
    # If no location-based action was valid, try all valid actions
    if env.check_action1_valide(vehicle_index):
        return 1
    if env.check_action3_valide(vehicle_index):
        return 3
    if env.check_action0_valide(vehicle_index):
        return 0
    if env.check_action2_valide(vehicle_index):
        return 2
    
    # No valid action
    return -1

def simple_scheduler(env, vehicle_index, current_node):
    """Simple baseline scheduler"""
    # First check if vehicle needs charging
    if env.check_action4_valide(vehicle_index):
        return 4
        
    # Simple priority-based scheduling logic
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

def run_single_simulation(scheduler_func, verbose=False):
    """Run a single simulation with the given scheduler function"""
    env = multiVehicleEnv()
    env.reset_test()
    
    # Stats to track
    total_reward = 0
    charging_events = {i: 0 for i in range(env.vehicle_num)}
    idle_processing_events = 0
    traffic_jam_events = 0
    
    # Initialize vehicles
    for i in range(env.vehicle_num):
        vehicle_type = "Drone" if env.vehicle_info[i, 4] == 1 else "AGV"
        action = scheduler_func(env, i, 'A')
        
        if verbose:
            print(f"Initializing {vehicle_type} {i} with action {action}")
        
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
            charging_events[i] += 1
        else:
            reward = 0
            
        total_reward += reward
    
    # Main simulation loop
    max_steps = 1000
    for t in range(max_steps):
        if len(env.time_queue) == 0:
            if verbose:
                print("Time queue empty, simulation finished")
            break
            
        env.time_queue.sort()
        current_time = env.time_queue.pop(0)
        
        action_data = env.vehicle_action_buffer[current_time]
        vehicle_index = action_data['vehicle_index']
        node = action_data['node']
        action_num = action_data['action_num']
        
        if verbose:
            print(f"Time {current_time:.1f}: {action_data['action']}")
        
        # Apply action effects
        if action_num == 0 or action_num == 3:
            env.put_part_in_B(vehicle_index)
        elif action_num == 1:
            env.put_tray_in_A(vehicle_index)
        elif action_num == 2:
            env.put_part_in_C(vehicle_index)
        # Action 4 (charging) already applied effects
        
        del env.vehicle_action_buffer[current_time]
        
        # Choose and execute next action
        action = scheduler_func(env, vehicle_index, node)
        
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
            charging_events[vehicle_index] += 1
        else:
            reward = 0
            
        total_reward += reward
    
    # Calculate results
    processed_parts = env.part_num - np.sum(env.A_part_info[:, 0])
    completion_time = np.max(env.vehicle_timer) if env.vehicle_timer.size > 0 else 0
    
    # Count processed parts with idle status
    idle_count = 0
    for i in range(env.B_location_num):
        if env.B_info[i, 3] == 2:  # Idle processing status
            idle_count += 1
            
    # Gather results
    results = {
        "processed_parts": processed_parts,
        "total_parts": env.part_num,
        "completion_time": completion_time,
        "reward": total_reward,
        "avg_battery": np.mean(env.vehicle_battery[:env.vehicle_num]),
        "min_battery": np.min(env.vehicle_battery[:env.vehicle_num]),
        "charging_events": sum(charging_events.values()),
        "idle_processing": idle_count
    }
    
    if verbose:
        print("\n--- Final Statistics ---")
        print(f"Processed {processed_parts}/{env.part_num} parts")
        print(f"Completion time: {completion_time}")
        print(f"Total reward: {total_reward}")
        print(f"Charging events: {sum(charging_events.values())}")
        
        # Battery status by vehicle type
        agv_batteries = env.vehicle_battery[:env.AGV_num]
        drone_batteries = env.vehicle_battery[env.AGV_num:env.vehicle_num]
        print(f"Average AGV battery: {np.mean(agv_batteries):.2f}")
        print(f"Average Drone battery: {np.mean(drone_batteries):.2f}")
    
    return results

def run_test(scheduler_funcs, num_instances=5, verbose=False, seed=None):
    """Run multiple simulations with different schedulers and compare results"""
    if seed is not None:
        np.random.seed(seed)
        
    results = {}
    
    for name, func in scheduler_funcs.items():
        print(f"\nTesting {name} scheduler...")
        scheduler_results = []
        
        for i in range(num_instances):
            print(f"  Running instance {i+1}/{num_instances}")
            instance_result = run_single_simulation(func, verbose=verbose)
            scheduler_results.append(instance_result)
            
        # Compute averages
        avg_results = {}
        for key in scheduler_results[0].keys():
            avg_results[key] = np.mean([r[key] for r in scheduler_results])
            
        results[name] = {
            "all_results": scheduler_results,
            "avg_results": avg_results
        }
        
        print(f"  Average completion time: {avg_results['completion_time']:.2f}")
        print(f"  Average processed parts: {avg_results['processed_parts']:.1f}/{avg_results['total_parts']}")
        
    return results

def format_results_table(results):
    """Format results for display in a table"""
    data = []
    for name, result in results.items():
        avg = result["avg_results"]
        row = [
            name,
            f"{avg['processed_parts']:.1f}/{avg['total_parts']}",
            f"{avg['completion_time']:.2f}",
            f"{avg['reward']:.2f}",
            f"{avg['avg_battery']:.2f}",
            f"{avg['min_battery']:.2f}",
            f"{avg['charging_events']:.1f}",
            f"{avg['idle_processing']:.1f}"
        ]
        data.append(row)
        
    headers = [
        "Scheduler", "Parts Processed", "Completion Time", "Reward", 
        "Avg Battery", "Min Battery", "Charging Events", "Idle Processing"
    ]
    
    return tabulate(data, headers=headers, tablefmt="grid")

def plot_results(results):
    """Create comparison plots for the results"""
    scheduler_names = list(results.keys())
    metrics = [
        ("completion_time", "Completion Time"),
        ("processed_parts", "Parts Processed"),
        ("reward", "Total Reward"),
        ("avg_battery", "Average Battery Level"),
        ("charging_events", "Charging Events")
    ]
    
    # Set up the figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        data = []
        
        for name in scheduler_names:
            # Get all instance results for this metric
            metric_data = [r[metric] for r in results[name]["all_results"]]
            data.append(metric_data)
            
        # Create box plot
        ax.boxplot(data, tick_labels=scheduler_names)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add individual points
        for j, d in enumerate(data):
            x = np.random.normal(j+1, 0.04, size=len(d))
            ax.plot(x, d, 'o', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("scheduler_comparison.png")
    print("Plot saved as scheduler_comparison.png")

def main():
    parser = argparse.ArgumentParser(description="Test AGV and drone scheduling strategies")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--instances", type=int, default=5, help="Number of instances to run for each scheduler")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Define schedulers to test
    schedulers = {
        "Simple": simple_scheduler,
        "Greedy": greedy_scheduler,
        "Balanced": balanced_scheduler,
        "Smart": smart_scheduler
    }
    
    # Run tests
    print(f"Running {args.instances} test instances for each scheduler...")
    results = run_test(schedulers, num_instances=args.instances, verbose=args.verbose, seed=args.seed)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(format_results_table(results))
    
    # Plot comparisons
    plot_results(results)
    
    # Calculate overall best performer
    best_scheduler = None
    best_time = float('inf')
    
    for name, result in results.items():
        avg_time = result["avg_results"]["completion_time"]
        if avg_time < best_time:
            best_time = avg_time
            best_scheduler = name
            
    print(f"\nBest performing scheduler: {best_scheduler} (avg completion time: {best_time:.2f})")

if __name__ == "__main__":
    main() 