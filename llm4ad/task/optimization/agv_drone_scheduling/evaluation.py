from __future__ import annotations

from typing import Any
import numpy as np
import os
import sys
from llm4ad.base import Evaluation
from llm4ad.task.optimization.agv_drone_scheduling.template import template_program, task_description
from llm4ad.task.optimization.agv_drone_scheduling.multi_vehicle_env import multiVehicleEnv
import time
import torch

__all__ = ['VehicleSchedulingEvaluation']

class VehicleSchedulingEvaluation(Evaluation):
    """Evaluator for AGV and drone scheduling problem."""

    def __init__(self,
                 timeout_seconds=60,
                 n_instance=10,
                 **kwargs):

        """
            Args:
                timeout_seconds: Maximum allowed time (in seconds) for the evaluation process
                n_instance: Number of problem instances to evaluate
            Raises:
                AttributeError: If the data key does not exist.
                FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.n_instance = n_instance
        self.env = multiVehicleEnv()

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, scheduling_func: callable) -> float:
        """
        Evaluate the scheduling function over multiple instances.
        
        Args:
            scheduling_func: Function that selects the next action for a vehicle
            
        Returns:
            The negative of the average completion time (higher is better)
        """
        total_completion_time = 0
        num_completed_instances = 0
        
        for instance in range(self.n_instance):
            print(f"\nRunning instance {instance+1}/{self.n_instance}")
            try:
                # Reset environment for a new instance
                self.env.reset_test()
                
                # Track initial part counts
                initial_parts = self.env.part_num
                print(f"Initial parts to process: {initial_parts}")
                
                # Track processed parts
                final_reward = 0
                
                # Initialize vehicles with actions
                for i in range(self.env.vehicle_num):
                    vehicle_type = "Drone" if self.env.vehicle_info[i, 4] == 1 else "AGV"
                    current_node = 'A'  # All vehicles start at A
                    
                    action = scheduling_func(self.env, i, current_node)
                    if action == 0: 
                        reward = self.env.take_part_from_A(i, current_node, 'B')
                    elif action == 1: 
                        reward = self.env.recycle_tray_from_B_to_A(i, current_node)
                    elif action == 2: 
                        reward = self.env.take_part_from_A(i, current_node, 'C')
                    elif action == 3: 
                        reward = self.env.carry_partandtray_from_C_to_B(i, current_node)
                    elif action == 4:
                        reward = self.env.charge_vehicle(i, current_node)
                    else:
                        reward = 0  # No valid action
                    final_reward += reward
                
                # Main simulation loop using time queue
                max_steps = 1000  # Prevent infinite loops
                completed = False
                
                for t in range(max_steps):
                    # Check if all parts are completely processed
                    parts_in_a = np.sum(self.env.A_part_info[:, 0])
                    parts_in_b = np.sum(self.env.B_info[:, 0] > 0)
                    parts_in_c = np.sum(self.env.C_info[:, 0] > 0)
                    parts_on_vehicles = np.sum(self.env.vehicle_info[:, 1] > 0)
                    total_parts_in_system = parts_in_a + parts_in_b + parts_in_c + parts_on_vehicles
                    
                    if total_parts_in_system == 0:
                        print("All parts fully processed, simulation complete")
                        completed = True
                        break
                    
                    # Check if time queue is empty - we need to handle the case where parts might still be processing
                    if not self.env.time_queue:
                        print(f"No more actions in queue at step {t}. Checking if processing is complete...")
                        
                        # If there are still parts in B but the queue is empty, handle the remaining parts
                        if parts_in_b > 0:
                            print(f"Found {parts_in_b} parts still in B. Processing them manually.")
                            
                            # Process each part in B and handle finishing
                            for b_index in range(self.env.B_location_num):
                                if self.env.B_info[b_index, 0] > 0:  # There's a part here
                                    part_id = self.env.B_info[b_index, 0]
                                    finish_time = self.env.B_info[b_index, 2]
                                    processing_status = self.env.B_info[b_index, 3]  # Check processing status
                                    current_time = np.max(self.env.vehicle_timer) if self.env.vehicle_timer.size > 0 else 0
                                    
                                    print(f"Manually processing part {part_id} at B[{b_index}], finish time: {finish_time}, status: {'normal' if processing_status == 1 else 'idle' if processing_status == 2 else 'unknown'}")
                                    
                                    # If the part hasn't finished processing yet, we need to wait
                                    if current_time < finish_time:
                                        print(f"Waiting until time {finish_time} for processing to complete")
                                        # Store the vehicle busy times before updating overall time
                                        vehicle_busy_times = {}
                                        for vehicle_idx in range(self.env.vehicle_num):
                                            # Calculate the absolute time when each vehicle will become available
                                            vehicle_busy_times[vehicle_idx] = current_time + self.env.vehicle_info[vehicle_idx, 3]
                                        
                                        # Update the simulation time to finish_time
                                        current_time = finish_time
                                        # Update all vehicle timers to the new current time
                                        for vehicle_idx in range(self.env.vehicle_num):
                                            self.env.vehicle_timer[vehicle_idx] = current_time
                                            # Update vehicle busy timer (3rd column of vehicle_info) to reflect remaining busy time
                                            # This preserves the original vehicle availability time
                                            if vehicle_busy_times[vehicle_idx] > current_time:
                                                self.env.vehicle_info[vehicle_idx, 3] = vehicle_busy_times[vehicle_idx] - current_time
                                            else:
                                                self.env.vehicle_info[vehicle_idx, 3] = 0
                                        
                                        # Find the vehicle that will be available first for recycling
                                        min_timer = float('inf')
                                        available_vehicle = -1
                                        for vehicle_idx in range(self.env.vehicle_num):
                                            if self.env.vehicle_info[vehicle_idx, 3] < min_timer:
                                                min_timer = self.env.vehicle_info[vehicle_idx, 3]
                                                available_vehicle = vehicle_idx
                                        
                                        # Wait until the first vehicle becomes available
                                        if min_timer > 0:
                                            vehicle_type = "Drone" if self.env.vehicle_info[available_vehicle, 4] == 1 else "AGV"
                                            print(f"Waiting until time {current_time + min_timer} for {vehicle_type} {available_vehicle} to be available")
                                            current_time += min_timer
                                            # Update all vehicle timers
                                            for vehicle_idx in range(self.env.vehicle_num):
                                                self.env.vehicle_timer[vehicle_idx] = current_time
                                                self.env.vehicle_info[vehicle_idx, 3] = max(0, self.env.vehicle_info[vehicle_idx, 3] - min_timer)
                                    
                                    # Find a vehicle to recycle the tray
                                    recycling_vehicle = -1
                                    for vehicle_idx in range(self.env.vehicle_num):
                                        if self.env.vehicle_info[vehicle_idx, 3] == 0:  # Timer is 0, vehicle is free
                                            recycling_vehicle = vehicle_idx
                                            break
                                    
                                    if recycling_vehicle != -1:
                                        # Mark the part as processed (keep the tray for now)
                                        self.env.B_info[b_index, 0] = 0
                                        
                                        # Move the vehicle to B if it's not already there
                                        original_location = self.env.vehicle_info[recycling_vehicle, 0]
                                        if original_location != 2:  # Not at B
                                            # Set vehicle location to B
                                            self.env.vehicle_info[recycling_vehicle, 0] = 2
                                        
                                        # Use the built-in recycle_tray_from_B_to_A method
                                        # This will properly update all state variables
                                        reward = self.env.recycle_tray_from_B_to_A(recycling_vehicle, 'B')
                                        final_reward += reward
                                        
                                        vehicle_type = "Drone" if self.env.vehicle_info[recycling_vehicle, 4] == 1 else "AGV"
                                        print(f"{vehicle_type} {recycling_vehicle} recycled tray from B to A using action 1")
                                    else:
                                        print(f"ERROR: No available vehicle to recycle tray for part {part_id}")
                                        # Even if we can't find a vehicle, still count the part
                                        self.env.B_info[b_index, 0] = 0
                                        final_reward += 1
                                    
                            # Force another check of all parts
                            continue
                        
                        # Handle empty slots in B with trays that need recycling
                        tray_recycle_needed = False
                        for b_index in range(self.env.B_location_num):
                            if self.env.B_info[b_index, 0] == 0 and self.env.B_info[b_index, 1] > 0:
                                tray_recycle_needed = True
                                tray_id = self.env.B_info[b_index, 1]
                                current_time = np.max(self.env.vehicle_timer) if self.env.vehicle_timer.size > 0 else 0
                                
                                print(f"Found tray {tray_id} at B[{b_index}] that needs recycling")
                                
                                # Print current vehicle status for debugging
                                for vehicle_idx in range(self.env.vehicle_num):
                                    vehicle_type = "Drone" if self.env.vehicle_info[vehicle_idx, 4] == 1 else "AGV"
                                    vehicle_location = 'A' if self.env.vehicle_info[vehicle_idx, 0] == 1 else 'B' if self.env.vehicle_info[vehicle_idx, 0] == 2 else 'C' if self.env.vehicle_info[vehicle_idx, 0] == 3 else 'Unknown'
                                    print(f"{vehicle_type} {vehicle_idx} at {vehicle_location}, busy for {self.env.vehicle_info[vehicle_idx, 3]} more time units, battery: {self.env.vehicle_battery[vehicle_idx]:.1f}")
                                
                                # Wait for a vehicle to be available if all are busy
                                if all(self.env.vehicle_info[:, 3] > 0):  # If all vehicles have non-zero timers
                                    # Find the vehicle that will become available first
                                    min_timer = np.min(self.env.vehicle_info[:, 3])
                                    available_vehicle = np.argmin(self.env.vehicle_info[:, 3])
                                    vehicle_type = "Drone" if self.env.vehicle_info[available_vehicle, 4] == 1 else "AGV"
                                    
                                    print(f"Waiting until time {current_time + min_timer} for {vehicle_type} {available_vehicle} to be available")
                                    current_time += min_timer
                                    
                                    # Update all vehicle timers
                                    for vehicle_idx in range(self.env.vehicle_num):
                                        self.env.vehicle_timer[vehicle_idx] = current_time
                                        self.env.vehicle_info[vehicle_idx, 3] = max(0, self.env.vehicle_info[vehicle_idx, 3] - min_timer)
                                
                                # Find a vehicle to recycle the tray
                                recycling_vehicle = -1
                                for vehicle_idx in range(self.env.vehicle_num):
                                    if self.env.vehicle_info[vehicle_idx, 3] == 0:  # Vehicle is free
                                        # Check battery level
                                        if self.env.check_battery_sufficient(vehicle_idx, 'B', 'A'):
                                            recycling_vehicle = vehicle_idx
                                            break
                                
                                if recycling_vehicle != -1:
                                    # Move the vehicle to B if it's not already there
                                    original_location = self.env.vehicle_info[recycling_vehicle, 0]
                                    if original_location != 2:  # Not at B
                                        # Set vehicle location to B
                                        self.env.vehicle_info[recycling_vehicle, 0] = 2
                                    
                                    # Use the built-in recycle_tray_from_B_to_A method
                                    reward = self.env.recycle_tray_from_B_to_A(recycling_vehicle, 'B')
                                    final_reward += reward
                                    
                                    vehicle_type = "Drone" if self.env.vehicle_info[recycling_vehicle, 4] == 1 else "AGV"
                                    print(f"{vehicle_type} {recycling_vehicle} recycled tray from B to A using action 1")
                                else:
                                    print(f"ERROR: No available vehicle with sufficient battery to recycle tray {tray_id}")
                        
                        if tray_recycle_needed:
                            continue
                        
                        # If we still have parts in the system but no actions in queue, we have a deadlock
                        if total_parts_in_system > 0:
                            print(f"DEADLOCK: Still have {total_parts_in_system} parts in system but no actions in queue")
                            print(f"Parts in A: {parts_in_a}, B: {parts_in_b}, C: {parts_in_c}, On vehicles: {parts_on_vehicles}")
                            break
                        
                        # If no parts in system, we're done
                        completed = True
                        break
                    
                    # Sort time queue to get the earliest action
                    self.env.time_queue.sort()
                    
                    # Get the earliest action from the queue
                    current_time = self.env.time_queue.pop(0)
                    action_data = self.env.vehicle_action_buffer[current_time]
                    
                    print(action_data['action'])
                    
                    vehicle_idx = action_data['vehicle_index']
                    node = action_data['node']
                    action_num = action_data['action_num']
                    
                    # Apply the action's effect
                    if action_num == 0 or action_num == 3:
                        self.env.put_part_in_B(vehicle_idx)
                    elif action_num == 1:
                        self.env.put_tray_in_A(vehicle_idx)
                    elif action_num == 2:
                        self.env.put_part_in_C(vehicle_idx)
                    # No else for action 4 (charging) as it's already handled in the charge_vehicle method
                    
                    # Remove this action from the buffer
                    del self.env.vehicle_action_buffer[current_time]
                    
                    # The vehicle that just completed its action needs a new task
                    action = scheduling_func(self.env, vehicle_idx, node)
                    
                    if action == 0:
                        reward = self.env.take_part_from_A(vehicle_idx, node, 'B')
                    elif action == 1:
                        reward = self.env.recycle_tray_from_B_to_A(vehicle_idx, node)
                    elif action == 2:
                        reward = self.env.take_part_from_A(vehicle_idx, node, 'C')
                    elif action == 3:
                        reward = self.env.carry_partandtray_from_C_to_B(vehicle_idx, node)
                    elif action == 4:
                        reward = self.env.charge_vehicle(vehicle_idx, node)
                    else:
                        # No valid action for this vehicle at this time
                        print(f"No valid action for {'Drone' if self.env.vehicle_info[vehicle_idx, 4] == 1 else 'AGV'} {vehicle_idx} at {node}")
                    
                    final_reward += reward
                
                if completed:
                    print(f"Instance {instance+1} completed successfully")
                    
                    # Completion time is the maximum timer value across all vehicles
                    completion_time = np.max(self.env.vehicle_timer)
                    print(f"Completion time: {completion_time}")
                    
                    total_completion_time += completion_time
                    num_completed_instances += 1
                else:
                    print(f"Instance {instance+1} failed to complete all parts")
                
            except Exception as e:
                print(f"Error in instance {instance+1}: {e}")
        
        if num_completed_instances > 0:
            avg_completion_time = total_completion_time / num_completed_instances
            print(f"\nAverage completion time: {avg_completion_time}")
            return -avg_completion_time  # Negative because lower is better
        else:
            print("\nNo instances completed successfully")
            return float('-inf')  # Worst possible score

    def simple_scheduler(env, vehicle_index, current_node):
        """A simple baseline scheduler that only considers the basic task."""
        # First check if vehicle needs charging
        if env.check_action4_valide(vehicle_index):
            return 4
            
        # Check if vehicle is an AGV or drone
        is_drone = (env.vehicle_info[vehicle_index, 4] == 1)
        
        # Very simple scheduling logic for demonstration
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