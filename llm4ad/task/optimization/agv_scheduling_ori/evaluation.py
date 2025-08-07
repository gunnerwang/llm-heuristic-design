from __future__ import annotations

from typing import Any
import numpy as np
import os
import sys
from llm4ad.base import Evaluation
from llm4ad.task.optimization.agv_scheduling_ori.template import template_program, task_description
from llm4ad.task.optimization.agv_scheduling_ori.multi_AGV_Env_1 import multiAGV_Env
import time
import torch

__all__ = ['AGVEvaluation']

class AGVEvaluation(Evaluation):
    """Evaluator for AGV scheduling problem."""

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
        self.env = multiAGV_Env()

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, scheduling_func: callable) -> float:
        """
        Evaluate the scheduling function over multiple instances.
        
        Args:
            scheduling_func: Function that selects the next action for an AGV
            
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
                
                # Initialize AGVs with actions
                for i in range(self.env.AGV_num):
                    action = scheduling_func(self.env, i, 'A')
                    if action == 0: 
                        reward = self.env.take_part_from_A(i, 'A', 'B')
                    elif action == 1: 
                        reward = self.env.recycle_tray_from_B_to_A(i, 'A')
                    elif action == 2: 
                        reward = self.env.take_part_from_A(i, 'A', 'C')
                    elif action == 3: 
                        reward = self.env.carry_partandtray_from_C_to_B(i, 'A')
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
                    parts_on_agvs = np.sum(self.env.AGV_info[:, 1] > 0)
                    total_parts_in_system = parts_in_a + parts_in_b + parts_in_c + parts_on_agvs
                    
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
                                    current_time = np.max(self.env.AGV_timer) if self.env.AGV_timer.size > 0 else 0
                                    
                                    print(f"Manually processing part {part_id} at B[{b_index}], finish time: {finish_time}")
                                    
                                    # If the part hasn't finished processing yet, we need to wait
                                    if current_time < finish_time:
                                        print(f"Waiting until time {finish_time} for processing to complete")
                                        # Store the AGV busy times before updating overall time
                                        agv_busy_times = {}
                                        for agv_idx in range(self.env.AGV_num):
                                            # Calculate the absolute time when each AGV will become available
                                            agv_busy_times[agv_idx] = current_time + self.env.AGV_info[agv_idx, 3]
                                        
                                        # Update the simulation time to finish_time
                                        current_time = finish_time
                                        # Update all AGV timers to the new current time
                                        for agv_idx in range(self.env.AGV_num):
                                            self.env.AGV_timer[agv_idx] = current_time
                                            # Update AGV busy timer (3rd column of AGV_info) to reflect remaining busy time
                                            # This preserves the original AGV availability time
                                            if agv_busy_times[agv_idx] > current_time:
                                                self.env.AGV_info[agv_idx, 3] = agv_busy_times[agv_idx] - current_time
                                            else:
                                                self.env.AGV_info[agv_idx, 3] = 0
                                        
                                        # Find the AGV that will be available first for recycling
                                        min_timer = float('inf')
                                        available_agv = -1
                                        for agv_idx in range(self.env.AGV_num):
                                            if self.env.AGV_info[agv_idx, 3] < min_timer:
                                                min_timer = self.env.AGV_info[agv_idx, 3]
                                                available_agv = agv_idx
                                        
                                        # Wait until the first AGV becomes available
                                        if min_timer > 0:
                                            print(f"Waiting until time {current_time + min_timer} for AGV {available_agv} to be available")
                                            current_time += min_timer
                                            # Update all AGV timers
                                            for agv_idx in range(self.env.AGV_num):
                                                self.env.AGV_timer[agv_idx] = current_time
                                                self.env.AGV_info[agv_idx, 3] = max(0, self.env.AGV_info[agv_idx, 3] - min_timer)
                                    
                                    # Find an AGV to recycle the tray
                                    recycling_agv = -1
                                    for agv_idx in range(self.env.AGV_num):
                                        if self.env.AGV_info[agv_idx, 3] == 0:  # Timer is 0, AGV is free
                                            recycling_agv = agv_idx
                                            break
                                    
                                    if recycling_agv != -1:
                                        # Mark the part as processed (keep the tray for now)
                                        self.env.B_info[b_index, 0] = 0
                                        
                                        # Move the AGV to B if it's not already there
                                        original_location = self.env.AGV_info[recycling_agv, 0]
                                        if original_location != 2:  # Not at B
                                            # Set AGV location to B
                                            self.env.AGV_info[recycling_agv, 0] = 2
                                        
                                        # Use the built-in recycle_tray_from_B_to_A method
                                        # This will properly update all state variables
                                        reward = self.env.recycle_tray_from_B_to_A(recycling_agv, 'B')
                                        final_reward += reward
                                        
                                        print(f"AGV {recycling_agv} recycled tray from B to A using action 1")
                                    else:
                                        print(f"ERROR: No available AGV to recycle tray for part {part_id}")
                                        # Even if we can't find an AGV, still count the part
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
                                current_time = np.max(self.env.AGV_timer) if self.env.AGV_timer.size > 0 else 0
                                
                                print(f"Found tray {tray_id} at B[{b_index}] that needs recycling")
                                
                                # Print current AGV status for debugging
                                for agv_idx in range(self.env.AGV_num):
                                    agv_location = 'A' if self.env.AGV_info[agv_idx, 0] == 1 else 'B' if self.env.AGV_info[agv_idx, 0] == 2 else 'C' if self.env.AGV_info[agv_idx, 0] == 3 else 'Unknown'
                                    print(f"AGV {agv_idx} at {agv_location}, busy for {self.env.AGV_info[agv_idx, 3]} more time units")
                                
                                # Wait for an AGV to be available if all are busy
                                if all(self.env.AGV_info[:, 3] > 0):  # If all AGVs have non-zero timers
                                    # Find the AGV that will become available first
                                    min_timer = np.min(self.env.AGV_info[:, 3])
                                    available_agv = np.argmin(self.env.AGV_info[:, 3])
                                    
                                    print(f"Waiting until time {current_time + min_timer} for AGV {available_agv} to be available")
                                    current_time += min_timer
                                    
                                    # Update all AGV timers
                                    for agv_idx in range(self.env.AGV_num):
                                        self.env.AGV_timer[agv_idx] = current_time
                                        self.env.AGV_info[agv_idx, 3] = max(0, self.env.AGV_info[agv_idx, 3] - min_timer)
                                
                                # Find an AGV to recycle the tray
                                recycling_agv = -1
                                for agv_idx in range(self.env.AGV_num):
                                    if self.env.AGV_info[agv_idx, 3] == 0:  # AGV is free
                                        recycling_agv = agv_idx
                                        break
                                
                                if recycling_agv != -1:
                                    # Move the AGV to B if it's not already there
                                    original_location = self.env.AGV_info[recycling_agv, 0]
                                    if original_location != 2:  # Not at B
                                        # Set AGV location to B
                                        self.env.AGV_info[recycling_agv, 0] = 2
                                    
                                    # Use the built-in recycle_tray_from_B_to_A method
                                    reward = self.env.recycle_tray_from_B_to_A(recycling_agv, 'B')
                                    final_reward += reward
                                    
                                    print(f"AGV {recycling_agv} recycled tray from B to A using action 1")
                                else:
                                    print(f"ERROR: No available AGV to recycle tray {tray_id}")
                        
                        if tray_recycle_needed:
                            continue
                        
                        # If we still have parts in the system but no actions in queue, we have a deadlock
                        if total_parts_in_system > 0:
                            print(f"DEADLOCK: Still have {total_parts_in_system} parts in system but no actions in queue")
                            print(f"Parts in A: {parts_in_a}, B: {parts_in_b}, C: {parts_in_c}, On AGVs: {parts_on_agvs}")
                            break
                        
                        # If no parts in system, we're done
                        break
                    
                    # Pop the earliest finishing action from the queue
                    self.env.time_queue.sort()
                    times = self.env.time_queue.pop(0)
                    
                    # Get action details from the buffer
                    message = self.env.agv_action_buffer[times]['action']
                    node = self.env.agv_action_buffer[times]['node']
                    AGV_index = self.env.agv_action_buffer[times]['AGV_index']
                    action_num = self.env.agv_action_buffer[times]['action_num']
                    
                    print(f"Step {t}: Time {times}, {message}")
                    
                    # Complete the action
                    if action_num == 0 or action_num == 3:
                        self.env.put_part_in_B(AGV_index)
                    elif action_num == 1:
                        self.env.put_tray_in_A(AGV_index)
                    elif action_num == 2:
                        self.env.put_part_in_C(AGV_index)
                    
                    # Remove the action from the buffer
                    del self.env.agv_action_buffer[times]
                    
                    # Let the now-free AGV select a new action
                    action = scheduling_func(self.env, AGV_index, node)
                    
                    # Execute the selected action
                    if action == 0: 
                        reward = self.env.take_part_from_A(AGV_index, node, 'B')
                    elif action == 1: 
                        reward = self.env.recycle_tray_from_B_to_A(AGV_index, node)
                    elif action == 2: 
                        reward = self.env.take_part_from_A(AGV_index, node, 'C')
                    elif action == 3: 
                        reward = self.env.carry_partandtray_from_C_to_B(AGV_index, node)
                    else:
                        print(f"No valid action for AGV {AGV_index} at node {node}")
                        continue
                    
                    final_reward += reward
                    
                    # Print status every 100 steps
                    if t % 100 == 0:
                        parts_in_a = np.sum(self.env.A_part_info[:, 0])
                        parts_in_b = np.sum(self.env.B_info[:, 0] > 0)
                        parts_in_c = np.sum(self.env.C_info[:, 0] > 0)
                        parts_on_agvs = np.sum(self.env.AGV_info[:, 1] > 0)
                        print(f"Parts in A: {parts_in_a}, B: {parts_in_b}, C: {parts_in_c}, On AGVs: {parts_on_agvs}")
                
                # Calculate final statistics 
                parts_in_a = np.sum(self.env.A_part_info[:, 0])
                parts_in_b = np.sum(self.env.B_info[:, 0] > 0)
                parts_in_c = np.sum(self.env.C_info[:, 0] > 0)
                parts_on_agvs = np.sum(self.env.AGV_info[:, 1] > 0)
                
                # Get the highest AGV timer as completion time
                if self.env.AGV_timer.size > 0:
                    completion_time = np.max(self.env.AGV_timer)
                else:
                    completion_time = 0
                
                # Check if all parts have been processed
                parts_processed = initial_parts - (parts_in_a + parts_in_b + parts_in_c + parts_on_agvs)
                
                # Print final state
                print("Final state:")
                print(f"Parts in A: {parts_in_a}")
                print(f"Trays in A: {np.sum(self.env.A_tray)}")
                print(f"Parts in B: {parts_in_b}")
                print(f"Parts in C: {parts_in_c}")
                print(f"Parts on AGVs: {parts_on_agvs}")
                print(f"Parts processed: {parts_processed}/{initial_parts}")
                print(f"Final reward: {final_reward}")
                print(f"Completion time: {completion_time}")
                
                # Success only if ALL parts are processed (100% completion)
                if parts_processed == initial_parts:
                    total_completion_time += completion_time
                    num_completed_instances += 1
                    print(f"Instance {instance+1} completed successfully: processed {parts_processed}/{initial_parts} parts")
                else:
                    print(f"Instance {instance+1} failed: only processed {parts_processed}/{initial_parts} parts")
                
            except Exception as e:
                print(f"Error in simulation instance {instance+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Return negative of average completion time (higher is better for optimization)
        if num_completed_instances > 0:
            avg_completion_time = total_completion_time / num_completed_instances
            print(f"Average completion time across {num_completed_instances} instances: {avg_completion_time}")
            return -avg_completion_time
        else:
            print("WARNING: No instances were successfully completed. Check the environment and scheduler logic.")
            return -float('inf')  # Return worst possible score if no instances completed

if __name__ == '__main__':
    # Example scheduling function for testing
    def simple_scheduler(env, agv_index, current_node):
        """A simple greedy scheduler that prioritizes finishing parts in B first"""
        # First, determine current node
        if env.AGV_info[agv_index, 0] == 1:
            current_node = 'A'
        elif env.AGV_info[agv_index, 0] == 2:
            current_node = 'B'
        elif env.AGV_info[agv_index, 0] == 3:
            current_node = 'C'
            
        # Check if there are finished parts at B to collect (action 1)
        if env.check_action1_valide() and current_node == 'B':
            return 1
            
        # If at A and can take part to B, do that first
        if env.check_action0_valide() and current_node == 'A':
            return 0
            
        # If at C and can move parts to B, do that
        if env.check_action3_valide() and current_node == 'C':
            return 3
            
        # If at A and can take part to C, do that
        if env.check_action2_valide() and current_node == 'A':
            return 2
            
        # If no valid action found
        return -1
    
    # Test the evaluation
    evaluation = AGVEvaluation(n_instance=3)
    result = evaluation.evaluate(simple_scheduler)
    print(f"Test score with simple scheduler: {-result}")  # Negative since evaluate returns negative completion time
    