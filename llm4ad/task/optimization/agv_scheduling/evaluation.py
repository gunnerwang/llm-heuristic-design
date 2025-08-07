from __future__ import annotations

from typing import Any
import numpy as np
import os
import sys
from llm4ad.base import Evaluation
from llm4ad.task.optimization.agv_scheduling.template import template_program, task_description

# Import the AGV environment
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)  # Add the current directory to path
from multi_AGV_Env_1 import multiAGV_Env

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
                self.env.reset()
                
                # Run simulation until all parts are processed or timeout
                max_steps = 1000  # Prevent infinite loops
                step_count = 0
                
                # Track initial part counts
                initial_parts = self.env.part_num
                print(f"Initial parts to process: {initial_parts}")
                
                # Track processed parts (those that completed processing in B)
                processed_parts = 0
                total_parts = initial_parts
                
                # Initialize some AGVs to start processing
                print("Initializing AGVs...")
                
                # Initialize up to 3 AGVs to take parts from A to B (never directly to C)
                # Parts must be processed at B first before they can go elsewhere
                for agv_index in range(min(3, self.env.AGV_num)):
                    if self.env.check_action0_valide():  # Can take part from A to B
                        self.env.take_part_from_A(agv_index, 'A', 'B')
                        print(f"AGV {agv_index} initialized with action 0 (A->B)")
                        # Print the AGV's state after initialization
                        location = "A" if self.env.AGV_info[agv_index, 0] == 1 else "B" if self.env.AGV_info[agv_index, 0] == 2 else "C"
                        print(f"  AGV {agv_index} state: Location={location}, Part={self.env.AGV_info[agv_index, 1]}, Tray={self.env.AGV_info[agv_index, 2]}, Timer={self.env.AGV_info[agv_index, 3]}")
                    else:
                        print(f"AGV {agv_index} not initialized - can't take more parts to B")
                        break
                
                print(f"Total parts to process: {total_parts}")
                
                # Global simulation time - we'll use the current step as the time unit
                simulation_time = 0
                time_step = 50  # Time advancement per simulation step
                
                # Print all AGV state information
                print("\nAGV States after initialization:")
                for agv_index in range(self.env.AGV_num):
                    location = "A" if self.env.AGV_info[agv_index, 0] == 1 else "B" if self.env.AGV_info[agv_index, 0] == 2 else "C"
                    print(f"  AGV {agv_index}: Location={location}, Part={self.env.AGV_info[agv_index, 1]}, Tray={self.env.AGV_info[agv_index, 2]}, Timer={self.env.AGV_info[agv_index, 3]}")
                    print(f"  AGV {agv_index} Time Registered: {self.env.AGV_timer[agv_index]}")
                
                while processed_parts < total_parts and step_count < max_steps:
                    # Advance simulation time
                    simulation_time += time_step
                    print(f"\nStep {step_count} (Sim time {simulation_time})")
                    
                    # Count parts in system
                    parts_in_a = np.sum(self.env.A_part_info[:, 0])
                    parts_in_b = np.sum(self.env.B_info[:, 0] > 0)  # Count only non-zero entries
                    parts_in_c = np.sum(self.env.C_info[:, 0] > 0)  # Count only non-zero entries
                    
                    # Print system state 
                    if step_count % 10 == 0:
                        print(f"Parts in A: {parts_in_a}, B: {parts_in_b}, C: {parts_in_c}, Processed: {processed_parts}")
                        # Print all AGV state information
                        print("AGV States:")
                        for agv_index in range(self.env.AGV_num):
                            location = "A" if self.env.AGV_info[agv_index, 0] == 1 else "B" if self.env.AGV_info[agv_index, 0] == 2 else "C"
                            print(f"  AGV {agv_index}: Location={location}, Part={self.env.AGV_info[agv_index, 1]}, Tray={self.env.AGV_info[agv_index, 2]}, Timer={self.env.AGV_info[agv_index, 3]}")
                    
                    # For each AGV, decrease its timer by time_step (but not below 0)
                    for agv_index in range(self.env.AGV_num):
                        if self.env.AGV_info[agv_index, 3] > 0:
                            self.env.AGV_info[agv_index, 3] = max(0, self.env.AGV_info[agv_index, 3] - time_step)
                            print(f"Decremented timer for AGV {agv_index} to {self.env.AGV_info[agv_index, 3]}")
                    
                    # Update AGV global timers to match simulation time
                    for agv_index in range(self.env.AGV_num):
                        self.env.AGV_timer[agv_index] = simulation_time
                    
                    # Process parts in B that have finished processing
                    for b_index in range(self.env.B_location_num):
                        # Check if there's a part at this position and if its processing time has elapsed
                        if self.env.B_info[b_index, 0] > 0:
                            if simulation_time >= self.env.B_info[b_index, 2]:
                                # Part has completed processing
                                part_id = self.env.B_info[b_index, 0]
                                print(f"Sim time {simulation_time}: Part {part_id} at B position {b_index} finished processing!")
                                
                                # Mark as processed but keep tray info for recycling
                                processed_parts += 1
                                print(f"Sim time {simulation_time}: Processed {processed_parts}/{total_parts} parts")
                                
                                # Clear part ID while keeping tray info for recycling
                                self.env.B_info[b_index, 0] = 0
                            else:
                                # Part still processing
                                time_left = self.env.B_info[b_index, 2] - simulation_time
                                if step_count % 5 == 0:
                                    print(f"Sim time {simulation_time}: Part at B position {b_index} processing - {time_left} time units remaining")
                    
                    # Check which AGVs are available for new actions
                    agvs_available = []
                    for agv_index in range(self.env.AGV_num):
                        # Check timer - if it's 0, the AGV is available
                        if self.env.AGV_info[agv_index, 3] == 0:
                            # If the AGV is at location A (code 1) and has a tray but no part, 
                            # it means it just finished action 1 (recycling tray to A)
                            if self.env.AGV_info[agv_index, 0] == 1 and self.env.AGV_info[agv_index, 1] == 0 and self.env.AGV_info[agv_index, 2] > 0:
                                print(f"Sim time {simulation_time}: AGV {agv_index} placing tray {self.env.AGV_info[agv_index, 2]} back at A")
                                self.env.put_tray_in_A(agv_index)
                                
                            agvs_available.append(agv_index)
                    
                    # If AGVs are available, let them take actions
                    if agvs_available:
                        print(f"Sim time {simulation_time}: {len(agvs_available)} AGVs available for actions: {agvs_available}")
                        
                        # For each available AGV, select and execute an action
                        for agv_index in agvs_available:
                            # Determine current node
                            current_node = 'A'  # Default
                            if self.env.AGV_info[agv_index, 0] == 2:
                                current_node = 'B'
                            elif self.env.AGV_info[agv_index, 0] == 3:
                                current_node = 'C'
                            
                            # Check which actions are valid for this AGV
                            valid_actions = []
                            if self.env.check_action0_valide():
                                valid_actions.append(0)  # A->B
                            if self.env.check_action1_valide():
                                valid_actions.append(1)  # B->A
                            if self.env.check_action2_valide():
                                valid_actions.append(2)  # A->C
                            if self.env.check_action3_valide():
                                valid_actions.append(3)  # C->B
                            
                            print(f"AGV {agv_index} at {current_node}: Valid actions = {valid_actions}")
                            
                            # Manually check validity of all actions
                            action0_valid = self.env.check_action0_valide()
                            action1_valid = self.env.check_action1_valide()
                            action2_valid = self.env.check_action2_valide()
                            action3_valid = self.env.check_action3_valide()
                            
                            print(f"Manual checks: action0_valid={action0_valid}, action1_valid={action1_valid}, action2_valid={action2_valid}, action3_valid={action3_valid}")
                            
                            # Select action using scheduler
                            action = scheduling_func(self.env, agv_index, current_node)
                            print(f"Scheduler selected action {action} for AGV {agv_index}")
                            
                            # Execute the selected action with proper location check
                            executed = False
                            if action == 0 and action0_valid:  # Take part from A to B
                                self.env.take_part_from_A(agv_index, current_node, 'B')
                                print(f"Sim time {simulation_time}: AGV {agv_index} at {current_node} taking action 0 (A->B)")
                                executed = True
                            elif action == 1 and action1_valid:  # Recycle tray from B to A
                                self.env.recycle_tray_from_B_to_A(agv_index, current_node)
                                print(f"Sim time {simulation_time}: AGV {agv_index} at {current_node} taking action 1 (B->A)")
                                executed = True
                            elif action == 2 and action2_valid:  # Take part from A to C
                                self.env.take_part_from_A(agv_index, current_node, 'C')
                                # self.env.put_part_in_C(agv_index)
                                print(f"Sim time {simulation_time}: AGV {agv_index} at {current_node} taking action 2 (A->C)")
                                executed = True
                            elif action == 3 and action3_valid:  # Move part from C to B
                                self.env.carry_partandtray_from_C_to_B(agv_index, current_node)
                                print(f"Sim time {simulation_time}: AGV {agv_index} at {current_node} taking action 3 (C->B)")
                                executed = True
                            
                            # If no action was executed but valid actions exist, use the first valid action
                            if not executed and valid_actions:
                                fallback_action = valid_actions[0]
                                print(f"Sim time {simulation_time}: Using fallback action {fallback_action} for AGV {agv_index}")
                                
                                if fallback_action == 0:
                                    self.env.take_part_from_A(agv_index, current_node, 'B')
                                    print(f"Sim time {simulation_time}: AGV {agv_index} at {current_node} taking fallback action 0 (A->B)")
                                elif fallback_action == 1:
                                    self.env.recycle_tray_from_B_to_A(agv_index, current_node)
                                    print(f"Sim time {simulation_time}: AGV {agv_index} at {current_node} taking fallback action 1 (B->A)")
                                elif fallback_action == 2:
                                    self.env.take_part_from_A(agv_index, current_node, 'C')
                                    self.env.put_part_in_C(agv_index)
                                    print(f"Sim time {simulation_time}: AGV {agv_index} at {current_node} taking fallback action 2 (A->C)")
                                elif fallback_action == 3:
                                    self.env.carry_partandtray_from_C_to_B(agv_index, current_node)
                                    print(f"Sim time {simulation_time}: AGV {agv_index} at {current_node} taking fallback action 3 (C->B)")
                                
                                executed = True
                            
                            if executed:
                                print(f"After action, AGV {agv_index} state: Location={self.env.AGV_info[agv_index, 0]}, Part={self.env.AGV_info[agv_index, 1]}, Tray={self.env.AGV_info[agv_index, 2]}, Timer={self.env.AGV_info[agv_index, 3]}")
                            else:
                                print(f"Sim time {simulation_time}: AGV {agv_index} at {current_node} has no valid action")
                    else:
                        print(f"Sim time {simulation_time}: No AGVs available for actions")
                    
                    # Print the status of B
                    if step_count % 10 == 0:
                        print("B status:")
                        for b_index in range(self.env.B_location_num):
                            if self.env.B_info[b_index, 0] > 0:
                                time_left = max(0, self.env.B_info[b_index, 2] - simulation_time)
                                print(f"  Position {b_index}: Part {self.env.B_info[b_index, 0]}, Tray {self.env.B_info[b_index, 1]}, Finish time: {self.env.B_info[b_index, 2]}, Time left: {time_left}")
                            elif self.env.B_info[b_index, 1] > 0:
                                print(f"  Position {b_index}: No part, Tray {self.env.B_info[b_index, 1]} (ready for recycling)")
                    
                    # Update step count
                    step_count += 1
                
                # Consider the instance successful if we processed at least 90% of parts
                min_parts_threshold = int(total_parts * 0.9)
                if processed_parts >= min_parts_threshold:
                    # Calculate completion time based on the simulation time
                    completion_time = simulation_time
                    if completion_time > 0:
                        total_completion_time += completion_time
                        num_completed_instances += 1
                        print(f"Instance {instance+1} completed: processed {processed_parts}/{total_parts} parts in {completion_time} time units")
                else:
                    print(f"Instance {instance+1} failed: only processed {processed_parts}/{total_parts} parts in {step_count} steps")
                
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
        # Simple greedy scheduler that picks the first valid action
        for action in [0, 1, 2, 3]:
            if action == 0 and env.check_action0_valide():
                return 0
            elif action == 1 and env.check_action1_valide():
                return 1
            elif action == 2 and env.check_action2_valide():
                return 2
            elif action == 3 and env.check_action3_valide():
                return 3
        return -1  # No valid action found
    
    # Test the evaluation
    evaluation = AGVEvaluation()
    result = evaluation.evaluate(simple_scheduler)
    print(f"Evaluation result: {-result}")  # Negative since evaluate returns negative completion time
    