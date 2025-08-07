from __future__ import annotations

import os
import json
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from ...base import Function


class EvolutionPathMemory:
    """
    Evolution Path Memory: records and analyzes successful evolution paths,
    extracts effective evolution patterns, and guides future heuristic generation.
    """
    
    def __init__(self, log_dir: Optional[str] = None, memory_capacity: int = 100):
        """
        Initialize the evolution path memory.
        
        Args:
            log_dir: Directory to save memory data for persistence
            memory_capacity: Maximum number of paths to store in memory
        """
        self._evolution_paths = []  # List of successful evolution paths
        self._pattern_statistics = defaultdict(int)  # Statistics about successful patterns
        self._operator_success_rates = defaultdict(lambda: [0, 0])  # [success_count, total_count]
        self._memory_capacity = memory_capacity
        
        # Set up logging directory
        self._log_dir = log_dir
        if self._log_dir:
            self._memory_dir = os.path.join(self._log_dir, 'evolution_memory')
            os.makedirs(self._memory_dir, exist_ok=True)
    
    def record_evolution_step(self, parent_funcs: List[Function], child_func: Function, 
                              operator_type: str, success: bool = False):
        """
        Record an evolution step including parent(s) and child functions.
        
        Args:
            parent_funcs: List of parent functions used to generate the child
            child_func: The resulting child function
            operator_type: The operator used (e.g., 'e1', 'e2', 'm1', 'm2', 'local_search')
            success: Whether this evolution step resulted in improvement
        """
        try:
            # Record operator success rate
            self._operator_success_rates[operator_type][1] += 1  # Increment total count
            
            # Check if child's score is valid (not infinite or NaN)
            has_valid_score = (child_func.score is not None and 
                              not math.isinf(child_func.score) and 
                              not math.isnan(child_func.score))
            
            # For tasks where lower scores are better (like negative scores), mark success appropriately
            if success and has_valid_score:
                self._operator_success_rates[operator_type][0] += 1  # Increment success count
            
            # For recording paths, we need additional validation to ensure quality
            if not has_valid_score:
                return
                
            # Calculate improvement - filtering valid parent scores
            valid_parent_scores = [func.score for func in parent_funcs if func.score is not None 
                                   and not math.isinf(func.score) and not math.isnan(func.score)]
                                   
            if not valid_parent_scores:
                # If no valid parent scores, we can't determine improvement
                return
                
            parent_best_score = max(valid_parent_scores)
            
            # Determine if this is an improvement 
            # Note: For minimization problems (negative scores), improvement is when child score > parent score
            is_improvement = child_func.score > parent_best_score
            
            # Only record if explicitly successful or detected as an improvement
            if not (success or is_improvement):
                return
                
            # Calculate improvement value
            improvement = child_func.score - parent_best_score
                
            # Create evolution step record
            evolution_step = {
                'timestamp': time.time(),
                'operator': operator_type,
                'parent_algorithms': [func.algorithm for func in parent_funcs],
                'parent_scores': [func.score for func in parent_funcs if func.score is not None],
                'child_algorithm': child_func.algorithm,
                'child_score': child_func.score,
                'improvement': improvement
            }
            
            # Add to evolution paths
            self._evolution_paths.append(evolution_step)
            
            # Limit memory capacity
            if len(self._evolution_paths) > self._memory_capacity:
                self._evolution_paths = sorted(
                    self._evolution_paths, 
                    key=lambda x: x['improvement'], 
                    reverse=True
                )[:self._memory_capacity]
            
            # Extract patterns from this successful evolution
            self._extract_patterns(evolution_step)
            
            # Save to disk if log_dir is provided
            if self._log_dir:
                self._save_memory()
            
            # Log the recorded evolution step
            print(f"Recorded evolution step: {operator_type} operator produced improvement of {improvement:.2f} ({parent_best_score:.2f} -> {child_func.score:.2f})")
                
        except Exception as e:
            print(f"Error recording evolution step: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _extract_patterns(self, evolution_step: Dict):
        """
        Extract patterns from successful evolution steps.
        
        Args:
            evolution_step: The evolution step to analyze
        """
        # Simple pattern: operator type
        self._pattern_statistics[f"operator:{evolution_step['operator']}"] += 1
        
        # Pattern: complexity change (algorithm description length)
        parent_complexity = np.mean([len(algo) for algo in evolution_step['parent_algorithms']])
        child_complexity = len(evolution_step['child_algorithm'])
        
        if child_complexity > parent_complexity * 1.2:
            self._pattern_statistics["pattern:complexity_increase"] += 1
        elif child_complexity < parent_complexity * 0.8:
            self._pattern_statistics["pattern:complexity_decrease"] += 1
        else:
            self._pattern_statistics["pattern:complexity_similar"] += 1
            
        # Other patterns can be extracted here
    
    def _save_memory(self):
        """Save the current memory state to disk."""
        try:
            memory_data = {
                'evolution_paths': self._evolution_paths,
                'pattern_statistics': dict(self._pattern_statistics),
                'operator_success_rates': {k: v for k, v in self._operator_success_rates.items()}
            }
            
            path = os.path.join(self._memory_dir, 'evolution_memory.json')
            with open(path, 'w') as json_file:
                json.dump(memory_data, json_file, indent=4)
            print(f"Evolution memory saved to {path}")
        except Exception as e:
            print(f"Error saving evolution memory: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_memory(self):
        """Load memory from disk if available."""
        if not self._log_dir:
            print("No log directory specified, memory won't be loaded or saved")
            return
            
        path = os.path.join(self._memory_dir, 'evolution_memory.json')
        try:
            if not os.path.exists(path):
                print(f"No evolution memory file found at {path}. Starting with empty memory.")
                return
            
            with open(path, 'r') as json_file:
                memory_data = json.load(json_file)
                
            self._evolution_paths = memory_data.get('evolution_paths', [])
            # Filter out any invalid paths with infinity or NaN scores
            self._evolution_paths = [path for path in self._evolution_paths 
                                    if not math.isinf(path.get('child_score', 0)) 
                                    and not math.isnan(path.get('child_score', 0))
                                    and path.get('improvement', 0) > 0]
                
            self._pattern_statistics = defaultdict(int)
            for k, v in memory_data.get('pattern_statistics', {}).items():
                self._pattern_statistics[k] = v
                
            self._operator_success_rates = defaultdict(lambda: [0, 0])
            for k, v in memory_data.get('operator_success_rates', {}).items():
                self._operator_success_rates[k] = v
                
            print(f"Evolution memory loaded from {path}: {len(self._evolution_paths)} paths")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading evolution memory: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_successful_evolution_paths(self, top_n: int = 5) -> List[Dict]:
        """
        Get the most successful evolution paths.
        
        Args:
            top_n: Number of top paths to return
            
        Returns:
            List of the most successful evolution paths
        """
        sorted_paths = sorted(
            self._evolution_paths, 
            key=lambda x: x['improvement'], 
            reverse=True
        )[:top_n]
        return sorted_paths
    
    def get_operator_success_rates(self) -> Dict[str, float]:
        """
        Get the success rates of different operators.
        
        Returns:
            Dictionary mapping operator names to their success rates
        """
        result = {}
        for operator, counts in self._operator_success_rates.items():
            success_count, total_count = counts
            success_rate = success_count / total_count if total_count > 0 else 0
            result[operator] = success_rate
        return result
    
    def get_most_successful_patterns(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """
        Get the most successful evolution patterns.
        
        Args:
            top_n: Number of top patterns to return
            
        Returns:
            List of (pattern, count) tuples
        """
        sorted_patterns = sorted(
            self._pattern_statistics.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        return sorted_patterns
    
    def get_guidance_prompt(self, task_prompt: str, operator_type: str) -> str:
        """
        Generate a guidance prompt based on successful evolution patterns.
        
        Args:
            task_prompt: The original task description
            operator_type: The operator being used
            
        Returns:
            An enhanced prompt with guidance from memory
        """
        # If we don't have any evolution paths yet, return empty guidance
        if not self._evolution_paths:
            print(f"No evolution paths available yet for {operator_type} operator, using default prompt")
            return ""
        
        # Get top successful paths
        successful_paths = self.get_successful_evolution_paths(top_n=3)
        
        # Generate guidance based on successful patterns
        top_patterns = self.get_most_successful_patterns()
        pattern_guidance = ""
        
        if top_patterns:
            pattern_guidance = "Based on past successful evolution, consider these strategies:\n"
            for pattern, _ in top_patterns:
                if pattern.startswith("pattern:"):
                    if pattern == "pattern:complexity_increase":
                        pattern_guidance += "- Adding more sophisticated logic to the algorithm\n"
                    elif pattern == "pattern:complexity_decrease":
                        pattern_guidance += "- Simplifying the algorithm by removing unnecessary steps\n"
                    elif pattern == "pattern:complexity_similar":
                        pattern_guidance += "- Refining existing logic while maintaining similar complexity\n"
        
        # Add examples from successful paths
        examples = ""
        if successful_paths:
            examples = "Here are examples of successful evolution steps:\n"
            for i, path in enumerate(successful_paths[:2], 1):  # Include up to 2 examples
                examples += f"Example {i}:\n"
                examples += f"- Parent algorithm: {path['parent_algorithms'][0]}\n"
                examples += f"- Improved child: {path['child_algorithm']}\n"
        
        # If we don't have any patterns or examples, return empty guidance
        if not pattern_guidance and not examples:
            print(f"No useful patterns or examples found for {operator_type} operator, using default prompt")
            return ""
        
        # Create the enhanced prompt
        guidance_prompt = f"""{task_prompt}
        
{pattern_guidance}

{examples if examples else ""}
"""
        # Print debug information about the generated prompt
        print(f"Generated guidance prompt for {operator_type} operator using {len(successful_paths)} successful paths")
        
        return guidance_prompt 