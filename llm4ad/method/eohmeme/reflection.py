from __future__ import annotations

import time
import json
import os
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
import re
import random

from ...base import Function
from .evolution_memory import EvolutionPathMemory


class EvolutionReflector:
    """
    A reflection mechanism for EoH that analyzes the evolution process,
    identifies patterns in successful and unsuccessful evolution steps,
    and provides targeted feedback to guide future heuristic generation.
    """
    
    def __init__(self, memory: Optional[EvolutionPathMemory] = None, log_dir: Optional[str] = None, 
                 llm_sampler=None, llm_reflector=None):
        """
        Initialize the evolution reflector.
        
        Args:
            memory: Optional evolution memory instance for additional insights
            log_dir: Directory to save reflection data
            llm_sampler: LLM sampler for generating heuristics (not used for reflection)
            llm_reflector: Dedicated LLM for generating reflections and analysis
        """
        self.memory = memory
        self._log_dir = log_dir
        self._llm_sampler = llm_sampler
        self._llm_reflector = llm_reflector  # Dedicated LLM for reflection
        
        if self._log_dir:
            self._reflection_dir = os.path.join(self._log_dir, 'evolution_reflections')
            os.makedirs(self._reflection_dir, exist_ok=True)
        
        # Track insights and reflections
        self._reflections = []
        self._current_insights = {}
        self._reflection_counter = 0
        
        # Track LLM reflections separately but integrated
        self._llm_reflections = []
        
        # Metrics to track
        self._diversity_history = []
        self._improvement_rate_history = []
        self._stagnation_periods = []
        
        # Track population performance history
        self._best_score_history = []
        self._worst_score_history = []
        self._avg_score_history = []
        
        # Track heuristic features that correlate with performance
        self._feature_correlation = {}
        self._high_performing_features = defaultdict(int)
        self._low_performing_features = defaultdict(int)
        
    def reflect(self, population: List[Function], generation: int) -> Dict[str, Any]:
        """
        Perform reflection on the current state of evolution based on best and worst performing samples.
        
        Args:
            population: Current population of functions
            generation: Current generation number
            
        Returns:
            Dictionary of insights from reflection
        """
        # Don't reflect if too early or no population
        if generation < 1 or not population:
            return {}
            
        # The outer EoH class will control when reflection is called based on reflection_frequency
        # No need for internal period checking here
        self._reflection_counter += 1
        
        # Generate timestamp for this reflection
        timestamp = time.time()
        
        # Debug LLM reflector availability
        has_llm_reflector = self._llm_reflector is not None
        print(f"LLM reflector available for reflection: {has_llm_reflector}")
        
        # Sort population by score
        valid_funcs = [f for f in population if hasattr(f, 'score') and f.score is not None and not np.isnan(f.score) and not np.isinf(f.score)]
        if not valid_funcs:
            print("No valid functions found for reflection")
            return {}
            
        print(f"Found {len(valid_funcs)} valid functions for reflection")
        sorted_pop = sorted(valid_funcs, key=lambda f: f.score, reverse=True)
        
        # Get best and worst performing samples
        best_samples = sorted_pop[:max(1, len(sorted_pop) // 3)]  # Top third
        worst_samples = sorted_pop[-max(1, len(sorted_pop) // 3):]  # Bottom third
        
        # Update performance history
        if sorted_pop:
            self._best_score_history.append(sorted_pop[0].score)
            self._worst_score_history.append(sorted_pop[-1].score)
            self._avg_score_history.append(sum(f.score for f in sorted_pop) / len(sorted_pop))
            
        # Analyze population diversity
        diversity = self._analyze_diversity(sorted_pop)
        
        # Analyze performance differences between best and worst
        perf_diff = self._analyze_performance_difference(best_samples, worst_samples)
        
        # Extract algorithmic features from best and worst samples
        self._extract_algorithmic_features(best_samples, worst_samples)
        
        # Analyze stagnation
        stagnation = self._detect_stagnation(sorted_pop)
        
        # Analyze successful vs unsuccessful patterns from current population
        pattern_insights = self._analyze_population_patterns(best_samples, worst_samples)
        
        # Generate specific reflections based on observations
        reflections = self._generate_reflections(
            population=sorted_pop,
            best_samples=best_samples,
            worst_samples=worst_samples,
            generation=generation,
            diversity=diversity,
            perf_diff=perf_diff,
            stagnation=stagnation,
            pattern_insights=pattern_insights
        )
        
        # Generate LLM reflections if reflector is available - no minimum size requirement initially
        llm_reflections = []
        if self._llm_reflector:
            print(f"Attempting to generate LLM reflections with {len(sorted_pop)} functions")
            llm_reflections = self._generate_llm_reflections(sorted_pop)
            print(f"Generated {len(llm_reflections)} LLM reflections")
        
        # Combine empirical and LLM reflections
        all_reflections = reflections + llm_reflections
        
        # Combine all insights
        insights = {
            "timestamp": timestamp,
            "generation": generation,
            "diversity": diversity,
            "performance_difference": perf_diff,
            "stagnation": stagnation,
            "pattern_insights": pattern_insights,
            "reflections": reflections,  # Empirical reflections
            "llm_reflections": llm_reflections,  # LLM reflections
            "all_reflections": all_reflections,  # Combined reflections
            "best_score": sorted_pop[0].score if sorted_pop else None,
            "worst_score": sorted_pop[-1].score if sorted_pop else None,
            "avg_score": sum(f.score for f in sorted_pop) / len(sorted_pop) if sorted_pop else None
        }
        
        # Save reflection
        self._reflections.append(insights)
        self._current_insights = insights
        
        # Save to disk
        if self._log_dir:
            self._save_reflection(insights)
            
        return insights
    
    def _generate_llm_reflections(self, population: List[Function]) -> List[str]:
        """
        Generate reflections using an LLM on the current population.
        
        Args:
            population: Current population of functions, sorted by performance
            
        Returns:
            List of LLM-generated reflections
        """
        if not self._llm_reflector:
            print("No LLM reflector available for reflection generation")
            return []
            
        if not population:
            print("Empty population, can't generate LLM reflections")
            return []
            
        # Test reflector before proceeding
        if not self._ensure_reflector_works():
            print("Skipping LLM reflection generation due to reflector issues")
            return []
            
        try:
            # Select a subset of the population for reflection - at least 3, but all if fewer
            max_sample_size = min(5, len(population))
            if len(population) <= 5:
                # Use all functions if 5 or fewer
                population_sample = population
                print(f"Using all {len(population_sample)} functions for LLM reflection")
            else:
                # Ensure diversity by selecting from top, middle and bottom
                top = population[:max(1, max_sample_size // 3)]
                middle_start = len(population) // 2 - max(1, max_sample_size // 3) // 2
                middle = population[middle_start:middle_start + max(1, max_sample_size // 3)]
                bottom = population[-max(1, max_sample_size // 3):]
                
                # Combine samples from different parts of the population
                population_sample = []
                population_sample.extend(top)
                population_sample.extend(middle)
                population_sample.extend(bottom)
                population_sample = population_sample[:max_sample_size]  # Limit to max_sample_size
                print(f"Selected {len(population_sample)} diverse functions for LLM reflection")
                
            # Create the reflection prompt
            prompt = self._create_llm_reflection_prompt(population_sample)
            print("Created LLM reflection prompt")
            print(f"Prompt length: {len(prompt)} characters")
            
            # Get reflections from LLM
            print("Calling LLM for reflection generation...")
            thought, _ = self._llm_reflector.get_thought_and_function(prompt)
            
            if not thought:
                print("LLM reflection returned empty thought")
                return []
                
            print(f"Generated LLM reflection text ({len(thought)} chars)")
            
            # Parse reflections from the LLM output
            reflections = []
            current_reflection = ""
            
            # Extract numbered reflections
            for line in thought.split('\n'):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Try to detect numbered items with regex
                match = re.match(r'^(\d+)[\.:\)]\s*(.*)', line)
                if match:
                    # If we have a previous reflection, save it
                    if current_reflection:
                        reflections.append(current_reflection.strip())
                        
                    # Start a new reflection
                    current_reflection = match.group(2)
                else:
                    # Continuation of previous reflection
                    current_reflection += " " + line
                    
            # Add the last reflection if any
            if current_reflection:
                reflections.append(current_reflection.strip())
                
            # If regex failed to find numbered items, try a different approach
            if not reflections:
                print("Regex approach failed to extract reflections, trying alternative parsing")
                # Split by common separators
                potential_reflections = re.split(r'(?:\n\n|\n\*|\n-)', thought)
                reflections = [r.strip() for r in potential_reflections if r.strip()]
                
            print(f"Extracted {len(reflections)} reflections from LLM output")
            
            # Store LLM reflections
            self._llm_reflections = reflections
            
            # Prefix LLM reflections to distinguish them
            prefixed_reflections = [f"[LLM] {reflection}" for reflection in reflections]
            
            return prefixed_reflections
            
        except Exception as e:
            print(f"Error generating LLM reflections: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_llm_reflection_prompt(self, population_sample: List[Function]) -> str:
        """
        Create a prompt for the LLM to reflect on the current population.
        
        Args:
            population_sample: Sample of functions to reflect on
            
        Returns:
            Prompt for the LLM
        """
        prompt = "Analyze the following algorithms and provide insights to improve future evolution:\n\n"
        
        # Include summary of current evolution state
        if self._current_insights:
            diversity = self._current_insights.get("diversity", {}).get("level", "unknown")
            stagnation = self._current_insights.get("stagnation", {}).get("status", "unknown")
            
            prompt += f"Current population diversity: {diversity}\n"
            prompt += f"Evolution status: {stagnation}\n\n"
        
        # Include algorithms from population sample
        for i, func in enumerate(population_sample, 1):
            prompt += f"Algorithm {i} (score: {func.score}):\n"
            prompt += f"{func.algorithm}\n\n"
        
        # Request specific types of reflections
        prompt += "Please provide 3-5 numbered insights addressing:\n"
        prompt += "1. Common patterns in high-performing solutions\n"
        prompt += "2. Potential improvements to existing algorithms\n"
        prompt += "3. Novel approaches that haven't been tried yet\n"
        prompt += "4. Reasons for observed performance differences\n"
        prompt += "Your reflections will guide the next generation of algorithms."
        
        return prompt
    
    def _ensure_reflector_works(self) -> bool:
        """
        Check if the reflector is working correctly by testing a simple prompt.
        If not, try to fix or report the issue.
        
        Returns:
            True if the reflector is working, False otherwise
        """
        if not self._llm_reflector:
            print("No LLM reflector available")
            return False
            
        try:
            # Create a very simple test prompt
            test_prompt = "Please provide one sentence describing optimization algorithms."
            
            # Try to get a response
            print("Testing reflector with a simple prompt...")
            thought, _ = self._llm_reflector.get_thought_and_function(test_prompt)
            
            if thought and isinstance(thought, str) and len(thought) > 10:
                print("Reflector test successful")
                return True
            else:
                print("Reflector test failed: empty or invalid response")
                return False
                
        except Exception as e:
            print(f"Reflector test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _analyze_diversity(self, population: List[Function]) -> Dict[str, Any]:
        """Analyze diversity of the current population."""
        if not population or len(population) < 2:
            return {"level": "unknown", "score": 0}
            
        # Compare algorithm descriptions for semantic diversity
        descriptions = [f.algorithm for f in population if hasattr(f, 'algorithm')]
        if not descriptions:
            return {"level": "unknown", "score": 0}
            
        # Simple measure: average character-wise difference between algorithms
        total_diff = 0
        count = 0
        
        for i in range(len(descriptions)):
            for j in range(i+1, len(descriptions)):
                # Normalized edit distance would be better, but this is a simple proxy
                len_diff = abs(len(descriptions[i]) - len(descriptions[j]))
                max_len = max(len(descriptions[i]), len(descriptions[j]))
                if max_len > 0:
                    total_diff += len_diff / max_len
                count += 1
                
        avg_diff = total_diff / count if count > 0 else 0
        self._diversity_history.append(avg_diff)
        
        # Classify diversity level
        if avg_diff < 0.2:
            level = "low"
        elif avg_diff < 0.5:
            level = "moderate"
        else:
            level = "high"
            
        return {"level": level, "score": avg_diff}
    
    def _analyze_performance_difference(self, best_samples: List[Function], worst_samples: List[Function]) -> Dict[str, Any]:
        """Analyze performance differences between best and worst performing algorithms."""
        if not best_samples or not worst_samples:
            return {"gap": "unknown", "significance": "unknown"}
            
        # Calculate performance gap
        best_avg = sum(f.score for f in best_samples) / len(best_samples)
        worst_avg = sum(f.score for f in worst_samples) / len(worst_samples)
        
        performance_gap = best_avg - worst_avg
        relative_gap = performance_gap / abs(worst_avg) if worst_avg != 0 else float('inf')
        
        # Classify significance of the gap
        if relative_gap < 0.1:
            significance = "minimal"
        elif relative_gap < 0.5:
            significance = "moderate"
        else:
            significance = "substantial"
            
        return {
            "gap": performance_gap,
            "relative_gap": relative_gap,
            "significance": significance,
            "best_avg": best_avg,
            "worst_avg": worst_avg
        }
    
    def _extract_algorithmic_features(self, best_samples: List[Function], worst_samples: List[Function]):
        """Extract algorithmic features that correlate with performance."""
        # Look for common patterns in high-performing algorithms
        for func in best_samples:
            # Extract features from algorithm description
            if hasattr(func, 'algorithm') and func.algorithm:
                features = self._extract_features_from_text(func.algorithm)
                for feature in features:
                    self._high_performing_features[feature] += 1
            
            # Extract features from code
            if hasattr(func, 'body') and func.body:
                code_features = self._extract_features_from_code(func.body)
                for feature in code_features:
                    self._high_performing_features[feature] += 1
                    
        # Look for common patterns in low-performing algorithms
        for func in worst_samples:
            # Extract features from algorithm description
            if hasattr(func, 'algorithm') and func.algorithm:
                features = self._extract_features_from_text(func.algorithm)
                for feature in features:
                    self._low_performing_features[feature] += 1
            
            # Extract features from code
            if hasattr(func, 'body') and func.body:
                code_features = self._extract_features_from_code(func.body)
                for feature in code_features:
                    self._low_performing_features[feature] += 1
                    
    def _extract_features_from_text(self, text: str) -> List[str]:
        """Extract key features from algorithm description text."""
        features = []
        
        # Look for keywords that might indicate algorithmic approaches
        key_phrases = [
            "priority", "greedy", "heuristic", "optimization", "search", 
            "sorting", "scheduling", "allocation", "assignment", "balancing",
            "weighted", "adaptive", "dynamic", "static", "multi-objective",
            "constraint", "validation", "checking", "efficiency", "optimal",
            "trade-off", "balance", "first", "best", "worst", "average"
        ]
        
        for phrase in key_phrases:
            if phrase.lower() in text.lower():
                features.append(f"concept:{phrase}")
                
        # Look for decision-making patterns
        if "if" in text.lower() and "then" in text.lower():
            features.append("pattern:conditional_rules")
        
        if "priority" in text.lower() or "prioritize" in text.lower():
            features.append("pattern:prioritization")
            
        if "balance" in text.lower() or "trade-off" in text.lower():
            features.append("pattern:balancing")
            
        return features
    
    def _extract_features_from_code(self, code: str) -> List[str]:
        """Extract key features from function body code."""
        features = []
        
        # Check for various code patterns
        if code.count("if ") > 3:
            features.append("code:multiple_conditions")
            
        if "sort" in code:
            features.append("code:sorting")
            
        if "max(" in code or "min(" in code:
            features.append("code:min_max_optimization")
            
        if "return -1" in code or "return None" in code:
            features.append("code:early_termination")
            
        # Count loops
        for_count = len(re.findall(r'\bfor\b', code))
        while_count = len(re.findall(r'\bwhile\b', code))
        
        if for_count > 0:
            features.append(f"code:for_loops_{for_count}")
            
        if while_count > 0:
            features.append(f"code:while_loops_{while_count}")
            
        # Check for specific data structures
        if "dict" in code or "{" in code:
            features.append("code:dictionary_usage")
            
        if "list" in code or "[" in code:
            features.append("code:list_usage")
            
        if "set" in code:
            features.append("code:set_usage")
            
        # Check for numpy usage
        if "np." in code:
            features.append("code:numpy_usage")
            
        return features
    
    def _detect_stagnation(self, population: List[Function]) -> Dict[str, Any]:
        """Detect periods of stagnation in evolution."""
        if not population:
            return {"status": "unknown", "duration": 0}
            
        # Check if best score has improved recently
        best_scores = [f.score for f in population if hasattr(f, 'score') and f.score is not None]
        if not best_scores:
            return {"status": "unknown", "duration": 0}
            
        current_best = max(best_scores)
        
        # Compare with recent best scores in reflections
        past_bests = []
        for reflection in self._reflections[-3:]:
            if "best_score" in reflection:
                past_bests.append(reflection["best_score"])
        
        # Determine if stagnating
        stagnating = False
        if past_bests and abs(current_best - past_bests[0]) < 1e-6:
            stagnating = True
            
        # Update stagnation periods
        if stagnating:
            if self._stagnation_periods and self._stagnation_periods[-1]["ongoing"]:
                self._stagnation_periods[-1]["duration"] += 1
            else:
                self._stagnation_periods.append({
                    "start_generation": self._reflection_counter,
                    "duration": 1,
                    "ongoing": True,
                    "best_score": current_best
                })
        elif self._stagnation_periods and self._stagnation_periods[-1]["ongoing"]:
            self._stagnation_periods[-1]["ongoing"] = False
            
        # Get current stagnation status
        current_duration = 0
        if self._stagnation_periods and self._stagnation_periods[-1]["ongoing"]:
            current_duration = self._stagnation_periods[-1]["duration"]
            
        return {
            "status": "stagnating" if stagnating else "improving",
            "duration": current_duration,
            "best_score": current_best
        }
    
    def _analyze_population_patterns(self, best_samples: List[Function], worst_samples: List[Function]) -> Dict[str, Any]:
        """Analyze patterns that distinguish best from worst performing samples."""
        if not best_samples or not worst_samples:
            return {}
            
        # Find features overrepresented in best samples
        effective_features = []
        for feature, count in self._high_performing_features.items():
            worst_count = self._low_performing_features.get(feature, 0)
            if count > worst_count * 2 and count >= 2:  # At least twice as common and appears at least twice
                effective_features.append((feature, count))
                
        # Find features overrepresented in worst samples
        ineffective_features = []
        for feature, count in self._low_performing_features.items():
            best_count = self._high_performing_features.get(feature, 0)
            if count > best_count * 2 and count >= 2:  # At least twice as common and appears at least twice
                ineffective_features.append((feature, count))
                
        # Sort by frequency
        effective_features.sort(key=lambda x: x[1], reverse=True)
        ineffective_features.sort(key=lambda x: x[1], reverse=True)
        
        # Get operator information if memory is available
        top_operators = []
        if self.memory:
            memory_patterns = self._analyze_evolution_patterns_from_memory()
            top_operators = memory_patterns.get("top_operators", [])
            
        return {
            "effective_features": effective_features[:5],  # Top 5 effective features
            "ineffective_features": ineffective_features[:5],  # Top 5 ineffective features
            "top_operators": top_operators
        }
        
    def _analyze_evolution_patterns_from_memory(self) -> Dict[str, Any]:
        """Analyze patterns in evolution memory (if available)."""
        if not self.memory or not self.memory._evolution_paths:
            return {}
            
        # Get most successful patterns
        successful_patterns = self.memory.get_most_successful_patterns(top_n=5)
        
        # Look at characteristics of successful evolution paths
        successful_paths = self.memory.get_successful_evolution_paths(top_n=5)
        
        # Count most frequent operators in successful paths
        op_counts = defaultdict(int)
        for path in successful_paths:
            op_counts[path.get("operator", "unknown")] += 1
            
        # Identify operators that appear to be most successful
        sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
        top_operators = [op for op, _ in sorted_ops[:2]] if sorted_ops else []
        
        return {
            "successful_patterns": successful_patterns,
            "top_operators": top_operators
        }
        
    def _generate_reflections(self, **kwargs) -> List[str]:
        """Generate specific reflections based on best and worst samples."""
        reflections = []
        
        # Get key data from kwargs
        population = kwargs.get("population", [])
        best_samples = kwargs.get("best_samples", [])
        worst_samples = kwargs.get("worst_samples", [])
        diversity = kwargs.get("diversity", {})
        perf_diff = kwargs.get("perf_diff", {})
        stagnation = kwargs.get("stagnation", {})
        pattern_insights = kwargs.get("pattern_insights", {})
        
        # Reflect on diversity
        if diversity.get("level") == "low":
            reflections.append(
                "The population shows low diversity. Consider using more exploration operators or "
                "increasing mutation rates to explore different regions of the solution space."
            )
        
        # Reflect on performance differences
        if perf_diff.get("significance") == "minimal":
            reflections.append(
                "The performance gap between best and worst algorithms is minimal. This suggests "
                "the current approaches may be reaching a local optimum. Consider more radical "
                "exploration to find potentially better strategies."
            )
        elif perf_diff.get("significance") == "substantial":
            reflections.append(
                "There is a substantial performance gap between best and worst algorithms. Focus on "
                "understanding what makes the top performers effective and incorporate those patterns."
            )
        
        # Reflect on stagnation
        if stagnation.get("status") == "stagnating" and stagnation.get("duration", 0) >= 2:
            reflections.append(
                "Evolution appears to be stagnating. Consider introducing new exploration strategies, "
                "applying more aggressive local search, or temporarily increasing mutation intensity."
            )
        
        # Reflect on effective features
        effective_features = pattern_insights.get("effective_features", [])
        if effective_features:
            # Extract feature names for readability
            feature_names = [f.split(':', 1)[1] if ':' in f else f for f, _ in effective_features[:3]]
            reflections.append(
                f"The most effective algorithmic features include: {', '.join(feature_names)}. "
                f"Consider emphasizing these elements in future algorithm generation."
            )
        
        # Reflect on ineffective features
        ineffective_features = pattern_insights.get("ineffective_features", [])
        if ineffective_features:
            # Extract feature names for readability
            feature_names = [f.split(':', 1)[1] if ':' in f else f for f, _ in ineffective_features[:2]]
            reflections.append(
                f"Features like {', '.join(feature_names)} appear in low-performing algorithms. "
                f"Consider avoiding or redesigning these aspects."
            )
        
        # Reflect on successful operators if memory is available
        top_operators = pattern_insights.get("top_operators", [])
        if top_operators:
            reflections.append(
                f"Operators {', '.join(top_operators)} have been most effective in generating improvements. "
                f"Consider increasing their usage."
            )
        
        # Analyze best algorithm if available
        if best_samples:
            best_algo = best_samples[0]
            if hasattr(best_algo, 'algorithm') and best_algo.algorithm:
                # Look for specific strategies in the best algorithm
                strategy_detected = False
                
                if any(kw in best_algo.algorithm.lower() for kw in ["priorit", "rank"]):
                    reflections.append(
                        "The best performing algorithm uses prioritization strategies. "
                        "Consider incorporating priority-based decision making in new algorithms."
                    )
                    strategy_detected = True
                
                elif any(kw in best_algo.algorithm.lower() for kw in ["balance", "trade-off", "allocat"]):
                    reflections.append(
                        "The best performing algorithm focuses on balancing resources or trade-offs. "
                        "Consider explicit resource allocation strategies in new algorithms."
                    )
                    strategy_detected = True
                
                if not strategy_detected and len(best_algo.algorithm) > 20:
                    # Generic reflection on best algorithm
                    reflections.append(
                        "Study the top-performing algorithm's approach and consider how its key "
                        "strategies could be enhanced or combined with other techniques."
                    )
        
        return reflections
    
    def _save_reflection(self, reflection: Dict[str, Any]):
        """Save a reflection to disk."""
        if not self._log_dir:
            return
            
        try:
            filename = f"reflection_{self._reflection_counter}.json"
            path = os.path.join(self._reflection_dir, filename)
            
            with open(path, 'w') as f:
                json.dump(reflection, f, indent=2)
                
        except Exception as e:
            print(f"Error saving reflection: {str(e)}")
    
    def get_reflection_prompt(self, task_prompt: str, operator_type: str) -> str:
        """
        Generate a reflection-enhanced prompt for a specific operator.
        
        Args:
            task_prompt: The original task description
            operator_type: The operator being used (e.g., 'e1', 'e2', 'm1', 'm2')
            
        Returns:
            An enhanced prompt incorporating reflections
        """
        if not self._current_insights:
            return ""
            
        # Create a more explicit context for evolution guidance
        reflection_prompt = f"ORIGINAL TASK DESCRIPTION:\n{task_prompt}\n\n"
        reflection_prompt += f"EVOLUTION CONTEXT AND GUIDANCE FOR {operator_type.upper()} OPERATOR:\n"
        
        # Use all LLM reflections and exclude empirical ones
        llm_reflections = self._current_insights.get("llm_reflections", [])
        
        if llm_reflections:
            reflection_prompt += "The following insights from LLM analysis of current algorithms should explicitly guide your next solution:\n\n"
            # Include all LLM reflections without limiting
            for i, reflection in enumerate(llm_reflections, 1):
                # Remove the "[LLM]" prefix when displaying
                clean_reflection = reflection.replace("[LLM] ", "") if reflection.startswith("[LLM]") else reflection
                reflection_prompt += f"Insight {i}: {clean_reflection}\n"
            reflection_prompt += "\n"
        
        # Add operator-specific guidance
        reflection_prompt += "SPECIFIC GUIDANCE FOR THIS EVOLUTION STEP:\n"
        
        success_rates = {}
        if self.memory:
            success_rates = self._current_insights.get("improvement_rates", {}).get("by_operator", {})
        op_rate = success_rates.get(operator_type, 0)
        
        if operator_type in self._current_insights.get("pattern_insights", {}).get("top_operators", []):
            reflection_prompt += f"• The {operator_type} operator has been particularly effective. "
            reflection_prompt += "You should exploit the strategies that have worked well in previous iterations.\n\n"
        elif op_rate < 0.2 and op_rate > 0:  # Low success rate (if we have memory data)
            reflection_prompt += f"• The {operator_type} operator has had limited success. "
            reflection_prompt += "You must try a different approach than what has been tried before.\n\n"
        
        # Add diversity guidance
        diversity = self._current_insights.get("diversity", {})
        if diversity.get("level") == "low":
            reflection_prompt += "• The current population lacks diversity. You must create a solution "
            reflection_prompt += "that is significantly different from existing approaches.\n"
        elif diversity.get("level") == "high":
            reflection_prompt += "• The current population has good diversity. You should focus on refining "
            reflection_prompt += "existing promising approaches rather than creating entirely new ones.\n"
        
        # Add performance gap guidance
        perf_diff = self._current_insights.get("performance_difference", {})
        if perf_diff.get("significance") == "substantial":
            reflection_prompt += "• There is a substantial performance gap between best and worst algorithms. "
            reflection_prompt += "Focus on understanding what makes the top performers effective and incorporate those elements.\n"
        
        # Add stagnation guidance
        stagnation = self._current_insights.get("stagnation", {})
        if stagnation.get("status") == "stagnating":
            reflection_prompt += "• The evolution process appears to be stagnating. You must try a novel approach "
            reflection_prompt += "or a radical modification to break through the current plateau.\n"
        
        # Add guidance on effective features
        effective_features = self._current_insights.get("pattern_insights", {}).get("effective_features", [])
        if effective_features:
            feature_names = [f.split(':', 1)[1] if ':' in f else f for f, _ in effective_features[:3]]
            reflection_prompt += f"• The most effective elements you should incorporate are: {', '.join(feature_names)}.\n"
        
        # Add final instruction to encourage explicit use of insights
        reflection_prompt += "\nYou MUST consider all the insights and guidance above when generating your solution.\n"
        
        return reflection_prompt 