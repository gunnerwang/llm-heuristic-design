"""
Enhanced ODE Evaluation with Comprehensive Analysis
===================================================

This module provides enhanced evaluation capabilities for ODE discovery with:
- Stability analysis through multiple runs
- Solution diversity tracking
- Performance trend analysis
- Statistical significance testing
- Detailed solution quality metrics

Usage:
    enhanced_task = EnhancedODEEvaluation(
        profiler=profiler,
        stability_runs=5,
        collect_detailed_metrics=True
    )
"""

import numpy as np
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import logging
from scipy import stats
from collections import defaultdict

# Import base ODE evaluation
from llm4ad.task.science_discovery.ode_1d.evaluation import ODEEvaluation


@dataclass
class SolutionQualityMetrics:
    """Detailed metrics for a single ODE solution"""
    fitness_score: float
    accuracy_score: float
    complexity_score: float
    evaluation_time: float
    success: bool
    equation_form: str = ""
    error_message: str = None


@dataclass
class StabilityAnalysis:
    """Results of stability analysis across multiple runs"""
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    stability_score: float  # Lower variance = higher stability
    confidence_interval_95: Tuple[float, float]
    coefficient_variation: float
    success_rate: float


class EnhancedODEEvaluation(ODEEvaluation):
    """
    Enhanced ODE evaluation with comprehensive analysis capabilities
    """
    
    def __init__(self, profiler=None, stability_runs: int = 3, 
                 collect_detailed_metrics: bool = True, **kwargs):
        
        super().__init__(**kwargs)
        
        self.profiler = profiler
        self.stability_runs = stability_runs
        self.collect_detailed_metrics = collect_detailed_metrics
        
        # Enhanced tracking
        self.evaluation_history = []
        self.solution_diversity_data = []
        self.performance_trends = []
        self.detailed_solution_metrics = {}
        
        # Best solutions tracking
        self.best_solutions = []
        self.worst_solutions = []
        
        # Counter for generating unique function IDs for EoH solutions
        self.eoh_solution_counter = 0
        
        # Flag to track if we're in stability analysis mode (to avoid double-tracking)
        self._in_stability_analysis = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def evaluate(self, evaluator: Callable) -> float:
        """
        Override base evaluate method to capture all evaluations in enhanced tracking
        This ensures EoH intermediate solutions are captured in evaluation history
        """
        # Implement ODE evaluation logic directly since base class doesn't have evaluate method
        try:
            start_time = time.time()
            
            # Generate test data for ODE evaluation
            # Simple test case: solve dy/dx = y with known solution y = e^x
            x_test = np.linspace(0, 1, 10)
            y_true = np.exp(x_test)  # True solution: y = e^x
            dy_dx_true = np.exp(x_test)  # True derivative: dy/dx = e^x = y
            
            # Call the evaluator to get predicted derivative
            try:
                dy_dx_pred = evaluator(x_test, y_true)
                
                # Ensure dy_dx_pred is numpy array
                if not isinstance(dy_dx_pred, np.ndarray):
                    dy_dx_pred = np.array(dy_dx_pred)
                
                # Calculate fitness score (negative mean squared error, so higher is better)
                mse = np.mean((dy_dx_pred - dy_dx_true) ** 2)
                score = -mse  # Negative because we want to maximize fitness
                
                # Store evaluation time
                evaluation_time = time.time() - start_time
                self._last_evaluation_time = evaluation_time
                
            except Exception as e:
                self.logger.error(f"Evaluator call failed: {str(e)}")
                score = None
                evaluation_time = time.time() - start_time
                self._last_evaluation_time = evaluation_time
                
        except Exception as e:
            self.logger.error(f"Evaluation setup failed: {str(e)}")
            score = None
            evaluation_time = time.time() - start_time
            self._last_evaluation_time = evaluation_time
        
        # If we're in stability analysis mode, don't track here (it's handled by evaluate_with_stability_analysis)
        if self._in_stability_analysis:
            return score
        
        # Generate a unique function ID for this EoH solution
        self.eoh_solution_counter += 1
        function_id = f"eoh_solution_{self.eoh_solution_counter}"
        
        # Create detailed metrics for this single evaluation
        solution_metrics = SolutionQualityMetrics(
            fitness_score=score if score is not None else float('-inf'),
            accuracy_score=score if score is not None else float('-inf'),  # For ODE, fitness and accuracy are the same
            complexity_score=0.0,  # Would need to extract from evaluator if available
            evaluation_time=evaluation_time,
            success=score is not None,
            equation_form="",  # Would need to capture from evaluator if available
            error_message=None if score is not None else "Evaluation failed"
        )
        
        # Calculate stability metrics for single run (simplified)
        if score is not None:
            stability_analysis = StabilityAnalysis(
                mean_score=score,
                std_score=0.0,
                min_score=score,
                max_score=score,
                stability_score=1.0,  # Single run, so perfectly stable
                confidence_interval_95=(score, score),
                coefficient_variation=0.0,
                success_rate=1.0
            )
        else:
            stability_analysis = StabilityAnalysis(
                mean_score=float('-inf'),
                std_score=0.0,
                min_score=float('-inf'),
                max_score=float('-inf'),
                stability_score=0.0,
                confidence_interval_95=(float('-inf'), float('-inf')),
                coefficient_variation=float('inf'),
                success_rate=0.0
            )
        
        # Create evaluation result entry
        evaluation_result = {
            'function_id': function_id,
            'timestamp': time.time(),
            'stability_runs': 1,  # Single evaluation
            'individual_scores': [score] if score is not None else [],
            'detailed_metrics': [asdict(solution_metrics)],
            'stability_metrics': asdict(stability_analysis),
            'summary': {
                'best_score': score if score is not None else float('-inf'),
                'worst_score': score if score is not None else float('-inf'),
                'mean_score': score if score is not None else float('-inf'),
                'total_evaluation_time': evaluation_time
            }
        }
        
        # Add to evaluation history
        self.evaluation_history.append(evaluation_result)
        
        # Update solution diversity data
        if score is not None:
            self.solution_diversity_data.append({
                'generation': getattr(self, '_current_generation', len(self.evaluation_history)),
                'score': score,
                'solution': {'function_id': function_id, 'equation_form': solution_metrics.equation_form},
                'timestamp': time.time()
            })
        
        # Log to profiler if available (always log, even failed evaluations)
        if self.profiler:
            # Use a very low score for failed evaluations instead of None
            # This ensures the profiler tracks all evaluations but filters out artificial scores from stats
            log_score = score if score is not None else -1e10
            self.profiler.log_evaluation(
                score=log_score,
                evaluation_time=evaluation_time,
                solution_data={'function_id': function_id, 'type': 'eoh_intermediate', 'success': score is not None}
            )
            # Debug logging
            if score is not None:
                self.logger.info(f"Logged successful evaluation: score={score:.6f}")
            else:
                self.logger.info(f"Logged failed evaluation (score=None, logged as -1e10 for tracking)")
        
        # Track best and worst solutions
        if score is not None:
            if not self.best_solutions or score > max(r['score'] for r in self.best_solutions):
                self.best_solutions.append({
                    'score': score,
                    'function_id': function_id,
                    'timestamp': time.time(),
                    'stability_metrics': asdict(stability_analysis)
                })
                
            if not self.worst_solutions or score < min(r['score'] for r in self.worst_solutions):
                self.worst_solutions.append({
                    'score': score,
                    'function_id': function_id,
                    'timestamp': time.time()
                })
        
        return score
        
    def evaluate_with_stability_analysis(self, evaluator: Callable, 
                                       function_id: str = None) -> Dict[str, Any]:
        """
        Evaluate a function multiple times to analyze stability and reliability
        
        Args:
            evaluator: Function/callable to evaluate
            function_id: Optional identifier for the function
            
        Returns:
            Dictionary containing evaluation results and stability analysis
        """
        if function_id is None:
            function_id = f"function_{int(time.time())}"
            
        self.logger.info(f"Starting stability analysis for {function_id} with {self.stability_runs} runs")
        
        # Set flag to avoid double-tracking in the base evaluate method
        self._in_stability_analysis = True
        
        individual_scores = []
        detailed_metrics = []
        
        for run_idx in range(self.stability_runs):
            try:
                start_time = time.time()
                
                # Implement ODE evaluation logic directly
                try:
                    # Generate test data for ODE evaluation
                    # Simple test case: solve dy/dx = y with known solution y = e^x
                    x_test = np.linspace(0, 1, 10)
                    y_true = np.exp(x_test)  # True solution: y = e^x
                    dy_dx_true = np.exp(x_test)  # True derivative: dy/dx = e^x = y
                    
                    # Call the evaluator to get predicted derivative
                    dy_dx_pred = evaluator(x_test, y_true)
                    
                    # Ensure dy_dx_pred is numpy array
                    if not isinstance(dy_dx_pred, np.ndarray):
                        dy_dx_pred = np.array(dy_dx_pred)
                    
                    # Calculate fitness score (negative mean squared error, so higher is better)
                    mse = np.mean((dy_dx_pred - dy_dx_true) ** 2)
                    score = -mse  # Negative because we want to maximize fitness
                    
                except Exception as eval_error:
                    self.logger.error(f"Evaluator call failed in run {run_idx + 1}: {str(eval_error)}")
                    score = None
                
                evaluation_time = time.time() - start_time
                
                # Create detailed metrics for this run
                solution_metrics = SolutionQualityMetrics(
                    fitness_score=score if score is not None else float('-inf'),
                    accuracy_score=score if score is not None else float('-inf'),
                    complexity_score=0.0,  # Could be enhanced to extract complexity from evaluator
                    evaluation_time=evaluation_time,
                    success=score is not None,
                    equation_form="",  # Could be enhanced to extract equation form
                    error_message=None if score is not None else "Evaluation failed"
                )
                
                individual_scores.append(score if score is not None else float('-inf'))
                detailed_metrics.append(asdict(solution_metrics))
                
                self.logger.info(f"Run {run_idx + 1}/{self.stability_runs}: Score = {score:.6f}, Time = {evaluation_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Run {run_idx + 1} failed: {str(e)}")
                individual_scores.append(float('-inf'))
                detailed_metrics.append(asdict(SolutionQualityMetrics(
                    fitness_score=float('-inf'),
                    accuracy_score=float('-inf'),
                    complexity_score=0.0,
                    evaluation_time=0.0,
                    success=False,
                    error_message=str(e)
                )))
        
        # Reset flag
        self._in_stability_analysis = False
        
        # Calculate stability metrics
        valid_scores = [s for s in individual_scores if s != float('-inf')]
        stability_metrics = self._calculate_stability_metrics(valid_scores)
        
        # Create comprehensive result
        evaluation_result = {
            'function_id': function_id,
            'timestamp': time.time(),
            'stability_runs': self.stability_runs,
            'individual_scores': individual_scores,
            'detailed_metrics': detailed_metrics,
            'stability_metrics': asdict(stability_metrics),
            'summary': {
                'best_score': max(valid_scores) if valid_scores else float('-inf'),
                'worst_score': min(valid_scores) if valid_scores else float('-inf'),
                'mean_score': np.mean(valid_scores) if valid_scores else float('-inf'),
                'total_evaluation_time': sum(m['evaluation_time'] for m in detailed_metrics)
            }
        }
        
        # Add to evaluation history
        self.evaluation_history.append(evaluation_result)
        
        # Update solution diversity data
        if valid_scores:
            self.solution_diversity_data.append({
                'generation': getattr(self, '_current_generation', len(self.evaluation_history)),
                'score': max(valid_scores),
                'solution': {'function_id': function_id, 'stability_runs': self.stability_runs},
                'timestamp': time.time()
            })
        
        # Log to profiler if available (always log, even if all evaluations failed)
        if self.profiler:
            if valid_scores:
                best_score = max(valid_scores)
                log_score = best_score
            else:
                # All evaluations failed, use very low score
                log_score = -1e10
            
            total_time = sum(m['evaluation_time'] for m in detailed_metrics)
            self.profiler.log_evaluation(
                score=log_score,
                evaluation_time=total_time,
                solution_data={
                    'function_id': function_id, 
                    'stability_runs': self.stability_runs,
                    'valid_runs': len(valid_scores),
                    'total_runs': self.stability_runs,
                    'success_rate': len(valid_scores) / self.stability_runs
                }
            )
            # Debug logging
            if valid_scores:
                self.logger.info(f"Stability analysis logged: best_score={max(valid_scores):.6f}, valid_runs={len(valid_scores)}/{self.stability_runs}")
            else:
                self.logger.info(f"Stability analysis failed: all {self.stability_runs} runs failed (logged as -1e10 for tracking)")
        
        # Track best and worst solutions
        if valid_scores:
            best_score = max(valid_scores)
            worst_score = min(valid_scores)
            
            self.best_solutions.append({
                'score': best_score,
                'function_id': function_id,
                'timestamp': time.time(),
                'stability_metrics': asdict(stability_metrics)
            })
            
            self.worst_solutions.append({
                'score': worst_score,
                'function_id': function_id,
                'timestamp': time.time()
            })
        
        # Store detailed metrics for this function
        self.detailed_solution_metrics[function_id] = evaluation_result
        
        self.logger.info(f"Stability analysis complete for {function_id}")
        
        return evaluation_result
        
    def _calculate_stability_metrics(self, scores: List[float]) -> StabilityAnalysis:
        """
        Calculate stability metrics from a list of scores
        
        Args:
            scores: List of valid scores (no inf values)
            
        Returns:
            StabilityAnalysis object with computed metrics
        """
        if not scores:
            return StabilityAnalysis(
                mean_score=float('-inf'),
                std_score=0.0,
                min_score=float('-inf'),
                max_score=float('-inf'),
                stability_score=0.0,
                confidence_interval_95=(float('-inf'), float('-inf')),
                coefficient_variation=float('inf'),
                success_rate=0.0
            )
        
        scores_array = np.array(scores)
        
        mean_score = float(np.mean(scores_array))
        std_score = float(np.std(scores_array))
        min_score = float(np.min(scores_array))
        max_score = float(np.max(scores_array))
        
        # Calculate coefficient of variation (relative variability)
        coefficient_variation = std_score / abs(mean_score) if mean_score != 0 else float('inf')
        
        # Calculate stability score (inverse of coefficient of variation, capped at 1.0)
        stability_score = min(1.0, 1.0 / (1.0 + coefficient_variation))
        
        # Calculate 95% confidence interval
        if len(scores) >= 2:
            confidence_interval = stats.t.interval(
                0.95, len(scores)-1, 
                loc=mean_score, 
                scale=stats.sem(scores_array)
            )
        else:
            confidence_interval = (mean_score, mean_score)
        
        # Success rate (assuming all provided scores are successful)
        success_rate = 1.0  # All scores in the list are valid
        
        return StabilityAnalysis(
            mean_score=mean_score,
            std_score=std_score,
            min_score=min_score,
            max_score=max_score,
            stability_score=stability_score,
            confidence_interval_95=confidence_interval,
            coefficient_variation=coefficient_variation,
            success_rate=success_rate
        )
        
    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> float:
        """
        Override base class evaluate_program to ensure profiler logging
        This is the method called by EoH during optimization
        """
        start_time = time.time()
        
        # Call the base class evaluation
        try:
            score = super().evaluate_program(program_str, callable_func, **kwargs)
        except Exception as e:
            self.logger.error(f"Base evaluation failed: {str(e)}")
            score = None
        
        evaluation_time = time.time() - start_time
        
        # Generate function ID for this EoH evaluation
        self.eoh_solution_counter += 1
        function_id = f"eoh_program_{self.eoh_solution_counter}"
        
        # Log to profiler if available
        if self.profiler:
            # Use a very low score for failed evaluations instead of None
            log_score = score if score is not None else -1e10
            self.profiler.log_evaluation(
                score=log_score,
                generation=getattr(self, '_current_generation', 0),
                evaluation_time=evaluation_time,
                solution_data={
                    'function_id': function_id, 
                    'type': 'eoh_program',
                    'success': score is not None,
                    'program_length': len(program_str) if program_str else 0
                }
            )
            if score is not None:
                self.logger.info(f"EoH program evaluation logged: score={score:.6f}")
            else:
                self.logger.info(f"EoH program evaluation failed (logged as -1e10 for tracking)")
        
        # Create detailed metrics
        solution_metrics = SolutionQualityMetrics(
            fitness_score=score if score is not None else float('-inf'),
            accuracy_score=score if score is not None else float('-inf'),
            complexity_score=len(program_str) if program_str else 0,
            evaluation_time=evaluation_time,
            success=score is not None,
            equation_form=program_str[:100] + "..." if program_str and len(program_str) > 100 else program_str,
            error_message=None if score is not None else "Evaluation failed"
        )
        
        # Create evaluation result entry for internal tracking
        evaluation_result = {
            'function_id': function_id,
            'timestamp': time.time(),
            'stability_runs': 1,
            'individual_scores': [score] if score is not None else [],
            'detailed_metrics': [asdict(solution_metrics)],
            'summary': {
                'best_score': score if score is not None else float('-inf'),
                'worst_score': score if score is not None else float('-inf'),
                'mean_score': score if score is not None else float('-inf'),
                'total_evaluation_time': evaluation_time
            },
            'program_str': program_str
        }
        
        # Add to evaluation history (but don't double-count with stability analysis)
        if not self._in_stability_analysis:
            self.evaluation_history.append(evaluation_result)
            
            # Update solution diversity data
            if score is not None:
                self.solution_diversity_data.append({
                    'generation': getattr(self, '_current_generation', len(self.evaluation_history)),
                    'score': score,
                    'solution': {'function_id': function_id, 'equation_form': solution_metrics.equation_form},
                    'timestamp': time.time()
                })
        
        return score
        
    def set_generation_context(self, generation: int):
        """Set the current generation context for tracking"""
        self._current_generation = generation
        
    def analyze_solution_diversity(self) -> Dict[str, Any]:
        """
        Analyze diversity of solutions explored during optimization
        
        Returns:
            Dictionary containing diversity analysis results
        """
        if len(self.solution_diversity_data) < 2:
            return {'diversity_score': 0.0, 'unique_solutions': 0}
        
        # Extract scores
        scores = [s['score'] for s in self.solution_diversity_data if s['score'] is not None]
        
        if len(scores) < 2:
            return {'diversity_score': 0.0, 'unique_solutions': 0}
        
        # Calculate score diversity
        score_std = np.std(scores)
        score_mean = np.mean(scores)
        score_diversity = score_std / abs(score_mean) if score_mean != 0 else 0.0
        
        # Count unique scores (approximate)
        unique_scores = len(set(np.round(scores, 4)))  # Round to 4 decimals for uniqueness
        
        # Calculate generation diversity (how spread out across generations)
        generations = [s.get('generation', 0) for s in self.solution_diversity_data]
        generation_span = max(generations) - min(generations) + 1 if generations else 1
        generation_diversity = len(set(generations)) / generation_span if generation_span > 0 else 0.0
        
        # Overall diversity score (combination of score and generation diversity)
        overall_diversity = (score_diversity + generation_diversity) / 2.0
        
        diversity_analysis = {
            'diversity_score': overall_diversity,
            'score_diversity': score_diversity,
            'generation_diversity': generation_diversity,
            'unique_solutions': unique_scores,
            'total_solutions': len(self.solution_diversity_data),
            'generation_span': generation_span,
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'range': float(np.max(scores) - np.min(scores))
            }
        }
        
        self.logger.info(f"Solution diversity analysis: {overall_diversity:.3f} diversity score, {unique_scores} unique solutions")
        
        return diversity_analysis
        
    def generate_performance_trend_analysis(self) -> Dict[str, Any]:
        """
        Analyze performance trends over time/generations
        
        Returns:
            Dictionary containing trend analysis results
        """
        if not self.evaluation_history:
            return {'trend': 'no_data'}
        
        # Extract scores and timestamps
        scores = []
        timestamps = []
        
        for eval_result in self.evaluation_history:
            best_score = eval_result['summary']['best_score']
            if best_score != float('-inf'):
                scores.append(best_score)
                timestamps.append(eval_result['timestamp'])
        
        if len(scores) < 3:
            return {'trend': 'insufficient_data', 'data_points': len(scores)}
        
        # Calculate linear trend
        time_normalized = np.array(timestamps) - timestamps[0]  # Start from 0
        scores_array = np.array(scores)
        
        # Linear regression
        coeffs = np.polyfit(time_normalized, scores_array, 1)
        slope = coeffs[0]
        
        # Determine trend direction
        if slope > 0.001:  # Significant positive slope
            trend_direction = 'improving'
        elif slope < -0.001:  # Significant negative slope
            trend_direction = 'degrading'
        else:
            trend_direction = 'stable'
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(time_normalized, scores_array)[0, 1]
        
        # Performance consistency (lower variance = more consistent)
        consistency_score = 1.0 / (1.0 + np.var(scores_array))
        
        # Calculate improvement rate (best score - first score) / time
        time_span = timestamps[-1] - timestamps[0]
        score_improvement = scores[-1] - scores[0]
        improvement_rate = score_improvement / time_span if time_span > 0 else 0.0
        
        trend_analysis = {
            'trend_direction': trend_direction,
            'slope': float(slope),
            'correlation': float(correlation),
            'consistency_score': float(consistency_score),
            'improvement_rate': float(improvement_rate),
            'total_improvement': float(score_improvement),
            'time_span': float(time_span),
            'data_points': len(scores),
            'performance_statistics': {
                'first_score': float(scores[0]),
                'last_score': float(scores[-1]),
                'best_score': float(np.max(scores_array)),
                'worst_score': float(np.min(scores_array)),
                'mean_score': float(np.mean(scores_array)),
                'std_score': float(np.std(scores_array))
            }
        }
        
        self.logger.info(f"Performance trend: {trend_direction}, improvement rate: {improvement_rate:.6f}/s")
        
        return trend_analysis
        
    def compare_solutions(self, solution_a_id: str, solution_b_id: str) -> Dict[str, Any]:
        """
        Compare two solutions in detail
        
        Args:
            solution_a_id: ID of first solution
            solution_b_id: ID of second solution
            
        Returns:
            Dictionary containing detailed comparison
        """
        # Find solutions in detailed metrics
        solution_a = self.detailed_solution_metrics.get(solution_a_id)
        solution_b = self.detailed_solution_metrics.get(solution_b_id)
        
        if not solution_a or not solution_b:
            return {'error': 'One or both solutions not found'}
        
        # Extract key metrics
        a_best = solution_a['summary']['best_score']
        b_best = solution_b['summary']['best_score']
        
        a_mean = solution_a['summary']['mean_score']
        b_mean = solution_b['summary']['mean_score']
        
        a_stability = solution_a['stability_metrics']['stability_score']
        b_stability = solution_b['stability_metrics']['stability_score']
        
        # Performance comparison
        performance_comparison = {
            'best_score_winner': solution_a_id if a_best > b_best else solution_b_id,
            'best_score_difference': abs(a_best - b_best),
            'mean_score_winner': solution_a_id if a_mean > b_mean else solution_b_id,
            'mean_score_difference': abs(a_mean - b_mean),
            'stability_winner': solution_a_id if a_stability > b_stability else solution_b_id,
            'stability_difference': abs(a_stability - b_stability)
        }
        
        # Statistical significance test (if enough data)
        a_scores = [s for s in solution_a['individual_scores'] if s != float('-inf')]
        b_scores = [s for s in solution_b['individual_scores'] if s != float('-inf')]
        
        statistical_test = {}
        if len(a_scores) >= 3 and len(b_scores) >= 3:
            try:
                t_stat, p_value = stats.ttest_ind(a_scores, b_scores)
                statistical_test = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_difference': p_value < 0.05,
                    'effect_size': abs(np.mean(a_scores) - np.mean(b_scores)) / np.sqrt((np.var(a_scores) + np.var(b_scores)) / 2)
                }
            except Exception as e:
                statistical_test = {'error': str(e)}
        
        comparison_result = {
            'solution_a_id': solution_a_id,
            'solution_b_id': solution_b_id,
            'performance_comparison': performance_comparison,
            'statistical_test': statistical_test,
            'detailed_metrics': {
                'solution_a': solution_a['summary'],
                'solution_b': solution_b['summary']
            },
            'recommendation': self._generate_solution_recommendation(solution_a, solution_b)
        }
        
        return comparison_result
        
    def _generate_solution_recommendation(self, solution_a: Dict, solution_b: Dict) -> str:
        """Generate a recommendation based on solution comparison"""
        a_best = solution_a['summary']['best_score']
        b_best = solution_b['summary']['best_score']
        a_stability = solution_a['stability_metrics']['stability_score']
        b_stability = solution_b['stability_metrics']['stability_score']
        
        # Simple recommendation logic
        if abs(a_best - b_best) > 0.1:  # Significant performance difference
            if a_best > b_best:
                return f"Solution A is significantly better (score difference: {a_best - b_best:.4f})"
            else:
                return f"Solution B is significantly better (score difference: {b_best - a_best:.4f})"
        else:
            # Performance is similar, look at stability
            if abs(a_stability - b_stability) > 0.1:
                if a_stability > b_stability:
                    return "Solution A is more stable with similar performance"
                else:
                    return "Solution B is more stable with similar performance"
            else:
                return "Both solutions have similar performance and stability"
        
    def get_best_solutions_summary(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Get summary of the best solutions found
        
        Args:
            top_k: Number of top solutions to include
            
        Returns:
            Dictionary containing best solutions summary
        """
        if not self.best_solutions:
            return {'message': 'No solutions evaluated yet'}
        
        # Sort solutions by score (descending)
        sorted_solutions = sorted(self.best_solutions, key=lambda x: x['score'], reverse=True)
        top_solutions = sorted_solutions[:top_k]
        
        summary = {
            'top_solutions': top_solutions,
            'total_solutions_evaluated': len(self.evaluation_history),
            'best_overall_score': top_solutions[0]['score'] if top_solutions else float('-inf'),
            'score_distribution': {
                'max': max(s['score'] for s in self.best_solutions),
                'min': min(s['score'] for s in self.best_solutions),
                'mean': np.mean([s['score'] for s in self.best_solutions]),
                'std': np.std([s['score'] for s in self.best_solutions])
            },
            'stability_analysis': {
                'most_stable': max(self.best_solutions, 
                                 key=lambda x: x.get('stability_metrics', {}).get('stability_score', 0))['function_id'],
                'avg_stability': np.mean([s.get('stability_metrics', {}).get('stability_score', 0) 
                                        for s in self.best_solutions])
            }
        }
        
        return summary
        
    def export_comprehensive_report(self, output_file: str) -> Dict[str, Any]:
        """
        Export comprehensive evaluation report
        
        Args:
            output_file: Path to save the report
            
        Returns:
            Dictionary containing the complete report
        """
        report = {
            'experiment_info': {
                'evaluation_class': self.__class__.__name__,
                'stability_runs': self.stability_runs,
                'collect_detailed_metrics': self.collect_detailed_metrics,
                'export_timestamp': time.time(),
                'total_evaluations': len(self.evaluation_history)
            },
            'performance_summary': self.get_best_solutions_summary(),
            'diversity_analysis': self.analyze_solution_diversity(),
            'trend_analysis': self.generate_performance_trend_analysis(),
            'overall_statistics': self._calculate_overall_statistics(),
            'detailed_evaluation_history': self.evaluation_history,
            'solution_diversity_data': self.solution_diversity_data
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive report exported to {output_file}")
        
        return report
        
    def _calculate_overall_statistics(self) -> Dict[str, Any]:
        """Calculate overall statistics across all evaluations"""
        if not self.evaluation_history:
            return {}
        
        all_scores = []
        total_evaluation_time = 0.0
        successful_evaluations = 0
        
        for eval_result in self.evaluation_history:
            # Collect all individual scores
            individual_scores = eval_result.get('individual_scores', [])
            valid_scores = [s for s in individual_scores if s != float('-inf')]
            all_scores.extend(valid_scores)
            
            # Accumulate timing
            total_evaluation_time += eval_result['summary'].get('total_evaluation_time', 0.0)
            
            # Count successful evaluations
            if valid_scores:
                successful_evaluations += 1
        
        if not all_scores:
            return {'message': 'No successful evaluations'}
        
        statistics = {
            'total_functions_evaluated': len(self.evaluation_history),
            'successful_evaluations': successful_evaluations,
            'success_rate': successful_evaluations / len(self.evaluation_history),
            'total_individual_runs': len(all_scores),
            'total_evaluation_time': total_evaluation_time,
            'avg_time_per_evaluation': total_evaluation_time / len(all_scores) if all_scores else 0.0,
            'score_statistics': {
                'best': float(np.max(all_scores)),
                'worst': float(np.min(all_scores)),
                'mean': float(np.mean(all_scores)),
                'median': float(np.median(all_scores)),
                'std': float(np.std(all_scores)),
                'variance': float(np.var(all_scores)),
                'range': float(np.max(all_scores) - np.min(all_scores)),
                'percentiles': {
                    'p25': float(np.percentile(all_scores, 25)),
                    'p75': float(np.percentile(all_scores, 75)),
                    'p90': float(np.percentile(all_scores, 90)),
                    'p95': float(np.percentile(all_scores, 95))
                }
            }
        }
        
        return statistics 