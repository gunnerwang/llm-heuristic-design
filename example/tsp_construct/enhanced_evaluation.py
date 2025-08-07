"""
Enhanced TSP Evaluation with Comprehensive Analysis
==================================================

This module provides enhanced evaluation capabilities for TSP optimization with:
- Stability analysis through multiple runs
- Solution diversity tracking
- Performance trend analysis
- Statistical significance testing
- Detailed solution quality metrics

Usage:
    enhanced_task = EnhancedTSPEvaluation(
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

# Import base TSP evaluation
from llm4ad.task.optimization.tsp_construct.evaluation import TSPEvaluation


@dataclass
class SolutionQualityMetrics:
    """Detailed metrics for a single TSP solution"""
    tour_length: float
    normalized_score: float
    tour_route: List[int]
    evaluation_time: float
    success: bool
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


class EnhancedTSPEvaluation(TSPEvaluation):
    """
    Enhanced TSP evaluation with comprehensive analysis capabilities
    """
    
    def __init__(self, profiler=None, stability_runs: int = 3, 
                 collect_detailed_metrics: bool = True,
                 timeout_seconds=30, n_instance=16, problem_size=50, **kwargs):
        
        super().__init__(
            timeout_seconds=timeout_seconds,
            n_instance=n_instance, 
            problem_size=problem_size,
            **kwargs
        )
        
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
        # Call the base class evaluate method to get the score
        score = super().evaluate(evaluator)
        
        # If we're in stability analysis mode, don't track here (it's handled by evaluate_with_stability_analysis)
        if self._in_stability_analysis:
            return score
        
        # Generate a unique function ID for this EoH solution
        self.eoh_solution_counter += 1
        function_id = f"eoh_solution_{self.eoh_solution_counter}"
        
        # Create detailed metrics for this single evaluation
        evaluation_time = getattr(self, '_last_evaluation_time', 0.0)  # Try to get timing if available
        
        solution_metrics = SolutionQualityMetrics(
            tour_length=-score if score is not None else float('inf'),
            normalized_score=score if score is not None else float('-inf'),
            tour_route=[],  # Would need to capture from evaluator if available
            evaluation_time=evaluation_time,
            success=score is not None,
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
                'solution': {'function_id': function_id},
                'timestamp': time.time()
            })
        
        # Log to profiler if available
        if self.profiler:
            self.profiler.log_evaluation(
                score=score,
                evaluation_time=evaluation_time,
                solution_data={'function_id': function_id, 'type': 'eoh_intermediate'}
            )
        
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
        
        self.logger.info(f"Captured EoH evaluation: {function_id} with score {score}")
        
        return score
        
    def evaluate_with_stability_analysis(self, evaluator: Callable, 
                                       function_id: str = None) -> Dict[str, Any]:
        """
        Evaluate a TSP solution with stability analysis across multiple runs
        """
        start_time = time.time()
        
        # Set flag to indicate we're in stability analysis mode
        self._in_stability_analysis = True
        
        results = []
        detailed_metrics = []
        
        for run_id in range(self.stability_runs):
            try:
                run_start = time.time()
                
                # Run evaluation
                score = self.evaluate(evaluator)
                
                run_time = time.time() - run_start
                
                # Create detailed metrics
                if score is not None:
                    solution_metrics = SolutionQualityMetrics(
                        tour_length=-score,  # Score is negative tour length
                        normalized_score=score,
                        tour_route=[],  # Would need to capture from evaluator
                        evaluation_time=run_time,
                        success=True
                    )
                    results.append(score)
                else:
                    solution_metrics = SolutionQualityMetrics(
                        tour_length=float('inf'),
                        normalized_score=float('-inf'),
                        tour_route=[],
                        evaluation_time=run_time,
                        success=False,
                        error_message="Evaluation returned None"
                    )
                    
                detailed_metrics.append(solution_metrics)
                
                # Log to profiler if available
                if self.profiler:
                    self.profiler.log_evaluation(
                        score=score,
                        evaluation_time=run_time,
                        solution_data={'run_id': run_id, 'function_id': function_id}
                    )
                    
            except Exception as e:
                self.logger.error(f"Evaluation run {run_id} failed: {str(e)}")
                solution_metrics = SolutionQualityMetrics(
                    tour_length=float('inf'),
                    normalized_score=float('-inf'),
                    tour_route=[],
                    evaluation_time=0,
                    success=False,
                    error_message=str(e)
                )
                detailed_metrics.append(solution_metrics)
        
        # Reset flag 
        self._in_stability_analysis = False
        
        # Calculate stability metrics
        stability_analysis = self._calculate_stability_metrics(results)
        
        # Store results
        evaluation_result = {
            'function_id': function_id,
            'timestamp': start_time,
            'stability_runs': self.stability_runs,
            'individual_scores': results,
            'detailed_metrics': [asdict(m) for m in detailed_metrics],
            'stability_metrics': asdict(stability_analysis),
            'summary': {
                'best_score': max(results) if results else float('-inf'),
                'worst_score': min(results) if results else float('-inf'),
                'mean_score': np.mean(results) if results else float('-inf'),
                'total_evaluation_time': sum(m.evaluation_time for m in detailed_metrics)
            }
        }
        
        # Store for analysis
        self.evaluation_history.append(evaluation_result)
        
        # Track best and worst solutions
        if results:
            best_score = max(results)
            worst_score = min(results)
            
            if not self.best_solutions or best_score > max(r['score'] for r in self.best_solutions):
                self.best_solutions.append({
                    'score': best_score,
                    'function_id': function_id,
                    'timestamp': start_time,
                    'stability_metrics': asdict(stability_analysis)
                })
                
            if not self.worst_solutions or worst_score < min(r['score'] for r in self.worst_solutions):
                self.worst_solutions.append({
                    'score': worst_score,
                    'function_id': function_id,
                    'timestamp': start_time
                })
        
        return evaluation_result
        
    def _calculate_stability_metrics(self, scores: List[float]) -> StabilityAnalysis:
        """Calculate comprehensive stability metrics"""
        if not scores:
            return StabilityAnalysis(
                mean_score=float('-inf'),
                std_score=float('inf'),
                min_score=float('-inf'),
                max_score=float('-inf'),
                stability_score=0.0,
                confidence_interval_95=(float('-inf'), float('-inf')),
                coefficient_variation=float('inf'),
                success_rate=0.0
            )
            
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        
        # Calculate confidence interval
        if len(scores) > 1:
            sem = stats.sem(scores_array)
            confidence_interval = stats.t.interval(
                0.95, len(scores)-1, loc=mean_score, scale=sem
            )
        else:
            confidence_interval = (mean_score, mean_score)
            
        # Stability score (higher is more stable)
        if std_score > 0 and abs(mean_score) > 1e-8:
            coefficient_variation = std_score / abs(mean_score)
            stability_score = 1.0 / (1.0 + coefficient_variation)
        else:
            coefficient_variation = 0.0
            stability_score = 1.0
            
        return StabilityAnalysis(
            mean_score=mean_score,
            std_score=std_score,
            min_score=float(np.min(scores_array)),
            max_score=float(np.max(scores_array)),
            stability_score=stability_score,
            confidence_interval_95=confidence_interval,
            coefficient_variation=coefficient_variation,
            success_rate=1.0  # All scores were successful if we got here
        )
        
    def set_generation_context(self, generation: int):
        """Set current generation context for tracking EoH evaluations"""
        self._current_generation = generation
        
    def analyze_solution_diversity(self) -> Dict[str, Any]:
        """Analyze diversity of solutions across evaluations"""
        if len(self.evaluation_history) < 2:
            return {'diversity_score': 0.0, 'analysis': 'Insufficient data for diversity analysis'}
            
        # Extract all scores from all evaluations (including baselines and EoH solutions)
        all_scores = []
        for eval_result in self.evaluation_history:
            all_scores.extend(eval_result['individual_scores'])
            
        if len(all_scores) < 2:
            return {'diversity_score': 0.0, 'analysis': 'Insufficient scores for diversity analysis'}
            
        # Calculate diversity metrics
        scores_array = np.array(all_scores)
        
        # Separate baseline and EoH solutions for analysis
        baseline_scores = []
        eoh_scores = []
        
        for eval_result in self.evaluation_history:
            if 'baseline' in eval_result['function_id']:
                baseline_scores.extend(eval_result['individual_scores'])
            elif 'eoh_solution' in eval_result['function_id']:
                eoh_scores.extend(eval_result['individual_scores'])
        
        diversity_metrics = {
            'total_evaluations': len(all_scores),
            'baseline_evaluations': len(baseline_scores),
            'eoh_evaluations': len(eoh_scores),
            'unique_scores': len(np.unique(scores_array)),
            'score_range': float(np.max(scores_array) - np.min(scores_array)),
            'score_variance': float(np.var(scores_array)),
            'diversity_score': len(np.unique(scores_array)) / len(all_scores),
            'quartile_spread': float(np.percentile(scores_array, 75) - np.percentile(scores_array, 25)),
            'score_distribution': {
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'skewness': float(stats.skew(scores_array)),
                'kurtosis': float(stats.kurtosis(scores_array))
            }
        }
        
        # Add comparison between baselines and EoH solutions
        if baseline_scores and eoh_scores:
            baseline_array = np.array(baseline_scores)
            eoh_array = np.array(eoh_scores)
            
            diversity_metrics['baseline_vs_eoh'] = {
                'baseline_mean': float(np.mean(baseline_array)),
                'baseline_std': float(np.std(baseline_array)),
                'eoh_mean': float(np.mean(eoh_array)),
                'eoh_std': float(np.std(eoh_array)),
                'improvement': float(np.mean(eoh_array) - np.mean(baseline_array)),
                'best_eoh_vs_best_baseline': float(np.max(eoh_array) - np.max(baseline_array))
            }
        
        return diversity_metrics
        
    def generate_performance_trend_analysis(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(self.evaluation_history) < 2:
            return {'trend': 'insufficient_data'}
            
        # Extract chronological performance data
        timeline_data = []
        for i, eval_result in enumerate(self.evaluation_history):
            timeline_data.append({
                'evaluation_index': i,
                'timestamp': eval_result['timestamp'],
                'best_score': eval_result['summary']['best_score'],
                'mean_score': eval_result['summary']['mean_score'],
                'stability_score': eval_result['stability_metrics']['stability_score']
            })
            
        # Calculate trends
        best_scores = [d['best_score'] for d in timeline_data if d['best_score'] != float('-inf')]
        mean_scores = [d['mean_score'] for d in timeline_data if d['mean_score'] != float('-inf')]
        
        trend_analysis = {
            'total_evaluations': len(timeline_data),
            'timeline_data': timeline_data
        }
        
        if len(best_scores) > 1:
            # Linear regression for trend
            x = np.arange(len(best_scores))
            slope_best, intercept_best, r_value_best, p_value_best, _ = stats.linregress(x, best_scores)
            slope_mean, intercept_mean, r_value_mean, p_value_mean, _ = stats.linregress(x, mean_scores)
            
            trend_analysis.update({
                'best_score_trend': {
                    'slope': float(slope_best),
                    'intercept': float(intercept_best),
                    'correlation': float(r_value_best),
                    'p_value': float(p_value_best),
                    'trend_direction': 'improving' if slope_best > 0 else 'declining' if slope_best < 0 else 'stable'
                },
                'mean_score_trend': {
                    'slope': float(slope_mean),
                    'intercept': float(intercept_mean),
                    'correlation': float(r_value_mean),
                    'p_value': float(p_value_mean),
                    'trend_direction': 'improving' if slope_mean > 0 else 'declining' if slope_mean < 0 else 'stable'
                }
            })
            
        return trend_analysis
        
    def compare_solutions(self, solution_a_id: str, solution_b_id: str) -> Dict[str, Any]:
        """Compare two solutions statistically"""
        # Find evaluations for both solutions
        eval_a = None
        eval_b = None
        
        for eval_result in self.evaluation_history:
            if eval_result['function_id'] == solution_a_id:
                eval_a = eval_result
            elif eval_result['function_id'] == solution_b_id:
                eval_b = eval_result
                
        if not eval_a or not eval_b:
            return {'error': 'One or both solutions not found in evaluation history'}
            
        scores_a = eval_a['individual_scores']
        scores_b = eval_b['individual_scores']
        
        if not scores_a or not scores_b:
            return {'error': 'Insufficient data for comparison'}
            
        # Statistical comparison
        try:
            # T-test for mean difference
            t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
            
            comparison = {
                'solution_a': {
                    'id': solution_a_id,
                    'mean_score': np.mean(scores_a),
                    'std_score': np.std(scores_a),
                    'best_score': max(scores_a),
                    'n_runs': len(scores_a)
                },
                'solution_b': {
                    'id': solution_b_id,
                    'mean_score': np.mean(scores_b),
                    'std_score': np.std(scores_b),
                    'best_score': max(scores_b),
                    'n_runs': len(scores_b)
                },
                'statistical_tests': {
                    't_test': {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    },
                    'mann_whitney_u': {
                        'statistic': float(u_stat),
                        'p_value': float(u_p_value),
                        'significant': u_p_value < 0.05
                    }
                },
                'practical_significance': {
                    'mean_difference': np.mean(scores_a) - np.mean(scores_b),
                    'effect_size': (np.mean(scores_a) - np.mean(scores_b)) / np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2),
                    'better_solution': solution_a_id if np.mean(scores_a) > np.mean(scores_b) else solution_b_id
                }
            }
            
            return comparison
            
        except Exception as e:
            return {'error': f'Statistical comparison failed: {str(e)}'}
            
    def get_best_solutions_summary(self, top_k: int = 5) -> Dict[str, Any]:
        """Get summary of top performing solutions"""
        if not self.evaluation_history:
            return {'error': 'No evaluations performed yet'}
            
        # Sort evaluations by best score
        sorted_evaluations = sorted(
            self.evaluation_history,
            key=lambda x: x['summary']['best_score'],
            reverse=True
        )
        
        top_solutions = sorted_evaluations[:top_k]
        
        summary = {
            'top_k': top_k,
            'total_evaluations': len(self.evaluation_history),
            'best_solutions': []
        }
        
        for i, eval_result in enumerate(top_solutions):
            solution_summary = {
                'rank': i + 1,
                'function_id': eval_result['function_id'],
                'best_score': eval_result['summary']['best_score'],
                'mean_score': eval_result['summary']['mean_score'],
                'stability_score': eval_result['stability_metrics']['stability_score'],
                'success_rate': eval_result['stability_metrics']['success_rate'],
                'evaluation_time': eval_result['summary']['total_evaluation_time']
            }
            summary['best_solutions'].append(solution_summary)
            
        return summary
        
    def export_comprehensive_report(self, output_file: str) -> Dict[str, Any]:
        """Export comprehensive evaluation report"""
        report = {
            'metadata': {
                'timestamp': time.time(),
                'stability_runs': self.stability_runs,
                'collect_detailed_metrics': self.collect_detailed_metrics,
                'total_evaluations': len(self.evaluation_history),
                'problem_parameters': {
                    'n_instance': self.n_instance,
                    'problem_size': self.problem_size,
                    'timeout_seconds': self.timeout_seconds
                }
            },
            'evaluation_history': self.evaluation_history,
            'solution_diversity': self.analyze_solution_diversity(),
            'performance_trends': self.generate_performance_trend_analysis(),
            'best_solutions_summary': self.get_best_solutions_summary(),
            'overall_statistics': self._calculate_overall_statistics()
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Comprehensive report exported to {output_file}")
        return report
        
    def _calculate_overall_statistics(self) -> Dict[str, Any]:
        """Calculate overall statistics across all evaluations"""
        if not self.evaluation_history:
            return {}
            
        # Collect all individual scores
        all_scores = []
        all_evaluation_times = []
        success_count = 0
        total_runs = 0
        
        for eval_result in self.evaluation_history:
            scores = eval_result['individual_scores']
            all_scores.extend(scores)
            all_evaluation_times.append(eval_result['summary']['total_evaluation_time'])
            
            for metric in eval_result['detailed_metrics']:
                total_runs += 1
                if metric['success']:
                    success_count += 1
                    
        if not all_scores:
            return {}
            
        scores_array = np.array(all_scores)
        
        return {
            'performance_summary': {
                'total_function_evaluations': len(self.evaluation_history),
                'total_individual_runs': total_runs,
                'overall_success_rate': success_count / total_runs if total_runs > 0 else 0,
                'best_score_overall': float(np.max(scores_array)),
                'worst_score_overall': float(np.min(scores_array)),
                'mean_score_overall': float(np.mean(scores_array)),
                'std_score_overall': float(np.std(scores_array)),
                'median_score_overall': float(np.median(scores_array))
            },
            'timing_summary': {
                'total_evaluation_time': sum(all_evaluation_times),
                'average_evaluation_time': np.mean(all_evaluation_times),
                'fastest_evaluation': min(all_evaluation_times) if all_evaluation_times else 0,
                'slowest_evaluation': max(all_evaluation_times) if all_evaluation_times else 0
            },
            'quality_metrics': {
                'score_improvement_range': float(np.max(scores_array) - np.min(scores_array)),
                'coefficient_of_variation': float(np.std(scores_array) / abs(np.mean(scores_array))) if abs(np.mean(scores_array)) > 1e-8 else float('inf'),
                'quartile_1': float(np.percentile(scores_array, 25)),
                'quartile_3': float(np.percentile(scores_array, 75)),
                'interquartile_range': float(np.percentile(scores_array, 75) - np.percentile(scores_array, 25))
            }
        } 