"""
Comprehensive Metrics System for ODE Discovery
==============================================

This module provides detailed metrics collection and analysis for ODE (Ordinary Differential Equation)
discovery experiments using LLM-based methods like EoH.

Features:
- Performance tracking (scores, convergence, diversity)
- Cost analysis (tokens, API calls, costs)
- Stability analysis (multiple runs, variance)
- Efficiency metrics (time, evaluations)
- Comparative analysis across experiments
- Detailed reporting and visualization

Usage:
    profiler = ComprehensiveProfiler(log_dir="logs", track_tokens=True)
    # ... run experiments ...
    profiler.calculate_final_metrics()
    profiler.save_metrics("metrics.json")
"""

import json
import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

# Import base profiler from LLM4AD framework
from llm4ad.tools.profiler import ProfilerBase


@dataclass
class PerformanceMetrics:
    """Core performance metrics for ODE discovery"""
    best_score: float = float('-inf')
    worst_score: float = float('inf')
    mean_score: float = 0.0
    median_score: float = 0.0
    std_score: float = 0.0
    scores_history: List[float] = None
    convergence_generation: int = -1
    total_evaluator_calls: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    
    def __post_init__(self):
        if self.scores_history is None:
            self.scores_history = []


@dataclass 
class TokenMetrics:
    """Token usage and cost metrics"""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_per_token: float = 0.0
    api_calls: int = 0
    avg_tokens_per_call: float = 0.0


@dataclass
class TimingMetrics:
    """Timing and efficiency metrics"""
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration: float = 0.0
    avg_evaluation_time: float = 0.0
    llm_call_time: float = 0.0
    evaluation_time: float = 0.0


@dataclass
class StabilityMetrics:
    """Stability and reliability metrics"""
    success_rate: float = 0.0
    convergence_consistency: float = 0.0
    solution_diversity: float = 0.0
    performance_variance: float = 0.0
    robustness_score: float = 0.0


class ComprehensiveProfiler(ProfilerBase):
    """
    Comprehensive profiler for ODE discovery experiments
    Inherits from ProfilerBase for compatibility with LLM4AD framework
    """
    
    def __init__(self, log_dir: str = "logs", log_style: str = "complex", track_tokens: bool = True):
        # Initialize base profiler
        super().__init__(log_dir=log_dir, log_style=log_style)
        
        self.track_tokens = track_tokens
        
        # Initialize enhanced metrics
        self.metrics = PerformanceMetrics()
        self.token_metrics = TokenMetrics()
        self.timing_metrics = TimingMetrics()
        self.stability_metrics = StabilityMetrics()
        
        # Additional tracking
        self.generation_scores = {}  # generation -> list of scores
        self.evaluation_times = []
        self.llm_call_logs = []
        self.solution_diversity_history = []
        self.experiment_metadata = {}
        
        # Set up enhanced logging
        self._setup_enhanced_logging()
        
        # Start timing
        self.timing_metrics.start_time = time.time()

    @property
    def log_dir(self) -> str:
        """Property to access log directory from base class"""
        return self._log_dir

    @property  
    def log_style(self) -> str:
        """Property to access log style from base class"""
        return self._log_style
        
    def _setup_enhanced_logging(self):
        """Setup enhanced logging configuration"""
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_file = os.path.join(self.log_dir, "comprehensive_metrics.log")
        
        # Create a specific logger for comprehensive metrics
        self.comp_logger = logging.getLogger(f"{__name__}.comprehensive")
        self.comp_logger.setLevel(logging.INFO)
        
        # Only add handlers if they don't exist
        if not self.comp_logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.comp_logger.addHandler(file_handler)
            self.comp_logger.addHandler(console_handler)
        
    def log_evaluation(self, score: float, generation: int = None, evaluation_time: float = None,
                      solution_data: Dict = None):
        """
        Log a single evaluation result
        Enhanced version that also calls the base class method for compatibility
        """
        # Enhanced logging - filter out artificial failure scores
        is_artificial_failure = score == -1e10
        
        if score is not None and not np.isnan(score) and not is_artificial_failure:
            self.metrics.scores_history.append(score)
            self.metrics.successful_evaluations += 1
            
            if score > self.metrics.best_score:
                self.metrics.best_score = score
                if generation is not None:
                    self.metrics.convergence_generation = generation
                    
            if score < self.metrics.worst_score:
                self.metrics.worst_score = score
        else:
            self.metrics.failed_evaluations += 1
            
        self.metrics.total_evaluator_calls += 1
        
        # Track by generation
        if generation is not None:
            if generation not in self.generation_scores:
                self.generation_scores[generation] = []
            if score is not None and not np.isnan(score):
                self.generation_scores[generation].append(score)
        
        # Track evaluation time
        if evaluation_time is not None:
            self.evaluation_times.append(evaluation_time)
            
        # Store solution data for diversity analysis
        if solution_data:
            self.solution_diversity_history.append({
                'generation': generation,
                'score': score,
                'solution_data': solution_data,
                'timestamp': time.time()
            })
            
        # Log to console and file with enhanced debugging
        if is_artificial_failure:
            self.comp_logger.info(f"Evaluation FAILED - Generation: {generation}, Time: {evaluation_time:.3f}s")
        else:
            self.comp_logger.info(f"Evaluation - Score: {score:.4f}, Generation: {generation}, Time: {evaluation_time:.3f}s")
        self.comp_logger.info(f"Current best score after logging: {self.metrics.best_score:.4f}")
        self.comp_logger.info(f"Total successful evaluations: {self.metrics.successful_evaluations}, Failed: {self.metrics.failed_evaluations}, Total calls: {self.metrics.total_evaluator_calls}")
        
    def log_llm_call(self, tokens_used: int, cost: float = 0.0, call_time: float = None,
                     prompt_tokens: int = None, completion_tokens: int = None):
        """Log LLM API call metrics"""
        if self.track_tokens:
            self.token_metrics.total_tokens += tokens_used
            self.token_metrics.total_cost_usd += cost
            self.token_metrics.api_calls += 1
            
            if prompt_tokens:
                self.token_metrics.prompt_tokens += prompt_tokens
            if completion_tokens:
                self.token_metrics.completion_tokens += completion_tokens
                
            if call_time:
                self.timing_metrics.llm_call_time += call_time
                
            # Store detailed call info
            self.llm_call_logs.append({
                'timestamp': time.time(),
                'tokens': tokens_used,
                'cost': cost,
                'call_time': call_time,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            })
            
    def calculate_final_metrics(self):
        """Calculate final aggregated metrics"""
        self.timing_metrics.end_time = time.time()
        self.timing_metrics.total_duration = self.timing_metrics.end_time - self.timing_metrics.start_time
        
        # Calculate performance metrics
        if self.metrics.scores_history:
            scores = np.array(self.metrics.scores_history)
            self.metrics.mean_score = float(np.mean(scores))
            self.metrics.median_score = float(np.median(scores))
            self.metrics.std_score = float(np.std(scores))
            
        # Calculate token metrics
        if self.token_metrics.api_calls > 0:
            self.token_metrics.avg_tokens_per_call = self.token_metrics.total_tokens / self.token_metrics.api_calls
            self.token_metrics.cost_per_token = self.token_metrics.total_cost_usd / self.token_metrics.total_tokens if self.token_metrics.total_tokens > 0 else 0
            
        # Calculate timing metrics
        if self.evaluation_times:
            self.timing_metrics.avg_evaluation_time = np.mean(self.evaluation_times)
            
        # Calculate stability metrics
        self._calculate_stability_metrics()
        
        self.comp_logger.info("Final metrics calculated")
        
    def _calculate_stability_metrics(self):
        """Calculate stability and robustness metrics"""
        if not self.metrics.scores_history:
            return
            
        scores = np.array(self.metrics.scores_history)
        
        # Success rate
        self.stability_metrics.success_rate = self.metrics.successful_evaluations / self.metrics.total_evaluator_calls if self.metrics.total_evaluator_calls > 0 else 0
        
        # Performance variance (lower is more stable)
        self.stability_metrics.performance_variance = float(np.var(scores))
        
        # Convergence consistency (how consistently we reach good solutions)
        if len(scores) >= 5:
            # Look at improvement over time
            windows = [scores[i:i+5] for i in range(len(scores)-4)]
            improvements = [np.max(window) - np.min(window) for window in windows]
            self.stability_metrics.convergence_consistency = 1.0 / (1.0 + np.mean(improvements))
        
        # Solution diversity (variety in solutions explored)
        if len(self.solution_diversity_history) >= 2:
            # Calculate score diversity as coefficient of variation
            unique_scores = list(set(s['score'] for s in self.solution_diversity_history if s['score'] is not None))
            if len(unique_scores) > 1:
                self.stability_metrics.solution_diversity = float(np.std(unique_scores) / np.mean(unique_scores))
        
        # Overall robustness score (combines multiple factors)
        self.stability_metrics.robustness_score = (
            self.stability_metrics.success_rate * 0.4 +
            (1.0 / (1.0 + self.stability_metrics.performance_variance)) * 0.3 +
            self.stability_metrics.convergence_consistency * 0.2 +
            min(self.stability_metrics.solution_diversity, 1.0) * 0.1
        )
        
    def get_generation_summary(self) -> Dict[int, Dict]:
        """Get summary statistics by generation"""
        summary = {}
        for gen, scores in self.generation_scores.items():
            if scores:
                summary[gen] = {
                    'best': float(np.max(scores)),
                    'worst': float(np.min(scores)),
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'count': len(scores)
                }
        return summary
        
    def get_efficiency_metrics(self) -> Dict:
        """Get efficiency-related metrics"""
        return {
            'evaluations_per_second': self.metrics.total_evaluator_calls / self.timing_metrics.total_duration if self.timing_metrics.total_duration > 0 else 0,
            'avg_evaluation_time': self.timing_metrics.avg_evaluation_time,
            'total_llm_time': self.timing_metrics.llm_call_time,
            'llm_time_percentage': self.timing_metrics.llm_call_time / self.timing_metrics.total_duration * 100 if self.timing_metrics.total_duration > 0 else 0
        }
        
    def get_stability_metrics(self) -> Dict:
        """Get stability metrics as dictionary"""
        return asdict(self.stability_metrics)
        
    def export_detailed_log(self, filepath: str):
        """Export detailed execution log"""
        detailed_log = {
            'experiment_metadata': self.experiment_metadata,
            'performance_metrics': asdict(self.metrics),
            'token_metrics': asdict(self.token_metrics),
            'timing_metrics': asdict(self.timing_metrics),
            'stability_metrics': asdict(self.stability_metrics),
            'generation_summary': self.get_generation_summary(),
            'efficiency_metrics': self.get_efficiency_metrics(),
            'solution_diversity_history': self.solution_diversity_history,
            'llm_call_logs': self.llm_call_logs,
            'evaluation_times': self.evaluation_times
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(detailed_log, f, indent=2, default=str)
            
        self.comp_logger.info(f"Detailed log exported to {filepath}")
        
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file"""
        self.export_detailed_log(filepath)
        
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        report = []
        report.append("="*60)
        report.append("COMPREHENSIVE ODE DISCOVERY METRICS REPORT")
        report.append("="*60)
        
        # Performance Summary
        report.append(f"\nPERFORMANCE SUMMARY:")
        report.append(f"  Best Score:           {self.metrics.best_score:.6f}")
        report.append(f"  Worst Score:          {self.metrics.worst_score:.6f}")
        report.append(f"  Mean Score:           {self.metrics.mean_score:.6f}")
        report.append(f"  Median Score:         {self.metrics.median_score:.6f}")
        report.append(f"  Standard Deviation:   {self.metrics.std_score:.6f}")
        report.append(f"  Total Evaluations:    {self.metrics.total_evaluator_calls}")
        report.append(f"  Successful Evals:     {self.metrics.successful_evaluations}")
        report.append(f"  Failed Evaluations:   {self.metrics.failed_evaluations}")
        report.append(f"  Convergence Gen:      {self.metrics.convergence_generation}")
        
        # Token and Cost Summary
        if self.track_tokens:
            report.append(f"\nTOKEN USAGE & COST:")
            report.append(f"  Total Tokens:         {self.token_metrics.total_tokens:,}")
            report.append(f"  Prompt Tokens:        {self.token_metrics.prompt_tokens:,}")
            report.append(f"  Completion Tokens:    {self.token_metrics.completion_tokens:,}")
            report.append(f"  Total API Calls:      {self.token_metrics.api_calls}")
            report.append(f"  Avg Tokens/Call:      {self.token_metrics.avg_tokens_per_call:.1f}")
            report.append(f"  Total Cost (USD):     ${self.token_metrics.total_cost_usd:.4f}")
            report.append(f"  Cost per Token:       ${self.token_metrics.cost_per_token:.6f}")
        
        # Timing Summary
        report.append(f"\nTIMING & EFFICIENCY:")
        report.append(f"  Total Duration:       {self.timing_metrics.total_duration:.2f}s")
        report.append(f"  Avg Evaluation Time:  {self.timing_metrics.avg_evaluation_time:.3f}s")
        report.append(f"  Total LLM Time:       {self.timing_metrics.llm_call_time:.2f}s")
        report.append(f"  Evaluations/Second:   {self.get_efficiency_metrics()['evaluations_per_second']:.2f}")
        
        # Stability Summary
        report.append(f"\nSTABILITY & ROBUSTNESS:")
        report.append(f"  Success Rate:         {self.stability_metrics.success_rate:.3f}")
        report.append(f"  Performance Variance: {self.stability_metrics.performance_variance:.6f}")
        report.append(f"  Convergence Consist:  {self.stability_metrics.convergence_consistency:.3f}")
        report.append(f"  Solution Diversity:   {self.stability_metrics.solution_diversity:.3f}")
        report.append(f"  Robustness Score:     {self.stability_metrics.robustness_score:.3f}")
        
        # Generation-wise breakdown
        gen_summary = self.get_generation_summary()
        if gen_summary:
            report.append(f"\nGENERATION BREAKDOWN:")
            for gen in sorted(gen_summary.keys()):
                stats = gen_summary[gen]
                report.append(f"  Gen {gen:2d}: Best={stats['best']:8.4f}, Mean={stats['mean']:8.4f}, "
                             f"Std={stats['std']:7.4f}, Count={stats['count']:2d}")
        
        report.append("="*60)
        
        return "\n".join(report)
        
    def get_best_score(self) -> float:
        """Get the best score achieved"""
        return self.metrics.best_score
        
    def get_total_evaluations(self) -> int:
        """Get total number of evaluations performed"""
        return self.metrics.total_evaluator_calls
    
    def debug_profiler_state(self) -> Dict:
        """Get current profiler state for debugging"""
        return {
            'best_score': self.metrics.best_score,
            'scores_history_length': len(self.metrics.scores_history),
            'scores_history_sample': self.metrics.scores_history[-5:] if self.metrics.scores_history else [],
            'successful_evaluations': self.metrics.successful_evaluations,
            'failed_evaluations': self.metrics.failed_evaluations,
            'total_evaluator_calls': self.metrics.total_evaluator_calls,
            'convergence_generation': self.metrics.convergence_generation
        }


class ExperimentComparator:
    """Compare multiple experiments and generate comparative analysis"""
    
    def __init__(self):
        self.experiments = {}
    
    def add_experiment(self, name: str, profiler: ComprehensiveProfiler):
        """Add an experiment for comparison"""
        profiler.calculate_final_metrics()
        
        self.experiments[name] = {
            'profiler': profiler,
            'metrics': {
                'performance': asdict(profiler.metrics),
                'tokens': asdict(profiler.token_metrics),
                'timing': asdict(profiler.timing_metrics),
                'stability': asdict(profiler.stability_metrics),
                'efficiency': profiler.get_efficiency_metrics()
            }
        }
        
    def generate_comparison_table(self) -> Dict:
        """Generate comparison table across all experiments"""
        if not self.experiments:
            return {}
            
        comparison = {
            'experiments': list(self.experiments.keys()),
            'metrics': {
                'best_scores': [],
                'mean_scores': [],
                'total_evaluations': [],
                'success_rates': [],
                'total_costs': [],
                'total_times': [],
                'robustness_scores': [],
                'efficiency_scores': []
            }
        }
        
        for name, exp in self.experiments.items():
            metrics = exp['metrics']
            comparison['metrics']['best_scores'].append(metrics['performance']['best_score'])
            comparison['metrics']['mean_scores'].append(metrics['performance']['mean_score'])
            comparison['metrics']['total_evaluations'].append(metrics['performance']['total_evaluator_calls'])
            comparison['metrics']['success_rates'].append(metrics['stability']['success_rate'])
            comparison['metrics']['total_costs'].append(metrics['tokens']['total_cost_usd'])
            comparison['metrics']['total_times'].append(metrics['timing']['total_duration'])
            comparison['metrics']['robustness_scores'].append(metrics['stability']['robustness_score'])
            comparison['metrics']['efficiency_scores'].append(metrics['efficiency']['evaluations_per_second'])
            
        # Add rankings
        comparison['rankings'] = {}
        for metric_name, values in comparison['metrics'].items():
            if metric_name in ['best_scores', 'mean_scores', 'success_rates', 'robustness_scores', 'efficiency_scores']:
                # Higher is better
                ranking = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
            else:
                # Lower is better (costs, times, evaluations)
                ranking = sorted(enumerate(values), key=lambda x: x[1])
                
            comparison['rankings'][metric_name] = [
                (comparison['experiments'][idx], value, rank+1) 
                for rank, (idx, value) in enumerate(ranking)
            ]
            
        return comparison
        
    def find_best_experiment(self, metric: str = 'best_score') -> str:
        """Find the best experiment based on specified metric"""
        if not self.experiments:
            return None
            
        best_name = None
        best_value = float('-inf') if metric in ['best_score', 'mean_score', 'success_rate', 'robustness_score'] else float('inf')
        
        for name, exp in self.experiments.items():
            if metric == 'best_score':
                value = exp['metrics']['performance']['best_score']
            elif metric == 'mean_score':
                value = exp['metrics']['performance']['mean_score']
            elif metric == 'success_rate':
                value = exp['metrics']['stability']['success_rate']
            elif metric == 'total_cost':
                value = exp['metrics']['tokens']['total_cost_usd']
            elif metric == 'total_time':
                value = exp['metrics']['timing']['total_duration']
            elif metric == 'robustness_score':
                value = exp['metrics']['stability']['robustness_score']
            else:
                continue
                
            if (metric in ['best_score', 'mean_score', 'success_rate', 'robustness_score'] and value > best_value) or \
               (metric in ['total_cost', 'total_time'] and value < best_value):
                best_value = value
                best_name = name
                
        return best_name
        
    def generate_summary_report(self, output_file: str = None) -> Dict:
        """Generate comprehensive comparison report"""
        if not self.experiments:
            return {}
            
        comparison = self.generate_comparison_table()
        
        # Generate summary
        summary = {
            'total_experiments': len(self.experiments),
            'comparison_table': comparison,
            'best_performers': {
                'best_score': self.find_best_experiment('best_score'),
                'mean_score': self.find_best_experiment('mean_score'),
                'lowest_cost': self.find_best_experiment('total_cost'),
                'fastest': self.find_best_experiment('total_time'),
                'most_robust': self.find_best_experiment('robustness_score')
            },
            'recommendations': self._generate_recommendations()
        }
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
        return summary
        
    def _generate_recommendations(self) -> Dict:
        """Generate recommendations based on comparative analysis"""
        if not self.experiments:
            return {}
            
        recommendations = {
            'overall_best': self.find_best_experiment('best_score'),
            'most_cost_effective': self.find_best_experiment('total_cost'),
            'most_reliable': self.find_best_experiment('robustness_score'),
            'analysis': []
        }
        
        # Add specific analysis points
        comparison = self.generate_comparison_table()
        
        # Best performance analysis
        best_scores = comparison['metrics']['best_scores']
        if max(best_scores) - min(best_scores) > 0.1:  # Significant difference
            recommendations['analysis'].append(
                f"Significant performance differences detected (range: {min(best_scores):.4f} to {max(best_scores):.4f})"
            )
            
        # Cost efficiency analysis  
        costs = comparison['metrics']['total_costs']
        if max(costs) / min(costs) > 2:  # 2x cost difference
            recommendations['analysis'].append(
                f"Large cost variations observed (range: ${min(costs):.4f} to ${max(costs):.4f})"
            )
            
        return recommendations 