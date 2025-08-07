"""
Comprehensive Metrics System for TSP Optimization
=================================================

This module provides detailed metrics collection and analysis for TSP (Traveling Salesman Problem)
optimization experiments using LLM-based methods like EoH.

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
    """Core performance metrics for TSP optimization"""
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
    Comprehensive profiler for TSP optimization experiments
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
        # Call base class method for compatibility - but only if we have a proper Function object
        # For now, we'll skip the base class call since we're dealing with raw scores
        # super().log_evaluation(score, generation, evaluation_time, solution_data)
        
        # Enhanced logging
        if score is not None and not np.isnan(score):
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
                'solution': solution_data,
                'timestamp': time.time()
            })
            
    def log_llm_call(self, tokens_used: int, cost: float = 0.0, call_time: float = None,
                     prompt_tokens: int = None, completion_tokens: int = None):
        """Log LLM API call metrics"""
        if not self.track_tokens:
            return
            
        self.token_metrics.total_tokens += tokens_used
        self.token_metrics.total_cost_usd += cost
        self.token_metrics.api_calls += 1
        
        if prompt_tokens:
            self.token_metrics.prompt_tokens += prompt_tokens
        if completion_tokens:
            self.token_metrics.completion_tokens += completion_tokens
            
        if call_time:
            self.timing_metrics.llm_call_time += call_time
            
        # Log individual call
        self.llm_call_logs.append({
            'timestamp': time.time(),
            'tokens': tokens_used,
            'cost': cost,
            'call_time': call_time,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        })
        
    def calculate_final_metrics(self):
        """Calculate final comprehensive metrics"""
        self.timing_metrics.end_time = time.time()
        self.timing_metrics.total_duration = self.timing_metrics.end_time - self.timing_metrics.start_time
        
        # Performance metrics
        if self.metrics.scores_history:
            scores = np.array(self.metrics.scores_history)
            self.metrics.mean_score = np.mean(scores)
            self.metrics.median_score = np.median(scores)
            self.metrics.std_score = np.std(scores)
            
        # Timing metrics
        if self.evaluation_times:
            self.timing_metrics.avg_evaluation_time = np.mean(self.evaluation_times)
            self.timing_metrics.evaluation_time = sum(self.evaluation_times)
            
        # Token metrics
        if self.token_metrics.api_calls > 0:
            self.token_metrics.avg_tokens_per_call = self.token_metrics.total_tokens / self.token_metrics.api_calls
            if self.token_metrics.total_tokens > 0:
                self.token_metrics.cost_per_token = self.token_metrics.total_cost_usd / self.token_metrics.total_tokens
                
        # Stability metrics
        self._calculate_stability_metrics()
        
        self.comp_logger.info("Final comprehensive metrics calculated")
        
    def _calculate_stability_metrics(self):
        """Calculate stability and reliability metrics"""
        if self.metrics.total_evaluator_calls > 0:
            self.stability_metrics.success_rate = self.metrics.successful_evaluations / self.metrics.total_evaluator_calls
            
        if len(self.metrics.scores_history) > 1:
            scores = np.array(self.metrics.scores_history)
            self.stability_metrics.performance_variance = np.var(scores)
            
            # Solution diversity (simplified measure based on score variance)
            if len(self.solution_diversity_history) > 1:
                recent_solutions = self.solution_diversity_history[-10:]  # Last 10 solutions
                recent_scores = [s['score'] for s in recent_solutions if s['score'] is not None]
                if len(recent_scores) > 1:
                    self.stability_metrics.solution_diversity = np.std(recent_scores) / max(np.mean(recent_scores), 1e-8)
                    
        # Convergence consistency
        if self.generation_scores:
            generation_means = []
            for gen, scores in self.generation_scores.items():
                if scores:
                    generation_means.append(np.mean(scores))
            if len(generation_means) > 1:
                # Measure how consistently performance improves
                improvements = np.diff(generation_means)
                positive_improvements = sum(1 for imp in improvements if imp > 0)
                self.stability_metrics.convergence_consistency = positive_improvements / len(improvements)
                
        # Robustness score (composite measure)
        self.stability_metrics.robustness_score = (
            self.stability_metrics.success_rate * 0.4 +
            self.stability_metrics.convergence_consistency * 0.3 +
            (1 - min(self.stability_metrics.performance_variance / max(abs(self.metrics.mean_score), 1), 1)) * 0.3
        )
        
    def get_generation_summary(self) -> Dict[int, Dict]:
        """Get summary statistics for each generation"""
        summary = {}
        for generation, scores in self.generation_scores.items():
            if scores:
                summary[generation] = {
                    'count': len(scores),
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'median': np.median(scores)
                }
        return summary
        
    def get_efficiency_metrics(self) -> Dict:
        """Get efficiency-related metrics"""
        return {
            'evaluations_per_second': self.metrics.total_evaluator_calls / max(self.timing_metrics.total_duration, 1),
            'score_per_evaluation': self.metrics.best_score / max(self.metrics.total_evaluator_calls, 1),
            'score_per_second': self.metrics.best_score / max(self.timing_metrics.total_duration, 1),
            'score_per_dollar': self.metrics.best_score / max(self.token_metrics.total_cost_usd, 1e-8),
            'cost_efficiency': self.token_metrics.total_cost_usd / max(abs(self.metrics.best_score), 1e-8)
        }
        
    def get_stability_metrics(self) -> Dict:
        """Get stability-related metrics"""
        return asdict(self.stability_metrics)
        
    def export_detailed_log(self, filepath: str):
        """Export detailed logs to file"""
        detailed_log = {
            'metadata': {
                'experiment_timestamp': datetime.fromtimestamp(self.timing_metrics.start_time).isoformat(),
                'log_dir': self.log_dir,
                'tracking_settings': {
                    'track_tokens': self.track_tokens,
                    'log_style': self.log_style
                }
            },
            'performance_metrics': asdict(self.metrics),
            'token_metrics': asdict(self.token_metrics),
            'timing_metrics': asdict(self.timing_metrics),
            'stability_metrics': asdict(self.stability_metrics),
            'generation_summary': self.get_generation_summary(),
            'efficiency_metrics': self.get_efficiency_metrics(),
            'detailed_logs': {
                'llm_calls': self.llm_call_logs,
                'solution_diversity': self.solution_diversity_history,
                'evaluation_times': self.evaluation_times
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(detailed_log, f, indent=2)
            
        self.comp_logger.info(f"Detailed log exported to {filepath}")
        
    def save_metrics(self, filepath: str):
        """Save comprehensive metrics to file"""
        self.export_detailed_log(filepath)
        
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        report = []
        report.append("=" * 60)
        report.append("TSP Optimization - Comprehensive Metrics Summary")
        report.append("=" * 60)
        
        # Performance summary
        report.append(f"\n📊 Performance Metrics:")
        report.append(f"  Best Score: {self.metrics.best_score:.3f}")
        report.append(f"  Mean Score: {self.metrics.mean_score:.3f} ± {self.metrics.std_score:.3f}")
        report.append(f"  Score Range: [{self.metrics.worst_score:.3f}, {self.metrics.best_score:.3f}]")
        report.append(f"  Convergence Generation: {self.metrics.convergence_generation}")
        
        # Evaluation summary
        report.append(f"\n🔧 Evaluation Summary:")
        report.append(f"  Total Evaluations: {self.metrics.total_evaluator_calls}")
        report.append(f"  Successful: {self.metrics.successful_evaluations}")
        report.append(f"  Failed: {self.metrics.failed_evaluations}")
        report.append(f"  Success Rate: {self.stability_metrics.success_rate:.3f}")
        
        # Cost summary
        if self.track_tokens:
            report.append(f"\n💰 Cost Analysis:")
            report.append(f"  Total Tokens: {self.token_metrics.total_tokens:,}")
            report.append(f"  Total Cost: ${self.token_metrics.total_cost_usd:.6f}")
            report.append(f"  API Calls: {self.token_metrics.api_calls}")
            report.append(f"  Avg Tokens/Call: {self.token_metrics.avg_tokens_per_call:.1f}")
            
        # Timing summary
        report.append(f"\n⏱️ Timing Analysis:")
        report.append(f"  Total Duration: {self.timing_metrics.total_duration:.1f}s")
        report.append(f"  Avg Evaluation Time: {self.timing_metrics.avg_evaluation_time:.3f}s")
        
        # Efficiency summary
        efficiency = self.get_efficiency_metrics()
        report.append(f"\n⚡ Efficiency Metrics:")
        report.append(f"  Evaluations/Second: {efficiency['evaluations_per_second']:.2f}")
        report.append(f"  Score/Evaluation: {efficiency['score_per_evaluation']:.4f}")
        if self.track_tokens and self.token_metrics.total_cost_usd > 0:
            report.append(f"  Score/Dollar: {efficiency['score_per_dollar']:.1f}")
            
        # Stability summary
        report.append(f"\n🎯 Stability Analysis:")
        report.append(f"  Solution Diversity: {self.stability_metrics.solution_diversity:.3f}")
        report.append(f"  Convergence Consistency: {self.stability_metrics.convergence_consistency:.3f}")
        report.append(f"  Robustness Score: {self.stability_metrics.robustness_score:.3f}")
        
        return "\n".join(report)

    # Override/implement base class abstract methods if necessary
    def get_best_score(self) -> float:
        """Get the best score - implements base class interface"""
        return self.metrics.best_score

    def get_total_evaluations(self) -> int:
        """Get total number of evaluations - implements base class interface"""
        return self.metrics.total_evaluator_calls


class ExperimentComparator:
    """
    Compare multiple TSP optimization experiments
    """
    
    def __init__(self):
        self.experiments = {}
        
    def add_experiment(self, name: str, profiler: ComprehensiveProfiler):
        """Add an experiment for comparison"""
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
        """Generate comparison table across experiments"""
        if not self.experiments:
            return {}
            
        comparison = {
            'experiments': list(self.experiments.keys()),
            'metrics_comparison': {}
        }
        
        # Key metrics for comparison
        key_metrics = [
            ('best_score', 'performance'),
            ('mean_score', 'performance'),
            ('std_score', 'performance'),
            ('total_evaluator_calls', 'performance'),
            ('success_rate', 'stability'),
            ('total_cost_usd', 'tokens'),
            ('total_duration', 'timing'),
            ('score_per_dollar', 'efficiency'),
            ('robustness_score', 'stability')
        ]
        
        for metric_name, category in key_metrics:
            comparison['metrics_comparison'][metric_name] = {}
            values = []
            
            for exp_name, exp_data in self.experiments.items():
                if category in exp_data['metrics']:
                    value = exp_data['metrics'][category].get(metric_name, None)
                    comparison['metrics_comparison'][metric_name][exp_name] = value
                    if value is not None:
                        values.append(value)
                        
            # Calculate statistics across experiments
            if values:
                comparison['metrics_comparison'][metric_name]['_stats'] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    'range': max(values) - min(values)
                }
                
        return comparison
        
    def find_best_experiment(self, metric: str = 'best_score') -> str:
        """Find the best performing experiment by a given metric"""
        best_exp = None
        best_value = float('-inf')
        
        for exp_name, exp_data in self.experiments.items():
            # Search for metric in different categories
            value = None
            for category in exp_data['metrics'].values():
                if isinstance(category, dict) and metric in category:
                    value = category[metric]
                    break
                    
            if value is not None and value > best_value:
                best_value = value
                best_exp = exp_name
                
        return best_exp
        
    def generate_summary_report(self, output_file: str = None) -> Dict:
        """Generate comprehensive comparison report"""
        if not self.experiments:
            return {}
            
        comparison = self.generate_comparison_table()
        
        # Add best experiment analysis
        best_by_score = self.find_best_experiment('best_score')
        best_by_efficiency = self.find_best_experiment('score_per_dollar')
        best_by_stability = self.find_best_experiment('robustness_score')
        
        summary_report = {
            'comparison_timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.experiments),
            'comparison_table': comparison,
            'best_experiments': {
                'by_score': best_by_score,
                'by_efficiency': best_by_efficiency,
                'by_stability': best_by_stability
            },
            'recommendations': self._generate_recommendations()
        }
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(summary_report, f, indent=2)
                
        return summary_report
        
    def _generate_recommendations(self) -> Dict:
        """Generate recommendations based on experiment comparison"""
        if not self.experiments:
            return {}
            
        recommendations = {
            'performance': "No clear recommendation available",
            'cost_efficiency': "No clear recommendation available", 
            'stability': "No clear recommendation available",
            'overall': "No clear recommendation available"
        }
        
        try:
            # Analyze patterns in the data
            scores = []
            costs = []
            stabilities = []
            
            for exp_name, exp_data in self.experiments.items():
                perf = exp_data['metrics'].get('performance', {})
                tokens = exp_data['metrics'].get('tokens', {})
                stability = exp_data['metrics'].get('stability', {})
                
                if perf.get('best_score') is not None:
                    scores.append((exp_name, perf['best_score']))
                if tokens.get('total_cost_usd') is not None:
                    costs.append((exp_name, tokens['total_cost_usd']))
                if stability.get('robustness_score') is not None:
                    stabilities.append((exp_name, stability['robustness_score']))
                    
            if scores:
                best_score_exp = max(scores, key=lambda x: x[1])
                recommendations['performance'] = f"Best performance: {best_score_exp[0]} (score: {best_score_exp[1]:.3f})"
                
            if costs:
                lowest_cost_exp = min(costs, key=lambda x: x[1])
                recommendations['cost_efficiency'] = f"Most cost-effective: {lowest_cost_exp[0]} (cost: ${lowest_cost_exp[1]:.6f})"
                
            if stabilities:
                most_stable_exp = max(stabilities, key=lambda x: x[1])
                recommendations['stability'] = f"Most stable: {most_stable_exp[0]} (robustness: {most_stable_exp[1]:.3f})"
                
        except Exception as e:
            logging.warning(f"Error generating recommendations: {e}")
            
        return recommendations 