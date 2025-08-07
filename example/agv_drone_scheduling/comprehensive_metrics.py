import numpy as np
import json
import os
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import sys # For stderr

# Attempt to import ProfilerBase and Function
try:
    from llm4ad.tools.profiler.profile import ProfilerBase
    from llm4ad.base import Function
    LLM4AD_IMPORTS_AVAILABLE = True
except ImportError:
    LLM4AD_IMPORTS_AVAILABLE = False
    print("Warning: llm4ad.tools.profiler.profile.ProfilerBase or llm4ad.base.Function not found. Using mocks.", file=sys.stderr)
    class ProfilerBase: # Mock version
        def __init__(self, log_dir: Optional[str] = None, *args, **kwargs):
            self._cur_best_program_score = float('-inf')
            self._log_dir = log_dir # Mock should store log_dir in _log_dir
            self._samples_json_dir = None
            if self._log_dir: # If mock has a log_dir, try to set up samples dir
                self._samples_json_dir = os.path.join(self._log_dir, 'samples')
                # os.makedirs(self._samples_json_dir, exist_ok=True) # Not strictly needed for mock functionality if not writing files

        def register_function(self, func_obj): pass
        def _write_json(self, func_obj, record_type): pass
        def _create_log_path(self):
            if self._log_dir: # Mock's _create_log_path also checks _log_dir
                self._samples_json_dir = os.path.join(self._log_dir, 'samples')
                if self._samples_json_dir:
                    os.makedirs(self._samples_json_dir, exist_ok=True)
        def record_parameters(self, *args, **kwargs): pass

    class Function: # Mock version
        def __init__(self, value, score, sample_time, evaluate_time, **kwargs): pass


@dataclass
class MetricsSnapshot:
    timestamp: float
    sample_order: int
    score: float
    sample_time: float
    evaluate_time: float
    token_usage: Optional[Dict[str, int]] = None
    llm_calls: int = 0
    evaluator_calls: int = 0
    generation: int = 0
    population_index: int = 0
    
@dataclass  
class ComprehensiveMetrics:
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_llm_calls: int = 0
    total_sample_time: float = 0.0
    total_evaluate_time: float = 0.0
    total_evaluator_calls: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    best_score: float = float('-inf')
    worst_score: float = float('inf')
    mean_score: float = 0.0
    std_score: float = 0.0
    median_score: float = 0.0
    scores: List[float] = field(default_factory=list)
    convergence_generation: Optional[int] = None
    improvement_generations: List[int] = field(default_factory=list)
    stagnation_count: int = 0
    scores_per_generation: List[float] = field(default_factory=list)
    best_scores_per_generation: List[float] = field(default_factory=list)
    time_per_generation: List[float] = field(default_factory=list)
    tokens_per_generation: List[int] = field(default_factory=list)
    score_variance_per_generation: List[float] = field(default_factory=list)
    success_rate_per_generation: List[float] = field(default_factory=list)
    score_per_token: float = 0.0
    score_per_time: float = 0.0
    score_per_evaluator_call: float = 0.0

class ComprehensiveProfiler(ProfilerBase):
    def __init__(self, log_dir: str = "metrics_logs", track_tokens: bool = True,
                 evaluation_name: str = 'ComprehensiveProblem',
                 method_name: str = 'ComprehensiveMethod',
                 initial_num_samples: int = 0):
        
        # Call superclass constructor
        # ProfilerBase is expected to initialize self._log_dir
        super().__init__(log_dir=log_dir,
                         evaluation_name=evaluation_name,
                         method_name=method_name,
                         initial_num_samples=initial_num_samples,
                         log_style='complex',
                         create_random_path=False) # Crucial

        # After super().__init__(), self._log_dir should be set by ProfilerBase.
        # However, if using a mock, or if ProfilerBase's logic is different, 
        # ensure self._log_dir is what ComprehensiveProfiler expects (the passed 'log_dir').
        # If ProfilerBase did set _log_dir but to something different than `log_dir` passed here,
        # we prioritize the `log_dir` passed to ComprehensiveProfiler for its own setup.
        current_super_log_dir = getattr(self, '_log_dir', None)

        if current_super_log_dir != log_dir:
            if current_super_log_dir is not None:
                print(f"Warning: ProfilerBase set self._log_dir to '{current_super_log_dir}', but ComprehensiveProfiler constructor received '{log_dir}'. Overriding self._log_dir to '{log_dir}'.", file=sys.stderr)
            else:
                print(f"Warning: self._log_dir was not set by ProfilerBase or is None. Setting self._log_dir to '{log_dir}' from ComprehensiveProfiler args.", file=sys.stderr)
            self._log_dir = log_dir
        
        # Ensure the main log directory (self._log_dir) exists.
        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)
        else:
            # This case should be rare if the above logic correctly sets self._log_dir from input `log_dir`.
            print("Critical Error: self._log_dir is None after attempting to set it. Defaulting to './default_metrics_logs_critical_error' for path creation.", file=sys.stderr)
            self._log_dir = "./default_metrics_logs_critical_error"
            os.makedirs(self._log_dir, exist_ok=True)

        # Directly set up self._samples_json_dir using self._log_dir
        # This is the directory ProfilerBase uses for its _write_json method (e.g., for samples_best.json).
        # This ensures it is set correctly even if ProfilerBase._create_log_path() isn't called or behaves differently.
        if self._log_dir: 
            self._samples_json_dir = os.path.join(self._log_dir, 'samples')
            os.makedirs(self._samples_json_dir, exist_ok=True)
        else:
            self._samples_json_dir = None # Should not be reached
            print("Critical Warning: self._samples_json_dir is None as self._log_dir is unavailable. ProfilerBase JSON writing might fail.", file=sys.stderr)

        # Initialize ComprehensiveProfiler specific attributes
        self.track_tokens = track_tokens
        self.metrics = ComprehensiveMetrics()
        self.snapshots: List[MetricsSnapshot] = []
        self.generation_data: Dict[int, List[MetricsSnapshot]] = defaultdict(list)
        self.lock = threading.Lock()
        
        # Initialize log file paths for ComprehensiveProfiler's own JSON outputs, using self._log_dir
        self.metrics_file = os.path.join(self._log_dir, "comprehensive_metrics.json")
        self.detailed_log = os.path.join(self._log_dir, "detailed_metrics.jsonl")
            
    def record_llm_call(self, prompt_tokens: int = 0, completion_tokens: int = 0, 
                       total_tokens: int = 0, call_time: float = 0.0):
        with self.lock:
            self.metrics.total_llm_calls += 1
            self.metrics.input_tokens += prompt_tokens
            self.metrics.output_tokens += completion_tokens
            self.metrics.total_tokens += total_tokens or (prompt_tokens + completion_tokens)
            self.metrics.total_sample_time += call_time
            
    def record_evaluation(self, score: float, evaluation_time: float, 
                         sample_order: int, 
                         generation: int = 0, 
                         population_index: int = 0, success: bool = True,
                         function_str: Optional[str] = None, 
                         sample_time: float = 0.0 
                         ):
        with self.lock:
            self.metrics.total_evaluator_calls += 1
            self.metrics.total_evaluate_time += evaluation_time
            
            snapshot_token_usage = None 

            # Treat infinite scores as failed evaluations
            if success and score is not None and np.isfinite(score) and not np.isnan(score):
                self.metrics.successful_evaluations += 1
                self.metrics.scores.append(score)
                
                if score > self.metrics.best_score:
                    old_best_cm = self.metrics.best_score
                    self.metrics.best_score = score
                    if old_best_cm == float('-inf') or self.metrics.convergence_generation is None:
                        self.metrics.convergence_generation = generation
                    if old_best_cm != float('-inf') and score > old_best_cm:
                         self.metrics.improvement_generations.append(generation)
                         self.metrics.stagnation_count = 0
                    elif old_best_cm != float('-inf') and score <= old_best_cm :
                         self.metrics.stagnation_count +=1

                if score < self.metrics.worst_score:
                    self.metrics.worst_score = score
                
                snapshot = MetricsSnapshot(
                    timestamp=time.time(),
                    sample_order=sample_order,
                    score=score,
                    sample_time=sample_time, 
                    evaluate_time=evaluation_time,
                    token_usage=snapshot_token_usage, 
                    llm_calls=0, 
                    evaluator_calls=1, 
                    generation=generation,
                    population_index=population_index
                )
                self.snapshots.append(snapshot)
                self.generation_data[generation].append(snapshot)
                self._write_detailed_log(snapshot)
                
            else:
                # Failed evaluation (either success=False or infinite/NaN score)
                self.metrics.failed_evaluations += 1
                
    def update_generation_metrics(self, generation: int):
        if generation not in self.generation_data:
            return
        gen_snapshots = self.generation_data[generation]
        if not gen_snapshots:
            return
            
        # Filter out None, NaN and infinite scores
        gen_scores = [s.score for s in gen_snapshots if s.score is not None and np.isfinite(s.score) and not np.isnan(s.score)] 
        if not gen_scores: 
            self.metrics.scores_per_generation.append(float('nan')) # Use NaN for undefined mean/max
            self.metrics.best_scores_per_generation.append(float('-inf')) # Or NaN depending on preference
            self.metrics.score_variance_per_generation.append(float('nan')) 
        else:
            self.metrics.scores_per_generation.append(np.mean(gen_scores) if gen_scores else float('nan'))
            self.metrics.best_scores_per_generation.append(max(gen_scores) if gen_scores else float('-inf'))
            self.metrics.score_variance_per_generation.append(np.var(gen_scores) if len(gen_scores) > 1 else 0.0) # Var is 0 for single point

        gen_eval_times = [s.evaluate_time for s in gen_snapshots]
        self.metrics.time_per_generation.append(sum(gen_eval_times))
        
        successful_in_gen = len(gen_scores)  # Now based on finite scores only
        total_attempts_in_gen = len(gen_snapshots) 
        self.metrics.success_rate_per_generation.append(
            successful_in_gen / total_attempts_in_gen if total_attempts_in_gen > 0 else 0
        )
        
        if generation > 0 and self.metrics.best_scores_per_generation and len(self.metrics.best_scores_per_generation) > 1:
            # Check if the current generation's best is not better than previous for stagnation count (handled in record_evaluation based on overall best)
            pass 

    def calculate_final_metrics(self):
        if self.metrics.scores:
            # Filter out NaN and infinite values (-inf, +inf)
            numeric_scores = [s for s in self.metrics.scores if isinstance(s, (int, float)) and np.isfinite(s) and not np.isnan(s)]
            if numeric_scores:
                self.metrics.mean_score = np.mean(numeric_scores)
                self.metrics.std_score = np.std(numeric_scores)
                self.metrics.median_score = np.median(numeric_scores)
                self.metrics.best_score = max(numeric_scores) 
                self.metrics.worst_score = min(numeric_scores)
            else: # No valid finite numeric scores
                self.metrics.mean_score = float('nan')
                self.metrics.std_score = float('nan')
                self.metrics.median_score = float('nan')
                self.metrics.best_score = float('-inf') 
                self.metrics.worst_score = float('inf')
        else: # self.metrics.scores is empty
            self.metrics.mean_score = float('nan')
            self.metrics.std_score = float('nan')
            self.metrics.median_score = float('nan')
            self.metrics.best_score = float('-inf') 
            self.metrics.worst_score = float('inf')
            
        if np.isnan(self.metrics.best_score) or self.metrics.best_score == float('-inf'):
            self.metrics.score_per_token = 0.0
            self.metrics.score_per_time = 0.0
            self.metrics.score_per_evaluator_call = 0.0
        else:
            self.metrics.score_per_token = self.metrics.best_score / self.metrics.total_tokens if self.metrics.total_tokens > 0 else 0.0
            total_time = self.metrics.total_sample_time + self.metrics.total_evaluate_time
            self.metrics.score_per_time = self.metrics.best_score / total_time if total_time > 0 else 0.0
            self.metrics.score_per_evaluator_call = self.metrics.best_score / self.metrics.total_evaluator_calls if self.metrics.total_evaluator_calls > 0 else 0.0
    
    def get_stability_metrics(self) -> Dict[str, Any]: # Changed to Any for NaN
        if not self.metrics.scores : self.calculate_final_metrics()

        if not self.metrics.scores or all(np.isnan(s) for s in self.metrics.scores if isinstance(s, float)):
            return {
                "coefficient_of_variation": float('nan'),
                "improvement_rate": 0.0,
                "convergence_stability": 0.0,
                "stagnation_ratio": 0.0,
                "success_rate": 0.0
            }
            
        cv = self.metrics.std_score / self.metrics.mean_score if self.metrics.mean_score != 0 and not np.isnan(self.metrics.mean_score) else float('nan')
        if self.metrics.mean_score == 0 and self.metrics.std_score == 0: cv = 0.0 
        
        improvements = len(self.metrics.improvement_generations)
        # Filter out NaN and infinite values from best_scores_per_generation before calculating total_generations_with_data
        valid_best_scores_gen = [s for s in self.metrics.best_scores_per_generation if np.isfinite(s) and not np.isnan(s)]
        total_generations_with_data = len(valid_best_scores_gen)
        improvement_rate = improvements / total_generations_with_data if total_generations_with_data > 0 else 0
        
        convergence_stability = 0.0
        if total_generations_with_data >= 5:
            last_5_best = valid_best_scores_gen[-5:]
            var_last_5 = np.var(last_5_best) if last_5_best else float('nan')
            convergence_stability = 1.0 / (1.0 + var_last_5) if not np.isnan(var_last_5) and var_last_5 > 1e-9 else (1.0 if not np.isnan(var_last_5) else 0.0)
        elif total_generations_with_data > 0: 
             var_all_best = np.var(valid_best_scores_gen) if valid_best_scores_gen else float('nan')
             convergence_stability = 1.0 / (1.0 + var_all_best) if not np.isnan(var_all_best) and var_all_best > 1e-9 else (1.0 if not np.isnan(var_all_best) else 0.0)

        success_rate = self.metrics.successful_evaluations / self.metrics.total_evaluator_calls if self.metrics.total_evaluator_calls > 0 else 0
        return {
            "coefficient_of_variation": cv,
            "improvement_rate": improvement_rate,
            "convergence_stability": convergence_stability,
            "stagnation_ratio": self.metrics.stagnation_count / total_generations_with_data if total_generations_with_data > 0 else 0,
            "success_rate": success_rate
        }
    
    def get_efficiency_metrics(self) -> Dict[str, Any]: # Changed to Any for NaN/inf
        if np.isnan(self.metrics.best_score) or self.metrics.best_score == float('-inf'): 
            self.calculate_final_metrics() # Recalculate to be sure
            # If still invalid, return metrics indicating no/poor performance
            if np.isnan(self.metrics.best_score) or self.metrics.best_score == float('-inf'):
                return {
                    "tokens_per_best_score": float('inf'),
                    "time_per_best_score": float('inf'),
                    "evaluations_per_best_score": float('inf'),
                    "average_time_per_evaluation": self.metrics.total_evaluate_time / self.metrics.total_evaluator_calls if self.metrics.total_evaluator_calls > 0 else 0,
                    "average_tokens_per_evaluation": self.metrics.total_tokens / self.metrics.total_evaluator_calls if self.metrics.total_evaluator_calls > 0 else 0,
                    "llm_call_efficiency": 0.0
                }

        denominator_score = self.metrics.best_score
        if denominator_score == 0: denominator_score = 1e-9 # Avoid division by zero if score is exactly 0
        elif np.isnan(denominator_score): denominator_score = float('nan') # Propagate NaN

        avg_time_per_eval = self.metrics.total_evaluate_time / self.metrics.total_evaluator_calls if self.metrics.total_evaluator_calls > 0 else 0
        avg_tokens_per_eval = self.metrics.total_tokens / self.metrics.total_evaluator_calls if self.metrics.total_evaluator_calls > 0 else 0
        llm_call_eff = self.metrics.best_score / self.metrics.total_llm_calls if self.metrics.total_llm_calls > 0 and not np.isnan(self.metrics.best_score) else 0.0

        if np.isnan(denominator_score):
            return {
                "tokens_per_best_score": float('nan'), "time_per_best_score": float('nan'), "evaluations_per_best_score": float('nan'),
                "average_time_per_evaluation": avg_time_per_eval, "average_tokens_per_evaluation": avg_tokens_per_eval, "llm_call_efficiency": llm_call_eff
            }

        return {
            "tokens_per_best_score": self.metrics.total_tokens / denominator_score if denominator_score != 0 else float('inf'),
            "time_per_best_score": (self.metrics.total_sample_time + self.metrics.total_evaluate_time) / denominator_score if denominator_score != 0 else float('inf'),
            "evaluations_per_best_score": self.metrics.total_evaluator_calls / denominator_score if denominator_score != 0 else float('inf'),
            "average_time_per_evaluation": avg_time_per_eval,
            "average_tokens_per_evaluation": avg_tokens_per_eval,
            "llm_call_efficiency": llm_call_eff
        }
        
    def generate_comparison_report(self, other_runs: List['ComprehensiveProfiler'] = None) -> Dict[str, Any]:
        self.calculate_final_metrics()
        
        # Helper to safely get metric, defaulting to NaN if not suitable for division or comparison
        def safe_metric(value, for_division=False):
            if isinstance(value, (int, float)) and not np.isnan(value):
                if for_division and value == 0: return 1e-9 # Avoid division by zero, use small epsilon
                return value
            return np.nan # Default to NaN if not valid number

        current_basic_metrics = {
            "best_score": safe_metric(self.metrics.best_score),
            "mean_score": safe_metric(self.metrics.mean_score),
            "std_score": safe_metric(self.metrics.std_score),
            "total_tokens": self.metrics.total_tokens,
            "total_evaluator_calls": self.metrics.total_evaluator_calls,
            "total_time": self.metrics.total_evaluate_time + self.metrics.total_sample_time
        }

        report = {
            "current_run": {
                "basic_metrics": current_basic_metrics,
                "stability_metrics": self.get_stability_metrics(),
                "efficiency_metrics": self.get_efficiency_metrics()
            }
        }
        
        if other_runs:
            comparisons = []
            for i, other in enumerate(other_runs):
                other.calculate_final_metrics()
                
                self_best_score = safe_metric(self.metrics.best_score)
                other_best_score = safe_metric(other.metrics.best_score)
                
                score_improvement = (self_best_score - other_best_score) / max(abs(safe_metric(other_best_score, for_division=True)), 1e-8) \
                                    if not (np.isnan(self_best_score) or np.isnan(other_best_score)) else np.nan

                self_total_tokens = safe_metric(self.metrics.total_tokens, for_division=True)
                other_total_tokens = safe_metric(other.metrics.total_tokens, for_division=True)
                token_efficiency_ratio = (other_total_tokens / self_total_tokens) -1 if self_total_tokens > 1e-9 else np.nan # Higher is better

                self_total_eval_time = safe_metric(self.metrics.total_evaluate_time, for_division=True)
                other_total_eval_time = safe_metric(other.metrics.total_evaluate_time, for_division=True)
                time_efficiency_ratio = (other_total_eval_time / self_total_eval_time) -1 if self_total_eval_time > 1e-9 else np.nan # Higher is better
                
                self_cv = safe_metric(self.get_stability_metrics().get("coefficient_of_variation"))
                other_cv = safe_metric(other.get_stability_metrics().get("coefficient_of_variation"))
                cv_improvement = other_cv - self_cv # Lower CV is better, so positive means improvement
                if np.isnan(self_cv) or np.isnan(other_cv): cv_improvement = np.nan

                comparison = {
                    "run_id": i,
                    "score_improvement_percentage": score_improvement * 100 if not np.isnan(score_improvement) else np.nan,
                    "token_efficiency_improvement_percentage": token_efficiency_ratio * 100 if not np.isnan(token_efficiency_ratio) else np.nan,
                    "time_efficiency_improvement_percentage": time_efficiency_ratio * 100 if not np.isnan(time_efficiency_ratio) else np.nan,
                    "stability_cv_change": cv_improvement
                }
                comparisons.append(comparison)
            report["comparisons"] = comparisons
            
        return report
    
    def save_metrics(self, filename: str = None):
        self.calculate_final_metrics() 
        
        actual_filename = filename if filename else self.metrics_file
        if not actual_filename:
            # This case implies self.metrics_file was also None, likely due to self._log_dir being None.
            print("Critical Error: Cannot save metrics. Log directory and filename are not set.", file=sys.stderr)
            return
            
        metrics_dict = {
            "timestamp": time.time(),
            "basic_metrics": {
                "total_tokens": self.metrics.total_tokens,
                "input_tokens": self.metrics.input_tokens,
                "output_tokens": self.metrics.output_tokens,
                "total_llm_calls": self.metrics.total_llm_calls,
                "total_evaluator_calls": self.metrics.total_evaluator_calls,
                "successful_evaluations": self.metrics.successful_evaluations,
                "failed_evaluations": self.metrics.failed_evaluations,
                "total_sample_time": self.metrics.total_sample_time,
                "total_evaluate_time": self.metrics.total_evaluate_time
            },
            "quality_metrics": {
                "best_score": self.metrics.best_score,
                "worst_score": self.metrics.worst_score,
                "mean_score": self.metrics.mean_score,
                "std_score": self.metrics.std_score,
                "median_score": self.metrics.median_score
            },
            "convergence_metrics": {
                "convergence_generation": self.metrics.convergence_generation,
                "improvement_generations": self.metrics.improvement_generations,
                "stagnation_count": self.metrics.stagnation_count
            },
            # Use the safe_metric approach for metrics that might be NaN/inf for JSON
            "efficiency_metrics": {k: (v if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else str(v)) for k,v in self.get_efficiency_metrics().items()},
            "stability_metrics": {k: (v if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else str(v)) for k,v in self.get_stability_metrics().items()},
            "cost_benefit_metrics": {
                "score_per_token": self.metrics.score_per_token if not (isinstance(self.metrics.score_per_token, float) and (np.isnan(self.metrics.score_per_token) or np.isinf(self.metrics.score_per_token))) else str(self.metrics.score_per_token),
                "score_per_time": self.metrics.score_per_time if not (isinstance(self.metrics.score_per_time, float) and (np.isnan(self.metrics.score_per_time) or np.isinf(self.metrics.score_per_time))) else str(self.metrics.score_per_time),
                "score_per_evaluator_call": self.metrics.score_per_evaluator_call if not (isinstance(self.metrics.score_per_evaluator_call, float) and (np.isnan(self.metrics.score_per_evaluator_call) or np.isinf(self.metrics.score_per_evaluator_call))) else str(self.metrics.score_per_evaluator_call),
            },
            "generation_trends": {
                "scores_per_generation": [s if not (isinstance(s, float) and np.isnan(s)) else 'NaN' for s in self.metrics.scores_per_generation],
                "best_scores_per_generation": [s if not (isinstance(s, float) and np.isnan(s)) else 'NaN' for s in self.metrics.best_scores_per_generation],
                "time_per_generation": self.metrics.time_per_generation,
                "score_variance_per_generation": [s if not (isinstance(s, float) and np.isnan(s)) else 'NaN' for s in self.metrics.score_variance_per_generation],
                "success_rate_per_generation": self.metrics.success_rate_per_generation
            }
        }
        
        try:
            with open(actual_filename, 'w') as f:
                # Custom JSON encoder to handle NaN/inf might be better, but str() is a simple fallback
                json.dump(metrics_dict, f, indent=2, default=lambda o: str(o) if isinstance(o, float) and (np.isnan(o) or np.isinf(o)) else o)
        except Exception as e:
            print(f"Error saving metrics to {actual_filename}: {e}", file=sys.stderr)
            
    def _write_detailed_log(self, snapshot: MetricsSnapshot):
        log_entry = {
            "timestamp": snapshot.timestamp,
            "sample_order": snapshot.sample_order,
            "generation": snapshot.generation,
            "population_index": snapshot.population_index,
            "score": snapshot.score,
            "sample_time": snapshot.sample_time,
            "evaluate_time": snapshot.evaluate_time,
            "cumulative_tokens": self.metrics.total_tokens,
            "cumulative_llm_calls": self.metrics.total_llm_calls,
            "cumulative_evaluator_calls": self.metrics.total_evaluator_calls
        }
        
        actual_detailed_log_path = self.detailed_log
        if not actual_detailed_log_path: 
            if self._log_dir:
                actual_detailed_log_path = os.path.join(self._log_dir, "detailed_metrics.jsonl")
            else:
                print("Critical Error: Cannot write detailed log. Log directory and detailed_log path are not set.", file=sys.stderr)
                return
        try:
            with open(actual_detailed_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Error writing detailed log to {actual_detailed_log_path}: {e}", file=sys.stderr)

class ExperimentComparator:
    def __init__(self):
        self.experiments: Dict[str, ComprehensiveProfiler] = {}
        
    def add_experiment(self, name: str, profiler: ComprehensiveProfiler):
        self.experiments[name] = profiler
        
    def compare_all(self) -> Dict[str, Any]:
        if len(self.experiments) < 2:
            return {"error": "Need at least 2 experiments to compare"}
            
        results = {}
        
        for name, profiler in self.experiments.items():
            profiler.calculate_final_metrics() # Ensure metrics are calculated
            # Store metrics safely, converting potential NaN/inf from calculations to string for JSON, or keeping as numbers
            results[name] = {
                "best_score": profiler.metrics.best_score if not (isinstance(profiler.metrics.best_score, float) and (np.isnan(profiler.metrics.best_score) or np.isinf(profiler.metrics.best_score))) else str(profiler.metrics.best_score),
                "mean_score": profiler.metrics.mean_score if not (isinstance(profiler.metrics.mean_score, float) and (np.isnan(profiler.metrics.mean_score) or np.isinf(profiler.metrics.mean_score))) else str(profiler.metrics.mean_score),
                "std_score": profiler.metrics.std_score if not (isinstance(profiler.metrics.std_score, float) and (np.isnan(profiler.metrics.std_score) or np.isinf(profiler.metrics.std_score))) else str(profiler.metrics.std_score),
                "total_tokens": profiler.metrics.total_tokens,
                "total_evaluator_calls": profiler.metrics.total_evaluator_calls,
                "convergence_generation": profiler.metrics.convergence_generation,
                "success_rate": profiler.get_stability_metrics().get('success_rate') # Get from calculated stability metrics
            }
        
        metrics_for_ranking = ["best_score", "mean_score", "success_rate"]
        cost_metrics = ["total_tokens", "total_evaluator_calls"]
        
        rankings = {}

        def get_metric_for_sort(metric_val):
            if isinstance(metric_val, str):
                try: return float(metric_val) # Convert 'NaN', 'inf' back for sorting if possible
                except ValueError: return float('-inf') # Treat unparseable strings as worst
            if metric_val is None: return float('-inf') # Treat None as worst
            return metric_val

        for metric in metrics_for_ranking:
            valid_experiments = []
            for name_exp, _ in self.experiments.items(): # Iterate over keys to access results dict
                metric_value = results[name_exp].get(metric)
                # Allow sorting if it can be converted to float, or is already a number
                if isinstance(metric_value, (int, float)) or isinstance(metric_value, str):
                     valid_experiments.append((name_exp, metric_value))
                else:
                    print(f"Warning: Experiment '{name_exp}' has unhandled type for metric '{metric}': {type(metric_value)}. Excluding from ranking.", file=sys.stderr)
            
            if valid_experiments:
                sorted_experiments = sorted(valid_experiments, 
                                          key=lambda item: get_metric_for_sort(item[1]), 
                                          reverse=True) # Higher is better for these metrics
                rankings[metric] = [item[0] for item in sorted_experiments]
            else:
                rankings[metric] = []

        for metric in cost_metrics:
            valid_experiments = []
            for name_exp, _ in self.experiments.items():
                metric_value = results[name_exp].get(metric)
                if isinstance(metric_value, (int, float)) or isinstance(metric_value, str):
                     valid_experiments.append((name_exp, metric_value))
                else:
                    print(f"Warning: Experiment '{name_exp}' has unhandled type for cost metric '{metric}': {type(metric_value)}. Excluding from ranking.", file=sys.stderr)

            if valid_experiments:
                sorted_experiments = sorted(valid_experiments, 
                                      key=lambda item: get_metric_for_sort(item[1]),
                                      reverse=False) # Lower is better for cost metrics
                rankings[f"{metric}_efficiency"] = [item[0] for item in sorted_experiments]
            else:
                rankings[f"{metric}_efficiency"] = []
        
        results["rankings"] = rankings
        
        pareto_analysis = self._pareto_frontier_analysis(results) # Pass results to use potentially cleaned metrics
        results["pareto_analysis"] = pareto_analysis
        
        return results
    
    def _pareto_frontier_analysis(self, experiment_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        points = []
        for name, metrics in experiment_results.items(): # Use the processed results from compare_all
            best_score_val = metrics.get('best_score')
            total_tokens_val = metrics.get('total_tokens')

            # Convert to float for Pareto analysis, handling 'NaN', 'inf' strings
            try: best_score = float(best_score_val) if isinstance(best_score_val, str) else best_score_val
            except (ValueError, TypeError): best_score = float('-inf')
            if not isinstance(best_score, (int, float)) or np.isnan(best_score): best_score = float('-inf')

            try: total_tokens = float(total_tokens_val) if isinstance(total_tokens_val, str) else total_tokens_val
            except (ValueError, TypeError): total_tokens = float('inf') # High cost if unparseable
            if not isinstance(total_tokens, (int, float)) or np.isnan(total_tokens): total_tokens = float('inf')
            
            points.append({
                "name": name,
                "score": best_score, 
                "cost": -total_tokens, 
                "efficiency": best_score / max(total_tokens, 1) if total_tokens > 0 and not np.isinf(best_score) and best_score > float('-inf') else 0
            })
        
        pareto_frontier = []
        if not points: return {"pareto_frontier": [], "dominated_solutions": []}

        for i, point1 in enumerate(points):
            is_pareto = True
            for j, point2 in enumerate(points):
                if i != j:
                    if (point2["score"] >= point1["score"] and point2["cost"] >= point1["cost"]) and \
                       (point2["score"] > point1["score"] or point2["cost"] > point1["cost"]):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_frontier.append(point1)
        
        dominated_names = [p["name"] for p in points if not any(pf["name"] == p["name"] for pf in pareto_frontier)]
        dominated_solutions_details = [p for p in points if p["name"] in dominated_names]

        return {
            "pareto_frontier": pareto_frontier,
            "dominated_solutions": dominated_solutions_details
        }
    
    def generate_summary_report(self, output_file: str = "experiment_comparison.json"):
        comparison_results = self.compare_all()
        if "error" in comparison_results:
            print(f"Error generating summary report: {comparison_results['error']}", file=sys.stderr)
            if output_file:
                try:
                    with open(output_file, 'w') as f:
                        json.dump(comparison_results, f, indent=2)
                except Exception as e:
                    print(f"Could not write error to {output_file}: {e}", file=sys.stderr)
            return comparison_results 

        all_scores_numeric = []
        all_tokens_numeric = []
        all_times_numeric = []
        
        for name, result_data in comparison_results.items():
            if name == "rankings" or name == "pareto_analysis": continue # Skip meta-keys
            # Safely extract and convert scores, tokens, times for overall statistics
            score = result_data.get('best_score')
            tokens = result_data.get('total_tokens')
            # total_time is not directly in results[name], but can be reconstructed or taken from profiler if needed
            # For now, let's focus on what's in `results` for overall stats. If total_time is needed, it needs to be added to `results[name]`
            
            try: s_val = float(score) if isinstance(score, str) else score
            except: s_val = np.nan
            if isinstance(s_val, (int,float)) and not np.isnan(s_val): all_scores_numeric.append(s_val)

            try: t_val = float(tokens) if isinstance(tokens, str) else tokens
            except: t_val = np.nan
            if isinstance(t_val, (int,float)) and not np.isnan(t_val): all_tokens_numeric.append(t_val)

            # For total_time, we might need to access the profiler instance again if it's not in `result_data`
            # profiler = self.experiments.get(name)
            # if profiler and isinstance(profiler.metrics.total_evaluate_time + profiler.metrics.total_sample_time, (int,float)):
            #     all_times_numeric.append(profiler.metrics.total_evaluate_time + profiler.metrics.total_sample_time)

        # Calculate overall statistics from the collected numeric lists
        mean_scores_list = []
        for name_key in self.experiments.keys(): # Iterate actual experiment names
            mean_s = comparison_results.get(name_key, {}).get('mean_score')
            try: ms_val = float(mean_s) if isinstance(mean_s, str) else mean_s
            except: ms_val = np.nan
            if isinstance(ms_val, (int,float)) and not np.isnan(ms_val): mean_scores_list.append(ms_val)
        
        mean_score_across_exp = np.mean(mean_scores_list) if mean_scores_list else float('nan')
        score_std_across_exp = np.std(all_scores_numeric) if all_scores_numeric else float('nan')

        summary = {
            "experiment_count": len(self.experiments),
            "overall_statistics": {
                "score_range": [min(all_scores_numeric), max(all_scores_numeric)] if all_scores_numeric else [str(float('-inf')), str(float('-inf'))],
                "token_range": [min(all_tokens_numeric), max(all_tokens_numeric)] if all_tokens_numeric else [0, 0],
                # "time_range": [min(all_times_numeric), max(all_times_numeric)] if all_times_numeric else [0, 0], # Add if all_times_numeric is populated
                "mean_score_across_experiments": str(mean_score_across_exp),
                "score_std_across_experiments": str(score_std_across_exp)
            },
            "detailed_comparison": comparison_results
        }
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=lambda o: str(o) if isinstance(o, float) and (np.isnan(o) or np.isinf(o)) else o)
            except Exception as e:
                print(f"Error saving summary report to {output_file}: {e}", file=sys.stderr)
            
        return summary 