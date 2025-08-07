import sys
import os
import numpy as np
import time
import json
from typing import Dict, Callable, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to sys.path to find the modules
sys.path.append('../../')  

from llm4ad.task.science_discovery.ode_1d import ODEEvaluation
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.funsearch import FunSearch
from llm4ad.method.hillclimb import HillClimb

# Import our enhanced metrics system
from comprehensive_metrics import ComprehensiveProfiler, ExperimentComparator
from enhanced_llm_wrapper import EnhancedLLMWrapper, TokenAwareLLMFactory
from enhanced_evaluation import EnhancedODEEvaluation





# Check for required environment variables
def check_api_key():
    """Check if API key is available in environment variables."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Please create a .env file with your API key.")
    return api_key

def run_comprehensive_experiment(experiment_name: str, llm_config: dict, 
                                method_config: dict, method_type: str = 'funsearch', num_runs: int = 3):
    """
    Run comprehensive ODE discovery experiment with multi-dimensional metrics
    
    Args:
        experiment_name: Name of the experiment
        llm_config: LLM configuration
        method_config: Method configuration
        method_type: Method type ('funsearch', 'hillclimb')
        num_runs: Number of runs
    
    Returns:
        Experiment results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Running ODE Discovery Experiment: {experiment_name} (Method: {method_type.upper()})")
    print(f"{'='*60}")
    
    # Create experiment comparator
    comparator = ExperimentComparator()
    
    all_results = []
    
    for run_id in range(num_runs):
        print(f"\n--- Run {run_id + 1}/{num_runs} ---")
        
        # Create comprehensive profiler
        profiler = ComprehensiveProfiler(
            log_dir=f"comprehensive_logs/{experiment_name}/run_{run_id}",
            track_tokens=True
        )
        
        # Create enhanced LLM
        enhanced_llm = TokenAwareLLMFactory.create_enhanced_llm(
            host=llm_config['host'],
            key=llm_config['key'],
            model=llm_config['model'],
            profiler=profiler,
            **llm_config.get('kwargs', {})
        )
        
        # Create enhanced evaluation task
        enhanced_task = EnhancedODEEvaluation(
            profiler=profiler,
            stability_runs=3,  # Run each solution 3 times for stability testing
            collect_detailed_metrics=True
        )
        
        
        # Create method instance based on method_type
        if method_type.lower() == 'funsearch':
            method = FunSearch(
                llm=enhanced_llm,
                profiler=profiler,
                evaluation=enhanced_task,
                **method_config
            )
        elif method_type.lower() == 'hillclimb':
            method = HillClimb(
                llm=enhanced_llm,
                profiler=profiler,
                evaluation=enhanced_task,
                **method_config
            )
        else:
            raise ValueError(f"Unsupported method type: {method_type}. Supported types: 'funsearch', 'hillclimb'")
        
        # Run optimization
        print(f"Running {method_type.upper()} optimization...")
        start_time = time.time()
        method.run()
        total_time = time.time() - start_time
        
        # Log final debug information
        final_debug_state = profiler.debug_profiler_state()
        print(f"\nDEBUG - Final profiler state after {method_type.upper()} optimization:")
        print(f"  Best score: {final_debug_state['best_score']}")
        print(f"  Total evaluations: {final_debug_state['total_evaluator_calls']}")
        print(f"  Successful evaluations: {final_debug_state['successful_evaluations']}")
        print(f"  Failed evaluations: {final_debug_state['failed_evaluations']}")
        
        if final_debug_state['best_score'] == float('-inf'):
            print(f"ERROR: Best score is still -inf after {method_type.upper()} optimization!")
            print("This suggests no valid scores were found during optimization.")
        else:
            print(f"SUCCESS: Found valid best score: {final_debug_state['best_score']:.6f}")
        
        # Collect post-run metrics
        profiler.calculate_final_metrics()
        llm_usage = enhanced_llm.get_usage_summary()
        
        # Generate comprehensive report
        evaluation_report = enhanced_task.export_comprehensive_report(
            f"comprehensive_logs/{experiment_name}/run_{run_id}/evaluation_report.json"
        )
        
        # Save LLM usage logs
        enhanced_llm.export_detailed_log(
            f"comprehensive_logs/{experiment_name}/run_{run_id}/llm_usage.json"
        )
        
        # Save comprehensive metrics
        profiler.save_metrics(
            f"comprehensive_logs/{experiment_name}/run_{run_id}/comprehensive_metrics.json"
        )
        
        # Get final scores with safe handling
        best_score = profiler.get_best_score()
        if best_score == float('-inf'):
            print(f"WARNING: Run {run_id + 1} completed with no valid scores (best_score = -inf)")
        

        
        run_result = {
            'run_id': run_id,
            'experiment_name': experiment_name,
            'method_type': method_type,
            'total_time': total_time,
            'final_metrics': {
                'best_score': best_score,
                'mean_score': getattr(profiler.metrics, 'mean_score', 0),
                'std_score': getattr(profiler.metrics, 'std_score', 0),
                'total_evaluator_calls': profiler.get_total_evaluations(),
                'successful_evaluations': getattr(profiler.metrics, 'successful_evaluations', 0),
                'convergence_generation': getattr(profiler.metrics, 'convergence_generation', None)
            },
            'llm_usage': llm_usage,
            'stability_metrics': profiler.get_stability_metrics(),
            'efficiency_metrics': profiler.get_efficiency_metrics(),
            'evaluation_diversity': evaluation_report.get('diversity_analysis', {}),
            'trend_analysis': evaluation_report.get('trend_analysis', {})
        }
        
        all_results.append(run_result)
        
        # Add to comparator
        comparator.add_experiment(f"{experiment_name}_run_{run_id}", profiler)
        
        print(f"\nRun {run_id + 1} Summary:")
        print(f"  Best Score: {run_result['final_metrics']['best_score']:.6f}")
        print(f"  Evaluations: {run_result['final_metrics']['total_evaluator_calls']}")
        print(f"  Total Tokens: {llm_usage['basic_stats']['total_tokens']}")
        print(f"  Total Cost: ${llm_usage['basic_stats']['total_cost_usd']:.6f}")
        print(f"  Success Rate: {run_result['stability_metrics'].get('success_rate', 0):.3f}")
    
    # Generate inter-experiment comparison report
    comparison_report = comparator.generate_summary_report(
        f"comprehensive_logs/{experiment_name}/experiment_comparison.json"
    )
    
    # Calculate experiment-level statistics
    experiment_summary = calculate_experiment_statistics(all_results, experiment_name, method_type)
    
    # Save complete experiment results
    complete_results = {
        'experiment_name': experiment_name,
        'method_type': method_type,
        'num_runs': num_runs,
        'experiment_summary': experiment_summary,
        'individual_runs': all_results,
        'comparison_report': comparison_report,
        'configurations': {
            'llm_config': llm_config,
            'method_config': method_config
        }
    }
    
    with open(f"comprehensive_logs/{experiment_name}/complete_results.json", 'w') as f:
        json.dump(complete_results, f, indent=2, default=str)
    
    print(f"\nExperiment '{experiment_name}' completed!")
    print_experiment_summary(experiment_summary)
    
    return complete_results

def calculate_experiment_statistics(results: list, experiment_name: str, method_type: str) -> dict:
    """
    Calculate comprehensive statistics across all runs of an experiment
    
    Args:
        results: List of individual run results
        experiment_name: Name of the experiment
        method_type: Type of method used
        
    Returns:
        Dictionary containing experiment-level statistics
    """
    if not results:
        return {'error': 'No results to analyze'}
    
    # Extract key metrics from all runs (filter out infinity values)
    best_scores = [r['final_metrics']['best_score'] for r in results 
                  if r['final_metrics']['best_score'] is not None and r['final_metrics']['best_score'] != float('-inf')]
    mean_scores = [r['final_metrics']['mean_score'] for r in results 
                  if r['final_metrics']['mean_score'] is not None and r['final_metrics']['mean_score'] != float('-inf')]
    evaluator_calls = [r['final_metrics']['total_evaluator_calls'] for r in results]
    total_costs = [r['llm_usage']['basic_stats']['total_cost_usd'] for r in results]
    total_tokens = [r['llm_usage']['basic_stats']['total_tokens'] for r in results]
    success_rates = [r['stability_metrics'].get('success_rate', 0) for r in results]
    

    
    # Helper function to calculate stats with fallback
    def safe_stats(values, default=0):
        if not values:
            return {
                'mean': default,
                'std': 0,
                'min': default,
                'max': default,
                'median': default
            }
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(min(values)),
            'max': float(max(values)),
            'median': float(np.median(values))
        }
    
    def safe_stats_with_total(values, default=0):
        if not values:
            return {
                'mean': default,
                'std': 0,
                'min': default,
                'max': default,
                'median': default,
                'total': default
            }
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(min(values)),
            'max': float(max(values)),
            'median': float(np.median(values)),
            'total': float(sum(values))
        }
    
    return {
        'method_type': method_type,
        'performance_metrics': {
            'best_score': safe_stats(best_scores),
            'mean_score': safe_stats(mean_scores)
        },
        'cost_metrics': {
            'total_cost_usd': safe_stats_with_total(total_costs),
            'total_tokens': safe_stats_with_total(total_tokens)
        },
        'efficiency_metrics': {
            'evaluator_calls': safe_stats_with_total(evaluator_calls),
            'cost_efficiency': {
                'score_per_dollar': [score / cost for score, cost in zip(best_scores, total_costs) 
                                   if cost > 0 and score is not None],
                'score_per_evaluation': [score / calls for score, calls in zip(best_scores, evaluator_calls) 
                                       if calls > 0 and score is not None]
            }
        },
        'stability_metrics': {
            'success_rate': safe_stats(success_rates)
        }
    }

def print_experiment_summary(summary: dict):
    """
    Print a formatted summary of experiment results
    
    Args:
        summary: Experiment summary dictionary
    """
    print("\n" + "="*80)
    print(f"ODE Discovery Experiment Statistical Summary ({summary.get('method_type', 'Unknown').upper()})")
    print("="*80)
    
    perf = summary['performance_metrics']
    cost = summary['cost_metrics']
    eff = summary['efficiency_metrics']
    stab = summary['stability_metrics']
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Best Score: {perf['best_score']['mean']:.6f} ± {perf['best_score']['std']:.6f}")
    print(f"  Score Range: [{perf['best_score']['min']:.6f}, {perf['best_score']['max']:.6f}]")
    
    print(f"\n💰 Cost Metrics:")
    print(f"  Average Total Cost: ${cost['total_cost_usd']['mean']:.6f} ± ${cost['total_cost_usd']['std']:.6f}")
    print(f"  Average Token Usage: {cost['total_tokens']['mean']:.0f} ± {cost['total_tokens']['std']:.0f}")
    print(f"  Total Cost: ${cost['total_cost_usd']['total']:.6f}")
    
    print(f"\n⚡ Efficiency Metrics:")
    print(f"  Average Evaluations: {eff['evaluator_calls']['mean']:.0f} ± {eff['evaluator_calls']['std']:.0f}")
    if eff['cost_efficiency']['score_per_dollar']:
        print(f"  Cost Efficiency (score/$): {np.mean(eff['cost_efficiency']['score_per_dollar']):.0f}")
    if eff['cost_efficiency']['score_per_evaluation']:
        print(f"  Evaluation Efficiency (score/eval): {np.mean(eff['cost_efficiency']['score_per_evaluation']):.6f}")
    
    print(f"\n🎯 Stability Metrics:")
    print(f"  Success Rate: {stab['success_rate']['mean']:.3f} ± {stab['success_rate']['std']:.3f}")

def run_comparative_experiments():
    """
    Run comparative experiments across different ODE discovery methods
    """
    print("Starting Comprehensive ODE Discovery Evaluation Suite")
    print("=" * 60)
    
    # Define LLM configuration
    base_llm_config = {
        'host': 'api.deepseek.com',
        'key': os.getenv('DEEPSEEK_API_KEY'),  # Replace with your key
        'model': 'deepseek-chat',
        'kwargs': {'timeout': 60}
    }
    
    # Define standard experimental configurations for each method
    experiments = [
        # FunSearch standard configuration
        {
            'name': 'ode_funsearch_standard',
            'method_type': 'funsearch',
            'llm_config': base_llm_config,
            'method_config': {
                'max_sample_nums': 20,
                'num_samplers': 1,
                'num_evaluators': 1,
                'samples_per_prompt': 4,
                'debug_mode': False
            }
        },
        # HillClimb standard configuration
        {
            'name': 'ode_hillclimb_standard',
            'method_type': 'hillclimb',
            'llm_config': base_llm_config,
            'method_config': {
                'max_sample_nums': 20,
                'num_samplers': 1,
                'num_evaluators': 1,
                'debug_mode': False
            }
        },

    ]
    
    # Run all experiments
    all_experiment_results = {}
    
    for exp_config in experiments:
        results = run_comprehensive_experiment(
            experiment_name=exp_config['name'],
            llm_config=exp_config['llm_config'],
            method_config=exp_config['method_config'],
            method_type=exp_config['method_type'],
            num_runs=1
        )
        all_experiment_results[exp_config['name']] = results
    
    # Generate cross-experiment comparison report
    print(f"\n{'='*80}")
    print("Cross-Method ODE Discovery Comparison Analysis")
    print(f"{'='*80}")
    
    comparison_table = []
    for exp_name, results in all_experiment_results.items():
        summary = results['experiment_summary']
        # Handle case where best_score might be 0 (no valid scores found)
        best_score_mean = summary['performance_metrics']['best_score']['mean']
        if best_score_mean == 0 and summary['performance_metrics']['best_score']['std'] == 0:
            best_score_mean = float('-inf')  # No valid scores were found
            
        comparison_table.append({
            'experiment': exp_name,
            'method_type': results['method_type'],
            'avg_best_score': best_score_mean,
            'score_std': summary['performance_metrics']['best_score']['std'],
            'avg_cost': summary['cost_metrics']['total_cost_usd']['mean'],
            'avg_tokens': summary['cost_metrics']['total_tokens']['mean'],
            'avg_evaluations': summary['efficiency_metrics']['evaluator_calls']['mean'],
            'success_rate': summary['stability_metrics']['success_rate']['mean']
        })
    
    # Print comparison table
    print(f"\n{'Experiment Name':<25} {'Method':<10} {'Avg Best Score':<12} {'Score Std':<10} {'Avg Cost($)':<12} {'Avg Tokens':<10} {'Evaluations':<10} {'Success Rate':<12}")
    print("-" * 110)
    for row in comparison_table:
        print(f"{row['experiment']:<25} {row['method_type']:<10} {row['avg_best_score']:<12.6f} {row['score_std']:<10.6f} {row['avg_cost']:<12.6f} {row['avg_tokens']:<10.0f} {row['avg_evaluations']:<10.0f} {row['success_rate']:<12.3f}")
    
    # Group analysis by method type
    method_groups = {}
    for row in comparison_table:
        method_type = row['method_type']
        if method_type not in method_groups:
            method_groups[method_type] = []
        method_groups[method_type].append(row)
    
    print(f"\n📊 Method Group Analysis:")
    method_summaries = {}
    
    for method_type, method_rows in method_groups.items():
        avg_score = np.mean([row['avg_best_score'] for row in method_rows if row['avg_best_score'] != float('-inf')])
        avg_cost = np.mean([row['avg_cost'] for row in method_rows])
        avg_success = np.mean([row['success_rate'] for row in method_rows])
        
        method_summaries[method_type] = {
            'avg_score': avg_score,
            'avg_cost': avg_cost,
            'avg_success_rate': avg_success,
            'configurations': [row['experiment'] for row in method_rows]
        }
        
        print(f"  {method_type.upper()} Average Performance:")
        print(f"    Average Best Score: {avg_score:.6f}")
        print(f"    Average Cost: ${avg_cost:.6f}")
        print(f"    Average Success Rate: {avg_success:.3f}")
        print(f"    Cost Efficiency (score/dollar): {avg_score/avg_cost:.0f}")
        print()
    
    # Cross-method comparison
    if len(method_summaries) > 1:
        best_method_by_score = max(method_summaries.items(), key=lambda x: x[1]['avg_score'] if x[1]['avg_score'] != float('-inf') else -float('inf'))
        best_method_by_efficiency = max(method_summaries.items(), key=lambda x: x[1]['avg_score'] / max(x[1]['avg_cost'], 1e-8) if x[1]['avg_score'] != float('-inf') else -float('inf'))
        most_stable_method = max(method_summaries.items(), key=lambda x: x[1]['avg_success_rate'])
        
        print(f"🏆 Method Comparison Winners:")
        print(f"  Highest Average Score: {best_method_by_score[0].upper()} (score: {best_method_by_score[1]['avg_score']:.6f})")
        print(f"  Most Cost Efficient: {best_method_by_efficiency[0].upper()} (score/cost: {best_method_by_efficiency[1]['avg_score']/max(best_method_by_efficiency[1]['avg_cost'], 1e-8):.0f})")
        print(f"  Most Stable: {most_stable_method[0].upper()} (success rate: {most_stable_method[1]['avg_success_rate']:.3f})")
    
    # Save cross-experiment comparison results
    cross_experiment_analysis = {
        'timestamp': time.time(),
        'experiments': list(all_experiment_results.keys()),
        'comparison_table': comparison_table,
        'method_summaries': method_summaries,
        'detailed_results': all_experiment_results
    }
    
    os.makedirs('comprehensive_logs/cross_experiment_analysis', exist_ok=True)
    with open('comprehensive_logs/cross_experiment_analysis/ode_all_methods_comparison.json', 'w') as f:
        json.dump(cross_experiment_analysis, f, indent=2, default=str)
    
    # Find optimal configurations
    best_by_score = max(comparison_table, key=lambda x: x['avg_best_score'] if x['avg_best_score'] != float('-inf') else -float('inf'))
    best_by_efficiency = max(comparison_table, key=lambda x: x['avg_best_score'] / max(x['avg_cost'], 1e-8) if x['avg_best_score'] != float('-inf') else -float('inf'))
    most_stable = max(comparison_table, key=lambda x: x['success_rate'])
    
    print(f"\n🏆 Optimal Configuration Analysis:")
    print(f"  Highest Score: {best_by_score['experiment']} ({best_by_score['method_type']}) - score: {best_by_score['avg_best_score']:.6f}")
    print(f"  Most Efficient: {best_by_efficiency['experiment']} ({best_by_efficiency['method_type']}) - score/cost: {best_by_efficiency['avg_best_score']/max(best_by_efficiency['avg_cost'], 1e-8):.0f}")
    print(f"  Most Stable: {most_stable['experiment']} ({most_stable['method_type']}) - success rate: {most_stable['success_rate']:.3f}")
    
    return cross_experiment_analysis

def main():
    """
    Main entry point for comprehensive ODE discovery evaluation across all methods
    """
    try:
        # Ensure log directory exists
        os.makedirs('comprehensive_logs', exist_ok=True)
        
        # Run comprehensive comparative experiments
        results = run_comparative_experiments()
        
        print(f"\n✅ All ODE discovery experiments completed!")
        print(f"📁 Detailed results saved in: comprehensive_logs/")
        print(f"📊 Cross-method comparison report: comprehensive_logs/cross_experiment_analysis/ode_all_methods_comparison.json")
        
        return results
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return None
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main() 