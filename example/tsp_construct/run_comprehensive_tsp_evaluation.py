import sys
import os
import numpy as np
import time
import json
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to sys.path to find the modules
sys.path.append('../../')  

from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.eohmeme import EoH

# Import our enhanced metrics system
from comprehensive_metrics import ComprehensiveProfiler, ExperimentComparator
from enhanced_llm_wrapper import EnhancedLLMWrapper, TokenAwareLLMFactory
from enhanced_evaluation import EnhancedTSPEvaluation

def create_simple_tsp_solver():
    """创建简单TSP求解器用于baseline对比"""
    def simple_greedy_solver(current_node: int, destination_node: int, 
                           unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
        """
        A simple greedy TSP solver for baseline comparison.
        Always chooses the nearest unvisited node.
        
        Args:
            current_node: ID of the current node
            destination_node: ID of the destination node (not used in this simple version)
            unvisited_nodes: Array of IDs of unvisited nodes
            distance_matrix: Distance matrix of nodes
            
        Returns:
            ID of the next node to visit
        """
        if len(unvisited_nodes) == 0:
            return destination_node
            
        # Find the nearest unvisited node
        distances = distance_matrix[current_node][unvisited_nodes]
        nearest_index = np.argmin(distances)
        return unvisited_nodes[nearest_index]
    
    return simple_greedy_solver

def create_advanced_tsp_solver():
    """创建更高级的TSP求解器用于对比"""
    def nearest_neighbor_with_lookahead(current_node: int, destination_node: int,
                                      unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
        """
        A more sophisticated TSP solver with 2-step lookahead.
        
        Args:
            current_node: ID of the current node
            destination_node: ID of the destination node
            unvisited_nodes: Array of IDs of unvisited nodes
            distance_matrix: Distance matrix of nodes
            
        Returns:
            ID of the next node to visit
        """
        if len(unvisited_nodes) == 0:
            return destination_node
            
        if len(unvisited_nodes) == 1:
            return unvisited_nodes[0]
            
        # Consider 2-step lookahead for better decisions
        best_next_node = None
        best_total_cost = float('inf')
        
        for next_node in unvisited_nodes:
            # Cost to go to next_node
            cost_to_next = distance_matrix[current_node][next_node]
            
            # Estimate cost from next_node to its best remaining option
            remaining_nodes = unvisited_nodes[unvisited_nodes != next_node]
            if len(remaining_nodes) > 0:
                min_cost_from_next = np.min(distance_matrix[next_node][remaining_nodes])
                total_estimated_cost = cost_to_next + min_cost_from_next
            else:
                # If this is the last node, add cost to return to destination
                total_estimated_cost = cost_to_next + distance_matrix[next_node][destination_node]
            
            if total_estimated_cost < best_total_cost:
                best_total_cost = total_estimated_cost
                best_next_node = next_node
                
        return best_next_node if best_next_node is not None else unvisited_nodes[0]
    
    return nearest_neighbor_with_lookahead


# Check for required environment variables
def check_api_key():
    """Check if API key is available in environment variables."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Please create a .env file with your API key.")
    return api_key

def run_comprehensive_experiment(experiment_name: str, llm_config: dict, 
                                method_config: dict, num_runs: int = 3):
    """
    运行综合实验，收集多维度指标
    
    Args:
        experiment_name: 实验名称
        llm_config: LLM配置
        method_config: 方法配置
        num_runs: 运行次数
    
    Returns:
        实验结果和指标
    """
    print(f"\n{'='*60}")
    print(f"运行TSP实验: {experiment_name}")
    print(f"{'='*60}")
    
    # 创建实验对比器
    comparator = ExperimentComparator()
    
    all_results = []
    
    for run_id in range(num_runs):
        print(f"\n--- 运行 {run_id + 1}/{num_runs} ---")
        
        # 创建综合profiler
        profiler = ComprehensiveProfiler(
            log_dir=f"comprehensive_logs/{experiment_name}/run_{run_id}",
            track_tokens=True
        )
        
        # 创建增强版LLM
        enhanced_llm = TokenAwareLLMFactory.create_enhanced_llm(
            host=llm_config['host'],
            key=llm_config['key'],
            model=llm_config['model'],
            profiler=profiler,
            **llm_config.get('kwargs', {})
        )
        
        # 创建增强版评估器
        enhanced_task = EnhancedTSPEvaluation(
            profiler=profiler,
            stability_runs=3,  # 每个解运行3次以测试稳定性
            collect_detailed_metrics=True
        )
        
        # 先测试baseline求解器
        print("测试简单贪心baseline...")
        simple_solver = create_simple_tsp_solver()
        baseline_result = enhanced_task.evaluate_with_stability_analysis(
            simple_solver, 
            function_id="simple_greedy_baseline"
        )
        print(f"简单贪心最佳分数: {baseline_result['summary']['best_score']:.3f}")
        print(f"简单贪心稳定性: {baseline_result['stability_metrics']['stability_score']:.3f}")
        
        # 测试高级baseline求解器
        print("测试高级baseline...")
        advanced_solver = create_advanced_tsp_solver()
        advanced_baseline_result = enhanced_task.evaluate_with_stability_analysis(
            advanced_solver,
            function_id="advanced_baseline"
        )
        print(f"高级baseline最佳分数: {advanced_baseline_result['summary']['best_score']:.3f}")
        print(f"高级baseline稳定性: {advanced_baseline_result['stability_metrics']['stability_score']:.3f}")
        
        # 配置EoH方法
        method = EoH(
            llm=enhanced_llm,
            profiler=profiler,
            evaluation=enhanced_task,
            **method_config
        )
        
        # 运行优化
        print("运行EoH优化...")
        start_time = time.time()
        method.run()
        total_time = time.time() - start_time
        
        # 收集运行后的指标
        profiler.calculate_final_metrics()
        llm_usage = enhanced_llm.get_usage_summary()
        
        # 生成综合报告
        evaluation_report = enhanced_task.export_comprehensive_report(
            f"comprehensive_logs/{experiment_name}/run_{run_id}/evaluation_report.json"
        )
        
        # 保存LLM使用日志
        enhanced_llm.export_detailed_log(
            f"comprehensive_logs/{experiment_name}/run_{run_id}/llm_usage.json"
        )
        
        # 保存综合指标
        profiler.save_metrics(
            f"comprehensive_logs/{experiment_name}/run_{run_id}/comprehensive_metrics.json"
        )
        
        # 收集本次运行的结果
        run_result = {
            'run_id': run_id,
            'experiment_name': experiment_name,
            'total_time': total_time,
            'baseline_performance': {
                'simple_greedy': baseline_result,
                'advanced': advanced_baseline_result
            },
            'final_metrics': {
                'best_score': profiler.metrics.best_score,
                'mean_score': profiler.metrics.mean_score,
                'std_score': profiler.metrics.std_score,
                'total_evaluator_calls': profiler.metrics.total_evaluator_calls,
                'successful_evaluations': profiler.metrics.successful_evaluations,
                'convergence_generation': profiler.metrics.convergence_generation
            },
            'llm_usage': llm_usage,
            'stability_metrics': profiler.get_stability_metrics(),
            'efficiency_metrics': profiler.get_efficiency_metrics(),
            'evaluation_diversity': enhanced_task.analyze_solution_diversity(),
            'trend_analysis': enhanced_task.generate_performance_trend_analysis()
        }
        
        all_results.append(run_result)
        
        # 添加到实验对比器
        comparator.add_experiment(f"{experiment_name}_run_{run_id}", profiler)
        
        print(f"完成运行 {run_id + 1}:")
        print(f"  最佳分数: {profiler.metrics.best_score:.3f}")
        print(f"  评估次数: {profiler.metrics.total_evaluator_calls}")
        print(f"  总Token数: {llm_usage['total_tokens']}")
        print(f"  总成本: ${llm_usage['cost_analysis']['total_cost_usd']:.6f}")
        print(f"  成功率: {profiler.get_stability_metrics().get('success_rate', 0):.3f}")
        
        # 与baseline对比
        simple_best = baseline_result['summary']['best_score']
        advanced_best = advanced_baseline_result['summary']['best_score']
        eoh_best = profiler.metrics.best_score
        
        if simple_best != float('-inf'):
            improvement_simple = ((eoh_best - simple_best) / abs(simple_best)) * 100
            print(f"  相对简单贪心改进: {improvement_simple:.1f}%")
        
        if advanced_best != float('-inf'):
            improvement_advanced = ((eoh_best - advanced_best) / abs(advanced_best)) * 100
            print(f"  相对高级baseline改进: {improvement_advanced:.1f}%")
    
    # 生成实验间对比报告
    comparison_report = comparator.generate_summary_report(
        f"comprehensive_logs/{experiment_name}/experiment_comparison.json"
    )
    
    # 计算统计摘要
    experiment_summary = calculate_experiment_statistics(all_results, experiment_name)
    
    # 保存完整的实验结果
    complete_results = {
        'experiment_name': experiment_name,
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
        json.dump(complete_results, f, indent=2)
    
    print(f"\n实验 '{experiment_name}' 完成!")
    print_experiment_summary(experiment_summary)
    
    return complete_results

def calculate_experiment_statistics(results: list, experiment_name: str) -> dict:
    """计算实验统计信息"""
    # 提取关键指标
    best_scores = [r['final_metrics']['best_score'] for r in results if r['final_metrics']['best_score'] is not None]
    mean_scores = [r['final_metrics']['mean_score'] for r in results if r['final_metrics']['mean_score'] is not None]
    evaluator_calls = [r['final_metrics']['total_evaluator_calls'] for r in results]
    total_costs = [r['llm_usage']['cost_analysis']['total_cost_usd'] for r in results]
    total_tokens = [r['llm_usage']['total_tokens'] for r in results]
    success_rates = [r['stability_metrics'].get('success_rate', 0) for r in results]
    
    # 提取baseline性能
    simple_baseline_scores = []
    advanced_baseline_scores = []
    
    for r in results:
        simple_score = r['baseline_performance']['simple_greedy']['summary']['best_score']
        advanced_score = r['baseline_performance']['advanced']['summary']['best_score']
        
        # Filter out invalid scores
        if simple_score != float('-inf') and simple_score is not None:
            simple_baseline_scores.append(simple_score)
        if advanced_score != float('-inf') and advanced_score is not None:
            advanced_baseline_scores.append(advanced_score)
    
    # 计算改进比例
    improvements_simple = []
    improvements_advanced = []
    
    for r in results:
        best = r['final_metrics']['best_score']
        simple_baseline = r['baseline_performance']['simple_greedy']['summary']['best_score']
        advanced_baseline = r['baseline_performance']['advanced']['summary']['best_score']
        
        if (best is not None and simple_baseline != float('-inf') and 
            simple_baseline is not None and abs(simple_baseline) > 1e-8):
            improvement = ((best - simple_baseline) / abs(simple_baseline)) * 100
            improvements_simple.append(improvement)
            
        if (best is not None and advanced_baseline != float('-inf') and 
            advanced_baseline is not None and abs(advanced_baseline) > 1e-8):
            improvement = ((best - advanced_baseline) / abs(advanced_baseline)) * 100
            improvements_advanced.append(improvement)
    
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
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'median': np.median(values)
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
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'median': np.median(values),
            'total': sum(values)
        }
    
    return {
        'performance_metrics': {
            'best_score': safe_stats(best_scores),
            'mean_score': safe_stats(mean_scores),
            'improvement_over_simple_baseline': safe_stats(improvements_simple),
            'improvement_over_advanced_baseline': safe_stats(improvements_advanced)
        },
        'cost_metrics': {
            'total_cost_usd': safe_stats_with_total(total_costs),
            'total_tokens': {
                'mean': np.mean(total_tokens) if total_tokens else 0,
                'std': np.std(total_tokens) if total_tokens else 0,
                'min': min(total_tokens) if total_tokens else 0,
                'max': max(total_tokens) if total_tokens else 0,
                'total': sum(total_tokens) if total_tokens else 0
            }
        },
        'efficiency_metrics': {
            'evaluator_calls': {
                'mean': np.mean(evaluator_calls) if evaluator_calls else 0,
                'std': np.std(evaluator_calls) if evaluator_calls else 0,
                'total': sum(evaluator_calls) if evaluator_calls else 0
            },
            'cost_efficiency': {
                'score_per_dollar': [score / cost for score, cost in zip(best_scores, total_costs) 
                                   if cost > 0 and score is not None],
                'score_per_evaluation': [score / calls for score, calls in zip(best_scores, evaluator_calls) 
                                       if calls > 0 and score is not None]
            }
        },
        'stability_metrics': {
            'success_rate': safe_stats(success_rates),
            'baseline_comparison': {
                'simple_baseline_score_mean': np.mean(simple_baseline_scores) if simple_baseline_scores else 0,
                'advanced_baseline_score_mean': np.mean(advanced_baseline_scores) if advanced_baseline_scores else 0,
                'improvement_consistency_simple': np.std(improvements_simple) if improvements_simple else 0,
                'improvement_consistency_advanced': np.std(improvements_advanced) if improvements_advanced else 0
            }
        }
    }

def print_experiment_summary(summary: dict):
    """打印实验摘要"""
    print("\n" + "="*80)
    print("TSP实验统计摘要")
    print("="*80)
    
    perf = summary['performance_metrics']
    cost = summary['cost_metrics']
    eff = summary['efficiency_metrics']
    stab = summary['stability_metrics']
    
    print(f"\n📊 性能指标:")
    print(f"  最佳分数: {perf['best_score']['mean']:.3f} ± {perf['best_score']['std']:.3f}")
    print(f"  分数范围: [{perf['best_score']['min']:.3f}, {perf['best_score']['max']:.3f}]")
    print(f"  相对简单贪心改进: {perf['improvement_over_simple_baseline']['mean']:.1f}% ± {perf['improvement_over_simple_baseline']['std']:.1f}%")
    print(f"  相对高级baseline改进: {perf['improvement_over_advanced_baseline']['mean']:.1f}% ± {perf['improvement_over_advanced_baseline']['std']:.1f}%")
    
    print(f"\n💰 成本指标:")
    print(f"  平均总成本: ${cost['total_cost_usd']['mean']:.6f} ± ${cost['total_cost_usd']['std']:.6f}")
    print(f"  平均Token使用: {cost['total_tokens']['mean']:.0f} ± {cost['total_tokens']['std']:.0f}")
    print(f"  总成本: ${cost['total_cost_usd']['total']:.6f}")
    
    print(f"\n⚡ 效率指标:")
    print(f"  平均评估次数: {eff['evaluator_calls']['mean']:.0f} ± {eff['evaluator_calls']['std']:.0f}")
    if eff['cost_efficiency']['score_per_dollar']:
        print(f"  成本效率 (分数/$): {np.mean(eff['cost_efficiency']['score_per_dollar']):.0f}")
    if eff['cost_efficiency']['score_per_evaluation']:
        print(f"  评估效率 (分数/评估): {np.mean(eff['cost_efficiency']['score_per_evaluation']):.3f}")
    
    print(f"\n🎯 稳定性指标:")
    print(f"  成功率: {stab['success_rate']['mean']:.3f} ± {stab['success_rate']['std']:.3f}")
    print(f"  简单Baseline分数: {stab['baseline_comparison']['simple_baseline_score_mean']:.3f}")
    print(f"  高级Baseline分数: {stab['baseline_comparison']['advanced_baseline_score_mean']:.3f}")

def run_comparative_experiments():
    """运行对比实验"""
    
    # 实验配置
    experiments = [
        {
            'name': 'tsp_deepseek_standard',
            'llm_config': {
                'host': 'api.deepseek.com',
                'key': os.getenv('DEEPSEEK_API_KEY'),
                'model': 'deepseek-chat',
                'kwargs': {'timeout': 300}
            },
            'method_config': {
                'max_sample_nums': 50,
                'max_generations': 5,
                'pop_size': 8,
                'num_samplers': 1,
                'num_evaluators': 1,
                'use_memetic': True,
                'memetic_frequency': 1,
                'memetic_intensity': 0.3,
                'use_hybrid_local_search': True,
                'hybrid_local_search_method': 'cma-es',
                'use_evolution_memory': True,
                'memory_capacity': 10,
                'use_reflection': True,
                'reflection_frequency': 1,
                'debug_mode': False
            }
        },
        {
            'name': 'tsp_deepseek_aggressive',
            'llm_config': {
                'host': 'api.deepseek.com',
                'key': os.getenv('DEEPSEEK_API_KEY'),
                'model': 'deepseek-chat',
                'kwargs': {'timeout': 300}
            },
            'method_config': {
                'max_sample_nums': 80,
                'max_generations': 8,
                'pop_size': 12,
                'num_samplers': 1,
                'num_evaluators': 1,
                'use_memetic': True,
                'memetic_frequency': 1,
                'memetic_intensity': 0.5,
                'use_evolution_memory': True,
                'memory_capacity': 15,
                'use_reflection': True,
                'reflection_frequency': 1,
                'debug_mode': False
            }
        },
        {
            'name': 'tsp_deepseek_conservative',
            'llm_config': {
                'host': 'api.deepseek.com',
                'key': os.getenv('DEEPSEEK_API_KEY'),
                'model': 'deepseek-chat',
                'kwargs': {'timeout': 300}
            },
            'method_config': {
                'max_sample_nums': 30,
                'max_generations': 3,
                'pop_size': 6,
                'num_samplers': 1,
                'num_evaluators': 1,
                'use_memetic': False,
                'use_evolution_memory': True,
                'memory_capacity': 5,
                'use_reflection': False,
                'debug_mode': False
            }
        }
    ]
    
    # 运行所有实验
    all_experiment_results = {}
    
    for exp_config in experiments[:]:
        results = run_comprehensive_experiment(
            experiment_name=exp_config['name'],
            llm_config=exp_config['llm_config'],
            method_config=exp_config['method_config'],
            num_runs=1
        )
        all_experiment_results[exp_config['name']] = results
    
    # 生成跨实验对比报告
    print(f"\n{'='*80}")
    print("跨TSP实验对比分析")
    print(f"{'='*80}")
    
    comparison_table = []
    for exp_name, results in all_experiment_results.items():
        summary = results['experiment_summary']
        comparison_table.append({
            'experiment': exp_name,
            'avg_best_score': summary['performance_metrics']['best_score']['mean'],
            'score_std': summary['performance_metrics']['best_score']['std'],
            'avg_cost': summary['cost_metrics']['total_cost_usd']['mean'],
            'avg_tokens': summary['cost_metrics']['total_tokens']['mean'],
            'avg_evaluations': summary['efficiency_metrics']['evaluator_calls']['mean'],
            'success_rate': summary['stability_metrics']['success_rate']['mean'],
            'improvement_simple': summary['performance_metrics']['improvement_over_simple_baseline']['mean'],
            'improvement_advanced': summary['performance_metrics']['improvement_over_advanced_baseline']['mean']
        })
    
    # 打印对比表格
    print(f"\n{'实验名称':<25} {'平均最佳分数':<12} {'分数标准差':<10} {'平均成本($)':<12} {'平均Token':<10} {'评估次数':<8} {'成功率':<8} {'简单改进(%)':<12} {'高级改进(%)':<12}")
    print("-" * 130)
    for row in comparison_table:
        print(f"{row['experiment']:<25} {row['avg_best_score']:<12.3f} {row['score_std']:<10.3f} {row['avg_cost']:<12.6f} {row['avg_tokens']:<10.0f} {row['avg_evaluations']:<8.0f} {row['success_rate']:<8.3f} {row['improvement_simple']:<12.1f} {row['improvement_advanced']:<12.1f}")
    
    # 保存跨实验对比结果
    cross_experiment_analysis = {
        'timestamp': time.time(),
        'experiments': list(all_experiment_results.keys()),
        'comparison_table': comparison_table,
        'detailed_results': all_experiment_results
    }
    
    os.makedirs('comprehensive_logs/cross_experiment_analysis', exist_ok=True)
    with open('comprehensive_logs/cross_experiment_analysis/complete_comparison.json', 'w') as f:
        json.dump(cross_experiment_analysis, f, indent=2)
    
    # 找出最优配置
    best_by_score = max(comparison_table, key=lambda x: x['avg_best_score'])
    best_by_efficiency = min(comparison_table, key=lambda x: x['avg_cost'] / max(abs(x['avg_best_score']), 1e-8))
    most_stable = max(comparison_table, key=lambda x: x['success_rate'])
    best_improvement = max(comparison_table, key=lambda x: x['improvement_advanced'])
    
    print(f"\n🏆 最优配置分析:")
    print(f"  最高分数: {best_by_score['experiment']} (分数: {best_by_score['avg_best_score']:.3f})")
    print(f"  最高效率: {best_by_efficiency['experiment']} (成本效率: {best_by_efficiency['avg_cost']/max(abs(best_by_efficiency['avg_best_score']), 1e-8):.6f})")
    print(f"  最稳定: {most_stable['experiment']} (成功率: {most_stable['success_rate']:.3f})")
    print(f"  最大改进: {best_improvement['experiment']} (相对高级baseline: {best_improvement['improvement_advanced']:.1f}%)")
    
    return cross_experiment_analysis

def main():
    """主函数"""
    print("开始综合多维度TSP评价指标实验...")
    
    # 创建日志目录
    os.makedirs('comprehensive_logs', exist_ok=True)
    
    # 运行对比实验
    cross_experiment_results = run_comparative_experiments()
    
    print(f"\n✅ 所有TSP实验完成!")
    print(f"📁 详细结果保存在: comprehensive_logs/")
    print(f"📊 跨实验对比报告: comprehensive_logs/cross_experiment_analysis/complete_comparison.json")

if __name__ == '__main__':
    main() 