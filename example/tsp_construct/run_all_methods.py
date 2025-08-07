import sys
import os
import numpy as np
import time
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to sys.path to find the modules
sys.path.append('../../')  

from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.funsearch import FunSearch
from llm4ad.method.hillclimb import HillClimb

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
                                method_config: dict, method_type: str = 'funsearch', num_runs: int = 3):
    """
    运行综合实验，收集多维度指标
    
    Args:
        experiment_name: 实验名称
        llm_config: LLM配置
        method_config: 方法配置
        method_type: 方法类型 ('funsearch', 'hillclimb')
        num_runs: 运行次数
    
    Returns:
        实验结果和指标
    """
    print(f"\n{'='*60}")
    print(f"运行TSP实验: {experiment_name} (方法: {method_type.upper()})")
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
        
        # 根据方法类型创建不同的方法实例
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
        
        # 运行优化
        print(f"运行{method_type.upper()}优化...")
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
            'method_type': method_type,
            'total_time': total_time,
            'baseline_performance': {
                'simple_greedy': baseline_result,
                'advanced_baseline': advanced_baseline_result
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
    
    # 生成实验间对比报告
    comparison_report = comparator.generate_summary_report(
        f"comprehensive_logs/{experiment_name}/experiment_comparison.json"
    )
    
    # 计算统计摘要
    experiment_summary = calculate_experiment_statistics(all_results, experiment_name)
    
    # 保存完整的实验结果
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
        json.dump(complete_results, f, indent=2)
    
    print(f"\n实验 '{experiment_name}' 完成!")
    print_experiment_summary(experiment_summary)
    
    return complete_results

def calculate_experiment_statistics(results: list, experiment_name: str) -> dict:
    """计算实验统计信息"""
    # 提取关键指标
    best_scores = [r['final_metrics']['best_score'] for r in results]
    mean_scores = [r['final_metrics']['mean_score'] for r in results]
    evaluator_calls = [r['final_metrics']['total_evaluator_calls'] for r in results]
    total_costs = [r['llm_usage']['cost_analysis']['total_cost_usd'] for r in results]
    total_tokens = [r['llm_usage']['total_tokens'] for r in results]
    success_rates = [r['stability_metrics'].get('success_rate', 0) for r in results]
    
    # 提取baseline性能 - 使用简单贪心作为主要基准
    baseline_scores = [r['baseline_performance']['simple_greedy']['summary']['best_score'] for r in results]
    baseline_stability = [r['baseline_performance']['simple_greedy']['stability_metrics']['stability_score'] for r in results]
    
    # 提取高级baseline性能
    advanced_baseline_scores = [r['baseline_performance']['advanced_baseline']['summary']['best_score'] for r in results]
    
    # 计算改进比例
    improvements_simple = [(best - baseline) / abs(baseline) * 100 
                          for best, baseline in zip(best_scores, baseline_scores) 
                          if baseline != 0]
    
    improvements_advanced = [(best - baseline) / abs(baseline) * 100 
                           for best, baseline in zip(best_scores, advanced_baseline_scores) 
                           if baseline != 0]
    
    return {
        'performance_metrics': {
            'best_score': {
                'mean': np.mean(best_scores),
                'std': np.std(best_scores),
                'min': min(best_scores),
                'max': max(best_scores),
                'median': np.median(best_scores)
            },
            'mean_score': {
                'mean': np.mean(mean_scores),
                'std': np.std(mean_scores),
                'median': np.median(mean_scores)
            },
            'improvement_over_simple_baseline': {
                'mean': np.mean(improvements_simple) if improvements_simple else 0,
                'std': np.std(improvements_simple) if improvements_simple else 0,
                'min': min(improvements_simple) if improvements_simple else 0,
                'max': max(improvements_simple) if improvements_simple else 0
            },
            'improvement_over_advanced_baseline': {
                'mean': np.mean(improvements_advanced) if improvements_advanced else 0,
                'std': np.std(improvements_advanced) if improvements_advanced else 0,
                'min': min(improvements_advanced) if improvements_advanced else 0,
                'max': max(improvements_advanced) if improvements_advanced else 0
            }
        },
        'cost_metrics': {
            'total_cost_usd': {
                'mean': np.mean(total_costs),
                'std': np.std(total_costs),
                'min': min(total_costs),
                'max': max(total_costs),
                'total': sum(total_costs)
            },
            'total_tokens': {
                'mean': np.mean(total_tokens),
                'std': np.std(total_tokens),
                'min': min(total_tokens),
                'max': max(total_tokens),
                'total': sum(total_tokens)
            }
        },
        'efficiency_metrics': {
            'evaluator_calls': {
                'mean': np.mean(evaluator_calls),
                'std': np.std(evaluator_calls),
                'total': sum(evaluator_calls)
            },
            'cost_efficiency': {
                'score_per_dollar': [score / cost for score, cost in zip(best_scores, total_costs) if cost > 0],
                'score_per_evaluation': [score / calls for score, calls in zip(best_scores, evaluator_calls) if calls > 0]
            }
        },
        'stability_metrics': {
            'success_rate': {
                'mean': np.mean(success_rates),
                'std': np.std(success_rates),
                'min': min(success_rates),
                'max': max(success_rates)
            },
            'baseline_comparison': {
                'simple_baseline_score_mean': np.mean(baseline_scores),
                'simple_baseline_stability_mean': np.mean(baseline_stability),
                'advanced_baseline_score_mean': np.mean(advanced_baseline_scores),
                'improvement_consistency_simple': np.std(improvements_simple) if improvements_simple else float('inf'),
                'improvement_consistency_advanced': np.std(improvements_advanced) if improvements_advanced else float('inf')
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
    print(f"  相对简单baseline改进: {perf['improvement_over_simple_baseline']['mean']:.1f}% ± {perf['improvement_over_simple_baseline']['std']:.1f}%")
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
    print(f"  改进一致性(简单): {stab['baseline_comparison']['improvement_consistency_simple']:.3f}")
    print(f"  改进一致性(高级): {stab['baseline_comparison']['improvement_consistency_advanced']:.3f}")

def run_comparative_experiments():
    """运行对比实验"""
    
    # 基础LLM配置
    base_llm_config = {
        'host': 'api.deepseek.com',
        'key': os.getenv('DEEPSEEK_API_KEY'),
        'model': 'deepseek-chat',
        'kwargs': {'timeout': 300}
    }
    
    # 实验配置 - 包括FunSearch和HillClimb对比
    experiments = [
        # FunSearch实验配置
        {
            'name': 'tsp_funsearch_standard',
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
        # HillClimb实验配置
        {
            'name': 'tsp_hillclimb_standard',
            'method_type': 'hillclimb',
            'llm_config': base_llm_config,
            'method_config': {
                'max_sample_nums': 20,
                'num_samplers': 1,
                'num_evaluators': 1,
                'debug_mode': False
            }
        }
    ]
    
    # 运行所有实验
    all_experiment_results = {}
    
    for exp_config in experiments:
        results = run_comprehensive_experiment(
            experiment_name=exp_config['name'],
            llm_config=exp_config['llm_config'],
            method_config=exp_config['method_config'],
            method_type=exp_config['method_type'],
            num_runs=3  # 运行3次获得更好的统计
        )
        all_experiment_results[exp_config['name']] = results
    
    # 生成跨实验对比报告
    print(f"\n{'='*80}")
    print("跨实验对比分析 (FunSearch vs HillClimb)")
    print(f"{'='*80}")
    
    comparison_table = []
    for exp_name, results in all_experiment_results.items():
        summary = results['experiment_summary']
        comparison_table.append({
            'experiment': exp_name,
            'method_type': results['method_type'],
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
    print(f"\n{'实验名称':<25} {'方法':<10} {'平均最佳分数':<12} {'分数标准差':<10} {'平均成本($)':<12} {'平均Token':<10} {'评估次数':<8} {'成功率':<8} {'改进简单(%)':<12} {'改进高级(%)':<12}")
    print("-" * 140)
    for row in comparison_table:
        print(f"{row['experiment']:<25} {row['method_type']:<10} {row['avg_best_score']:<12.3f} {row['score_std']:<10.3f} {row['avg_cost']:<12.6f} {row['avg_tokens']:<10.0f} {row['avg_evaluations']:<8.0f} {row['success_rate']:<8.3f} {row['improvement_simple']:<12.1f} {row['improvement_advanced']:<12.1f}")
    
    # 按方法类型分组分析
    funsearch_results = [row for row in comparison_table if row['method_type'] == 'funsearch']
    hillclimb_results = [row for row in comparison_table if row['method_type'] == 'hillclimb']
    
    print(f"\n📊 方法对比分析:")
    if funsearch_results:
        fs_avg_score = np.mean([row['avg_best_score'] for row in funsearch_results])
        fs_avg_cost = np.mean([row['avg_cost'] for row in funsearch_results])
        fs_avg_success = np.mean([row['success_rate'] for row in funsearch_results])
        fs_avg_improvement_simple = np.mean([row['improvement_simple'] for row in funsearch_results])
        fs_avg_improvement_advanced = np.mean([row['improvement_advanced'] for row in funsearch_results])
        
        print(f"  FUNSEARCH平均表现:")
        print(f"    平均最佳分数: {fs_avg_score:.3f}")
        print(f"    平均成本: ${fs_avg_cost:.6f}")
        print(f"    平均成功率: {fs_avg_success:.3f}")
        print(f"    平均改进(简单baseline): {fs_avg_improvement_simple:.1f}%")
        print(f"    平均改进(高级baseline): {fs_avg_improvement_advanced:.1f}%")
    
    if hillclimb_results:
        hc_avg_score = np.mean([row['avg_best_score'] for row in hillclimb_results])
        hc_avg_cost = np.mean([row['avg_cost'] for row in hillclimb_results])
        hc_avg_success = np.mean([row['success_rate'] for row in hillclimb_results])
        hc_avg_improvement_simple = np.mean([row['improvement_simple'] for row in hillclimb_results])
        hc_avg_improvement_advanced = np.mean([row['improvement_advanced'] for row in hillclimb_results])
        
        print(f"  HILLCLIMB平均表现:")
        print(f"    平均最佳分数: {hc_avg_score:.3f}")
        print(f"    平均成本: ${hc_avg_cost:.6f}")
        print(f"    平均成功率: {hc_avg_success:.3f}")
        print(f"    平均改进(简单baseline): {hc_avg_improvement_simple:.1f}%")
        print(f"    平均改进(高级baseline): {hc_avg_improvement_advanced:.1f}%")
    
    if funsearch_results and hillclimb_results:
        print(f"  方法对比:")
        print(f"    分数差异: FunSearch vs HillClimb = {fs_avg_score:.3f} vs {hc_avg_score:.3f} ({((fs_avg_score - hc_avg_score) / hc_avg_score * 100):+.1f}%)")
        print(f"    成本效率: FunSearch={fs_avg_score/fs_avg_cost:.0f} vs HillClimb={hc_avg_score/hc_avg_cost:.0f} (分数/美元)")
        print(f"    稳定性: FunSearch={fs_avg_success:.3f} vs HillClimb={hc_avg_success:.3f}")
    
    # 保存跨实验对比结果
    cross_experiment_analysis = {
        'timestamp': time.time(),
        'experiments': list(all_experiment_results.keys()),
        'comparison_table': comparison_table,
        'method_comparison': {
            'funsearch_summary': {
                'avg_score': fs_avg_score if funsearch_results else None,
                'avg_cost': fs_avg_cost if funsearch_results else None,
                'avg_success_rate': fs_avg_success if funsearch_results else None,
                'avg_improvement_simple': fs_avg_improvement_simple if funsearch_results else None,
                'avg_improvement_advanced': fs_avg_improvement_advanced if funsearch_results else None,
                'configurations': [row['experiment'] for row in funsearch_results]
            },
            'hillclimb_summary': {
                'avg_score': hc_avg_score if hillclimb_results else None,
                'avg_cost': hc_avg_cost if hillclimb_results else None,
                'avg_success_rate': hc_avg_success if hillclimb_results else None,
                'avg_improvement_simple': hc_avg_improvement_simple if hillclimb_results else None,
                'avg_improvement_advanced': hc_avg_improvement_advanced if hillclimb_results else None,
                'configurations': [row['experiment'] for row in hillclimb_results]
            } if hillclimb_results else None
        },
        'detailed_results': all_experiment_results
    }
    
    os.makedirs('comprehensive_logs/cross_experiment_analysis', exist_ok=True)
    with open('comprehensive_logs/cross_experiment_analysis/tsp_funsearch_vs_hillclimb_comparison.json', 'w') as f:
        json.dump(cross_experiment_analysis, f, indent=2)
    
    # 找出最优配置
    best_by_score = max(comparison_table, key=lambda x: x['avg_best_score'])
    best_by_efficiency = max(comparison_table, key=lambda x: x['avg_best_score'] / max(x['avg_cost'], 1e-8))
    most_stable = max(comparison_table, key=lambda x: x['success_rate'])
    best_improvement_simple = max(comparison_table, key=lambda x: x['improvement_simple'])
    best_improvement_advanced = max(comparison_table, key=lambda x: x['improvement_advanced'])
    
    print(f"\n🏆 最优配置分析:")
    print(f"  最高分数: {best_by_score['experiment']} ({best_by_score['method_type']}) - 分数: {best_by_score['avg_best_score']:.3f}")
    print(f"  最高效率: {best_by_efficiency['experiment']} ({best_by_efficiency['method_type']}) - 分数/成本: {best_by_efficiency['avg_best_score']/max(best_by_efficiency['avg_cost'], 1e-8):.0f}")
    print(f"  最稳定: {most_stable['experiment']} ({most_stable['method_type']}) - 成功率: {most_stable['success_rate']:.3f}")
    print(f"  最大改进(简单): {best_improvement_simple['experiment']} ({best_improvement_simple['method_type']}) - 改进: {best_improvement_simple['improvement_simple']:.1f}%")
    print(f"  最大改进(高级): {best_improvement_advanced['experiment']} ({best_improvement_advanced['method_type']}) - 改进: {best_improvement_advanced['improvement_advanced']:.1f}%")
    
    return cross_experiment_analysis

def main():
    """主函数"""
    print("开始TSP综合多维度评价指标实验 (FunSearch vs HillClimb)...")
    
    # 创建日志目录
    os.makedirs('comprehensive_logs', exist_ok=True)
    
    # 运行对比实验
    cross_experiment_results = run_comparative_experiments()
    
    print(f"\n✅ 所有TSP实验完成!")
    print(f"📁 详细结果保存在: comprehensive_logs/")
    print(f"📊 跨实验对比报告: comprehensive_logs/cross_experiment_analysis/tsp_funsearch_vs_hillclimb_comparison.json")

if __name__ == '__main__':
    main() 