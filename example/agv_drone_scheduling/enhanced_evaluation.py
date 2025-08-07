import numpy as np
import time
import json
import statistics
from typing import Dict, List, Any, Callable, Tuple, Optional
from collections import defaultdict
from llm4ad.task.optimization.agv_drone_scheduling.evaluation import VehicleSchedulingEvaluation
from comprehensive_metrics import ComprehensiveProfiler

class EnhancedVehicleSchedulingEvaluation(VehicleSchedulingEvaluation):
    """增强版AGV无人机调度评估器，支持多维度指标分析"""
    
    # Class-level instance counter
    _instance_count = 0
    _all_instances = []
    
    def __init__(self, 
                 profiler: Optional[ComprehensiveProfiler] = None,
                 stability_runs: int = 5,
                 collect_detailed_metrics: bool = True,
                 timeout_seconds: int = 60,  # For VehicleSchedulingEvaluation
                 n_instance: int = 10,       # For VehicleSchedulingEvaluation
                 **kwargs): # For Evaluation base class (e.g. random_seed, use_protected_div)
        """
        Args:
            profiler: 综合指标记录器
            stability_runs: 稳定性测试运行次数
            collect_detailed_metrics: 是否收集详细指标
            timeout_seconds: Maximum allowed time (in seconds) for the evaluation process (passed to VehicleSchedulingEvaluation)
            n_instance: Number of problem instances to evaluate (passed to VehicleSchedulingEvaluation)
            **kwargs: Additional arguments for the llm4ad.base.Evaluation base class.
        """
        super().__init__(
            timeout_seconds=timeout_seconds, 
            n_instance=n_instance,
            **kwargs # Pass through kwargs to VehicleSchedulingEvaluation -> Evaluation
        )
        
        # Update instance counter
        EnhancedVehicleSchedulingEvaluation._instance_count += 1
        self._instance_id = EnhancedVehicleSchedulingEvaluation._instance_count
        EnhancedVehicleSchedulingEvaluation._all_instances.append(self)
        
        self.profiler = profiler
        self.stability_runs = stability_runs
        self.collect_detailed_metrics = collect_detailed_metrics
        
        # evaluation_history and other specific attributes are initialized here
        self.evaluation_history = []
        print(f"DEBUG: EnhancedVehicleSchedulingEvaluation.__init__ - Instance #{self._instance_id} created")
        print(f"DEBUG: evaluation_history initialized with id: {id(self.evaluation_history)}")
        print(f"DEBUG: EnhancedVehicleSchedulingEvaluation instance id: {id(self)}")
        print(f"DEBUG: Total instances created so far: {EnhancedVehicleSchedulingEvaluation._instance_count}")
        import sys
        sys.stdout.flush()
        
        self.function_performance_cache = {}
        
        # 稳定性分析相关
        self.repeated_evaluations = defaultdict(list)
        self.score_distributions = defaultdict(list)
        
        # 质量分析相关
        self.convergence_tracking = []
        self.solution_diversity_metrics = []
        
        self.call_log = [] # Added for debugging calls from EoH
        
    def evaluate_with_stability_analysis(self, scheduling_func: Callable, 
                                       function_id: str = None,
                                       function_str: Optional[str] = None,
                                       sample_time: float = 0.0) -> Dict[str, Any]:
        """
        带稳定性分析的评估
        
        Args:
            scheduling_func: 调度函数
            function_id: 函数标识符（用于缓存和分析）
            function_str: 函数字符串表示
            sample_time: 函数生成或采样时间
            
        Returns:
            包含多维度指标的评估结果
        """
        print(f"DEBUG: Starting evaluate_with_stability_analysis for function_id: {function_id}")
        print(f"DEBUG: EnhancedVehicleSchedulingEvaluation instance id: {id(self)}")
        print(f"DEBUG: evaluation_history id at start: {id(self.evaluation_history)}")
        print(f"DEBUG: evaluation_history length at start: {len(self.evaluation_history)}")
        import sys
        sys.stdout.flush()  # Force flush the output
        
        eval_start_time = time.time()
        
        scores = []
        detailed_results = []
        
        print(f"DEBUG: Starting {self.stability_runs} stability runs")
        for run_idx in range(self.stability_runs):
            run_start_individual = time.time()
            current_score = None
            success_flag = False
            error_msg = None
            try:
                print(f"DEBUG: Starting evaluation run {run_idx}")
                import sys
                sys.stdout.flush()  # Force flush the output
                
                current_score = super().evaluate(scheduling_func)
                print(f"DEBUG: Evaluation run {run_idx} completed with score: {current_score}")
                sys.stdout.flush()  # Force flush the output
                
                run_time_individual = time.time() - run_start_individual
                success_flag = True
                
                if current_score is not None:
                    scores.append(current_score)
                
                detailed_results.append({
                    'run': run_idx,
                    'score': current_score,
                    'evaluation_time': run_time_individual,
                    'success': True
                })
                
            except Exception as e:
                print(f"DEBUG: Evaluation run {run_idx} failed with error: {str(e)}")
                print(f"DEBUG: Exception type: {type(e).__name__}")
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                import sys
                sys.stdout.flush()  # Force flush the output
                
                run_time_individual = time.time() - run_start_individual
                error_msg = str(e)
                detailed_results.append({
                    'run': run_idx,
                    'score': None,
                    'evaluation_time': run_time_individual,
                    'success': False,
                    'error': error_msg
                })

            if self.profiler:
                profiler_sample_order = len(self.profiler.snapshots) if hasattr(self.profiler, 'snapshots') else 0

                self.profiler.record_evaluation(
                    score=current_score if success_flag and current_score is not None else 0.0,
                    evaluation_time=run_time_individual,
                    sample_order=profiler_sample_order,
                    success=success_flag and current_score is not None,
                    function_str=function_str,
                    sample_time=sample_time,
                )
        
        print(f"DEBUG: Completed all {self.stability_runs} runs, calculating metrics")
        import sys
        sys.stdout.flush()  # Force flush the output
        
        total_evaluation_time_for_func = time.time() - eval_start_time
        
        valid_scores = [s for s in scores if s is not None]
        print(f"DEBUG: Valid scores: {valid_scores}")
        sys.stdout.flush()  # Force flush the output

        # 进一步过滤掉NaN和非有限值
        finite_valid_scores = [s for s in valid_scores if np.isfinite(s) and not np.isnan(s)]
        print(f"DEBUG: Finite valid scores: {finite_valid_scores}")
        sys.stdout.flush()

        print("DEBUG: Calculating stability metrics")
        sys.stdout.flush()  # Force flush the output
        stability_metrics = self._calculate_stability_metrics(finite_valid_scores)
        
        print("DEBUG: Calculating quality metrics")
        sys.stdout.flush()  # Force flush the output
        quality_metrics = self._calculate_quality_metrics(finite_valid_scores)
        
        print("DEBUG: Finished calculating metrics")
        sys.stdout.flush()  # Force flush the output
        
        result = {
            'function_id': function_id,
            'function_str': function_str,
            'sample_time': sample_time,
            'timestamp': time.time(),
            'scores': valid_scores,  # Keep original valid scores for record
            'finite_scores': finite_valid_scores,  # Also keep finite scores
            'detailed_results': detailed_results,
            'total_evaluation_time': total_evaluation_time_for_func,
            'stability_metrics': stability_metrics,
            'quality_metrics': quality_metrics,
            'summary': {
                'best_score': max(finite_valid_scores) if finite_valid_scores else float('-inf'),
                'worst_score': min(finite_valid_scores) if finite_valid_scores else float('inf'),
                'mean_score': np.mean(finite_valid_scores) if finite_valid_scores else float('nan'),
                'median_score': np.median(finite_valid_scores) if finite_valid_scores else float('nan'),
                'success_rate': len(valid_scores) / self.stability_runs if self.stability_runs > 0 else 0,
                'finite_success_rate': len(finite_valid_scores) / self.stability_runs if self.stability_runs > 0 else 0
            }
        }
        
        print(f"DEBUG: About to append to evaluation_history (current length: {len(self.evaluation_history)})")
        print(f"DEBUG: evaluation_history object id: {id(self.evaluation_history)}")
        print(f"DEBUG: evaluation_history type: {type(self.evaluation_history)}")
        print(f"DEBUG: evaluation_history contents: {[r.get('function_id', 'unknown') for r in self.evaluation_history]}")
        print(f"DEBUG: Result to append - function_id: {result.get('function_id', 'unknown')}")
        import sys
        sys.stdout.flush()  # Force flush the output
        
        # Store original length for comparison
        original_length = len(self.evaluation_history)
        
        self.evaluation_history.append(result)
        
        # Check if append actually worked
        new_length = len(self.evaluation_history)
        print(f'DEBUG: Evaluation history updated from length {original_length} to {new_length}')
        print(f"DEBUG: evaluation_history object id after append: {id(self.evaluation_history)}")
        print(f"DEBUG: Last item in history: {self.evaluation_history[-1].get('function_id', 'unknown') if self.evaluation_history else 'None'}")
        print(f'DEBUG: Added evaluation for function_id: {function_id}, best_score: {result["summary"]["best_score"]:.3f}')
        sys.stdout.flush()  # Force flush the output
        
        if function_id and valid_scores:
            self.repeated_evaluations[function_id].extend(valid_scores)
            self.score_distributions[function_id] = valid_scores
        
        print(f"DEBUG: Finished evaluate_with_stability_analysis for function_id: {function_id}")
        sys.stdout.flush()  # Force flush the output
        return result
    
    def _calculate_stability_metrics(self, scores: List[float]) -> Dict[str, float]:
        """计算稳定性相关指标"""
        # 过滤掉NaN和非有限值
        finite_scores = [score for score in scores if np.isfinite(score) and not np.isnan(score)]
        
        if not finite_scores or len(finite_scores) < 2:
            return {
                'stability_score': 0.0,
                'coefficient_of_variation': float('nan'),
                'score_range': 0.0,
                'consistency_score': 0.0,
                'standard_deviation': 0.0 if len(finite_scores) <= 1 else float('nan'),
                'interquartile_range': 0.0 if len(finite_scores) <= 1 else float('nan'),
                'finite_score_count': len(finite_scores),
                'total_score_count': len(scores)
            }
        
        mean_score = np.mean(finite_scores)
        std_score = np.std(finite_scores)
        
        # 变异系数（CV）
        cv = std_score / abs(mean_score) if mean_score != 0 else float('inf')
        
        # 稳定性分数（基于相对标准差的倒数）
        stability_score = 1.0 / (1.0 + cv) if cv != float('inf') else 0.0
        
        # 分数范围
        score_range = max(finite_scores) - min(finite_scores)
        
        # 一致性分数（基于四分位距）
        if len(finite_scores) >= 4:  # 需要足够的数据点计算四分位距
            q1 = np.percentile(finite_scores, 25)
            q3 = np.percentile(finite_scores, 75)
            iqr = q3 - q1
            consistency_score = 1.0 / (1.0 + iqr / max(abs(mean_score), 1e-8))
        else:
            iqr = score_range  # 如果数据点不足，使用范围作为替代
            consistency_score = 1.0 / (1.0 + iqr / max(abs(mean_score), 1e-8))
        
        return {
            'stability_score': stability_score,
            'coefficient_of_variation': cv,
            'score_range': score_range,
            'consistency_score': consistency_score,
            'standard_deviation': std_score,
            'interquartile_range': iqr,
            'finite_score_count': len(finite_scores),
            'total_score_count': len(scores)
        }
    
    def _calculate_quality_metrics(self, scores: List[float]) -> Dict[str, float]:
        """计算质量相关指标"""
        print(f"DEBUG: _calculate_quality_metrics called with {len(scores)} scores")
        if not scores:
            print("DEBUG: No scores provided, returning default metrics")
            return {
                'mean': float('nan'), 'median': float('nan'), 'std': float('nan'),
                'min': float('-inf'), 'max': float('inf'), 'range': float('nan'),
                'quality_grade': 'Unknown' 
            }
        
        # 过滤掉NaN和非有限值
        finite_scores = [score for score in scores if np.isfinite(score) and not np.isnan(score)]
        
        if not finite_scores:
            print("DEBUG: No finite scores available, returning default metrics")
            return {
                'mean': float('nan'), 'median': float('nan'), 'std': float('nan'),
                'min': float('-inf'), 'max': float('inf'), 'range': float('nan'),
                'quality_grade': 'Unknown',
                'finite_score_count': 0,
                'total_score_count': len(scores)
            }
        
        print("DEBUG: Calculating basic metrics")
        # 基础统计指标 - 只使用有限值
        metrics = {
            'mean': np.mean(finite_scores),
            'median': np.median(finite_scores),
            'std': np.std(finite_scores),
            'min': min(finite_scores),
            'max': max(finite_scores),
            'range': max(finite_scores) - min(finite_scores),
            'finite_score_count': len(finite_scores),
            'total_score_count': len(scores)
        }
        
        print("DEBUG: Calculating percentiles")
        # 百分位数 - 只对有限值计算
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            metrics[f'percentile_{p}'] = np.percentile(finite_scores, p)
        
        print("DEBUG: Calculating skewness and kurtosis")
        # 偏度和峰度 - 只对有限值计算
        if len(finite_scores) >= 3:
            try:
                from scipy import stats
                metrics['skewness'] = stats.skew(finite_scores)
                if len(finite_scores) >= 4:
                    metrics['kurtosis'] = stats.kurtosis(finite_scores)
            except ImportError:
                pass  # scipy not available
        
        print("DEBUG: Calculating quality grade")
        # 质量等级评估 - 修复空历史数据的问题
        if len(finite_scores) >= 2:
            mean_score = metrics['mean']
            
            print(f"DEBUG: Current evaluation_history length: {len(self.evaluation_history)}")
            # 获取历史评估的平均分数，过滤掉None值和NaN值
            historical_mean_scores = []
            for r in self.evaluation_history:
                mean_val = r['summary']['mean_score']
                if mean_val is not None and np.isfinite(mean_val) and not np.isnan(mean_val):
                    historical_mean_scores.append(mean_val)
            
            print(f"DEBUG: Historical mean scores: {len(historical_mean_scores)} valid scores")
            
            # 只有当有足够的历史数据时才进行百分位数比较
            if len(historical_mean_scores) >= 5:  # 至少需要5个历史数据点
                try:
                    p90 = np.percentile(historical_mean_scores, 90)
                    p70 = np.percentile(historical_mean_scores, 70)
                    p50 = np.percentile(historical_mean_scores, 50)
                    
                    if mean_score >= p90:
                        metrics['quality_grade'] = 'A'
                    elif mean_score >= p70:
                        metrics['quality_grade'] = 'B'
                    elif mean_score >= p50:
                        metrics['quality_grade'] = 'C'
                    else:
                        metrics['quality_grade'] = 'D'
                except (IndexError, ValueError):
                    # 如果计算百分位数失败，使用默认等级
                    metrics['quality_grade'] = 'Unknown'
            else:
                # 历史数据不足，使用基于当前分数的简单分级
                if mean_score > 0:
                    metrics['quality_grade'] = 'B'  # 默认给一个中等等级
                else:
                    metrics['quality_grade'] = 'C'
        else:
            metrics['quality_grade'] = 'Unknown'
        
        print(f"DEBUG: Quality metrics calculation completed, grade: {metrics.get('quality_grade', 'Unknown')}")
        return metrics
    
    def analyze_solution_diversity(self) -> Dict[str, Any]:
        """分析解的多样性"""
        if len(self.evaluation_history) < 2:
            return {'error': 'Insufficient data for diversity analysis'}
        
        all_scores = []
        function_means = []
        
        for result in self.evaluation_history:
            if result['summary']['mean_score'] is not None:
                all_scores.extend(result['scores'])
                function_means.append(result['summary']['mean_score'])
        
        # 过滤掉非有限值和NaN值
        finite_function_means = [score for score in function_means if np.isfinite(score) and not np.isnan(score)]
        finite_all_scores = [score for score in all_scores if np.isfinite(score) and not np.isnan(score)]
        
        if len(finite_function_means) < 2:
            return {'error': 'Insufficient finite scores for diversity analysis',
                   'total_functions': len(function_means),
                   'finite_functions': len(finite_function_means)}
        
        # 计算多样性指标
        diversity_metrics = {
            'total_unique_scores': len(set(finite_all_scores)),
            'mean_diversity': np.std(finite_function_means),
            'score_spread': max(finite_function_means) - min(finite_function_means),
            'diversity_ratio': len(set(finite_function_means)) / len(finite_function_means),
            'entropy': self._calculate_score_entropy(finite_function_means),
            'finite_score_ratio': len(finite_function_means) / len(function_means) if function_means else 0
        }
        
        # 聚类分析（简单的基于分数的聚类）
        clusters = self._simple_score_clustering(finite_function_means)
        diversity_metrics['clusters'] = clusters
        
        return diversity_metrics
    
    def _calculate_score_entropy(self, scores: List[float]) -> float:
        """计算分数分布的熵"""
        # 过滤掉非有限值（-inf, inf, NaN）
        finite_scores = [score for score in scores if np.isfinite(score) and not np.isnan(score)]
        
        if len(finite_scores) < 2:
            return 0.0  # 如果没有足够的有限分数，返回0熵
        
        # 将分数离散化为区间
        bins = min(10, len(finite_scores))  # 确保bins不超过数据点数量
        hist, _ = np.histogram(finite_scores, bins=bins)
        hist = hist / np.sum(hist)  # 归一化为概率
        
        # 计算熵
        entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])
        return entropy
    
    def _simple_score_clustering(self, scores: List[float], n_clusters: int = 3) -> Dict[str, Any]:
        """简单的基于分数的聚类分析"""
        # 过滤掉非有限值（-inf, inf, NaN）
        finite_scores = [score for score in scores if np.isfinite(score) and not np.isnan(score)]
        
        if len(finite_scores) < n_clusters:
            return {'error': f'Not enough finite scores for clustering. Found {len(finite_scores)}, need {n_clusters}',
                   'total_scores': len(scores),
                   'finite_scores': len(finite_scores)}
        
        # 使用K-means聚类
        try:
            from sklearn.cluster import KMeans
            scores_array = np.array(finite_scores).reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scores_array)
            
            clusters = {}
            for i in range(n_clusters):
                cluster_scores = [finite_scores[j] for j, label in enumerate(cluster_labels) if label == i]
                clusters[f'cluster_{i}'] = {
                    'size': len(cluster_scores),
                    'mean': np.mean(cluster_scores),
                    'std': np.std(cluster_scores),
                    'center': kmeans.cluster_centers_[i][0]
                }
            
            return clusters
            
        except ImportError:
            # 如果没有sklearn，使用简单的基于百分位数的分组
            scores_sorted = sorted(finite_scores)
            n = len(scores_sorted)
            
            clusters = {}
            for i in range(n_clusters):
                start_idx = i * n // n_clusters
                end_idx = (i + 1) * n // n_clusters
                cluster_scores = scores_sorted[start_idx:end_idx]
                
                clusters[f'cluster_{i}'] = {
                    'size': len(cluster_scores),
                    'mean': np.mean(cluster_scores),
                    'std': np.std(cluster_scores),
                    'range': [min(cluster_scores), max(cluster_scores)]
                }
            
            return clusters
    
    def generate_performance_trend_analysis(self) -> Dict[str, Any]:
        """生成性能趋势分析"""
        if len(self.evaluation_history) < 3:
            return {'error': 'Insufficient data for trend analysis'}
        
        timestamps = [r['timestamp'] for r in self.evaluation_history]
        mean_scores = [r['summary']['mean_score'] for r in self.evaluation_history 
                      if r['summary']['mean_score'] is not None and np.isfinite(r['summary']['mean_score']) and not np.isnan(r['summary']['mean_score'])]
        best_scores = [r['summary']['best_score'] for r in self.evaluation_history 
                      if r['summary']['best_score'] is not None and np.isfinite(r['summary']['best_score']) and not np.isnan(r['summary']['best_score'])]
        stability_scores = [r['stability_metrics']['stability_score'] for r in self.evaluation_history 
                           if np.isfinite(r['stability_metrics']['stability_score']) and not np.isnan(r['stability_metrics']['stability_score'])]
        
        # 趋势分析
        trend_analysis = {
            'total_evaluations': len(self.evaluation_history),
            'time_span': max(timestamps) - min(timestamps),
            'score_trends': {
                'mean_score_trend': self._calculate_trend(mean_scores),
                'best_score_trend': self._calculate_trend(best_scores),
                'stability_trend': self._calculate_trend(stability_scores)
            },
            'improvement_rate': self._calculate_improvement_rate(best_scores),
            'convergence_analysis': self._analyze_convergence(best_scores)
        }
        
        return trend_analysis
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """计算数值序列的趋势"""
        if len(values) < 2:
            return {'slope': 0.0, 'correlation': 0.0, 'direction': 'insufficient_data'}
        
        # 过滤掉非有限值和NaN值
        finite_values = [v for v in values if np.isfinite(v) and not np.isnan(v)]
        if len(finite_values) < 2:
            return {'slope': 0.0, 'correlation': 0.0, 'direction': 'insufficient_finite_data',
                   'total_values': len(values), 'finite_values': len(finite_values)}
        
        x = np.arange(len(finite_values))
        
        # 计算相关性
        try:
            correlation = np.corrcoef(x, finite_values)[0, 1] if len(finite_values) > 1 else 0.0
        except:
            correlation = 0.0
        
        # 简单线性回归计算斜率
        try:
            slope = np.polyfit(x, finite_values, 1)[0] if len(finite_values) > 1 else 0.0
        except:
            slope = 0.0
        
        return {
            'slope': slope,
            'correlation': correlation,
            'direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
        }
    
    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """计算改进率"""
        if len(scores) < 2:
            return 0.0
        
        improvements = 0
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1]:
                improvements += 1
        
        return improvements / (len(scores) - 1)
    
    def _analyze_convergence(self, scores: List[float]) -> Dict[str, Any]:
        """分析收敛性"""
        # 过滤掉非有限值和NaN值
        finite_scores = [score for score in scores if np.isfinite(score) and not np.isnan(score)]
        
        if len(finite_scores) < 5:
            return {'status': 'insufficient_data', 'finite_score_count': len(finite_scores), 'total_score_count': len(scores)}
        
        # 检查最后几个分数的变化
        recent_scores = finite_scores[-5:]
        score_variance = np.var(recent_scores)
        
        # 检查是否已收敛（最后几个分数变化很小）
        mean_recent = np.mean(recent_scores)
        is_converged = score_variance < 0.01 * abs(mean_recent) if mean_recent != 0 else score_variance < 0.01
        
        # 找到最佳分数首次出现的位置
        best_score = max(finite_scores)
        first_best_index = finite_scores.index(best_score)
        
        return {
            'status': 'converged' if is_converged else 'not_converged',
            'convergence_generation': first_best_index,
            'recent_variance': score_variance,
            'best_score': best_score,
            'plateau_length': len(finite_scores) - first_best_index,
            'finite_score_count': len(finite_scores),
            'total_score_count': len(scores)
        }
    
    def export_comprehensive_report(self, filename: str = "comprehensive_evaluation_report.json"):
        """导出综合评估报告"""
        report = {
            'evaluation_summary': {
                'total_evaluations': len(self.evaluation_history),
                'stability_runs_per_evaluation': self.stability_runs,
                'total_function_evaluations': sum(len(r['scores']) for r in self.evaluation_history)
            },
            'diversity_analysis': self.analyze_solution_diversity(),
            'trend_analysis': self.generate_performance_trend_analysis(),
            'detailed_results': self.evaluation_history,
            'function_performance_summary': self._generate_function_summary(),
            'evaluator_call_log': self.call_log # Added for debugging
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_function_summary(self) -> Dict[str, Any]:
        """生成函数性能摘要"""
        if not self.evaluation_history:
            return {}
        
        summary = {
            'best_performing_function': None,
            'most_stable_function': None,
            'performance_statistics': {}
        }
        
        best_score = float('-inf')
        best_stability = float('-inf')
        
        for result in self.evaluation_history:
            func_id = result['function_id']
            mean_score = result['summary']['mean_score']
            stability_score = result['stability_metrics']['stability_score']
            
            # 记录性能统计
            summary['performance_statistics'][func_id] = {
                'mean_score': mean_score,
                'best_score': result['summary']['best_score'],
                'stability_score': stability_score,
                'success_rate': result['summary']['success_rate'],
                'evaluation_time': result['total_evaluation_time']
            }
            
            # 找出最佳性能函数
            if mean_score and mean_score > best_score:
                best_score = mean_score
                summary['best_performing_function'] = func_id
            
            # 找出最稳定函数
            if stability_score > best_stability:
                best_stability = stability_score
                summary['most_stable_function'] = func_id
        
        return summary
    
    @classmethod
    def get_all_instances_info(cls):
        """Get information about all created instances and their evaluation history"""
        info = {
            'total_instances': cls._instance_count,
            'instances': []
        }
        
        for i, instance in enumerate(cls._all_instances):
            instance_info = {
                'instance_id': instance._instance_id,
                'object_id': id(instance),
                'evaluation_history_length': len(instance.evaluation_history),
                'evaluation_history_id': id(instance.evaluation_history),
                'function_ids': [r.get('function_id', 'unknown') for r in instance.evaluation_history]
            }
            info['instances'].append(instance_info)
        
        return info
    
    def evaluate(self, scheduling_func: Callable, 
                 function_id: Optional[str] = None,
                 function_str: Optional[str] = None,
                 sample_time: float = 0.0) -> float:
        # Log the call for debugging
        log_entry = {
            "method": "evaluate",
            "timestamp": time.time(),
            "scheduling_func_type": str(type(scheduling_func)),
            "function_id": function_id,
            "function_str_present": function_str is not None,
            "function_str_snippet": function_str[:100] if function_str else None, # Log a snippet
            "sample_time": sample_time
        }
        self.call_log.append(log_entry)
        print(f"DEBUG: EnhancedVehicleSchedulingEvaluation.evaluate CALLED. ID: {function_id}, FuncStr: {function_str is not None}, SampleTime: {sample_time}")

        result = self.evaluate_with_stability_analysis(
            scheduling_func,
            function_id=function_id,
            function_str=function_str,
            sample_time=sample_time
        )
        return result['summary']['best_score'] if result['summary']['best_score'] != float('-inf') else float('-inf')
    
    def evaluate_program(self, program_str: str, callable_func: callable) -> float:
        # Extract function name from program string for identification
        function_name = None
        try:
            # Try to extract function name from the program string
            import ast
            tree = ast.parse(program_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    break
        except:
            function_name = "unknown_function"
        
        # Generate a meaningful function_id based on the evaluation history length and function name
        function_id = f"eoh_{len(self.evaluation_history)}_{function_name}" if function_name else f"eoh_{len(self.evaluation_history)}"
        
        # Log the call for debugging
        log_entry = {
            "method": "evaluate_program",
            "timestamp": time.time(),
            "program_str_present": program_str is not None,
            "program_str_snippet": program_str[:100] if program_str else None, # Log a snippet
            "callable_func_type": str(type(callable_func)),
            "function_id": function_id,
            "sample_time": 0.0,  # We don't have sample time from EoH context
            "extracted_function_name": function_name
        }
        self.call_log.append(log_entry)
        print(f"DEBUG: EnhancedVehicleSchedulingEvaluation.evaluate_program CALLED. ID: {function_id}, ProgStr: {program_str is not None}, FuncName: {function_name}")

        return self.evaluate(
            callable_func,
            function_id=function_id,
            function_str=program_str,
            sample_time=0.0  # We don't have access to the actual sample time from EoH
        )
    
    def evaluate_with_history_return(self, scheduling_func: Callable, 
                                    function_id: str = None,
                                    function_str: Optional[str] = None,
                                    sample_time: float = 0.0) -> Dict[str, Any]:
        """
        Variant of evaluate_with_stability_analysis that returns both the score and the updated evaluation history.
        This is useful when running in multiprocessing mode where evaluation_history changes are lost.
        """
        result = self.evaluate_with_stability_analysis(
            scheduling_func, function_id, function_str, sample_time
        )
        
        # Return both the score and the evaluation history
        return {
            'score': result['summary']['best_score'] if result['summary']['best_score'] != float('-inf') else float('-inf'),
            'evaluation_history': self.evaluation_history,
            'detailed_result': result
        }
    
    def merge_evaluation_history(self, other_history: List[Dict[str, Any]]):
        """Merge evaluation history from another process/instance"""
        for item in other_history:
            # Only add if not already present (check by function_id and timestamp)
            if not any(existing['function_id'] == item['function_id'] and 
                      abs(existing['timestamp'] - item['timestamp']) < 0.1 
                      for existing in self.evaluation_history):
                self.evaluation_history.append(item)
                print(f"DEBUG: Merged evaluation history item: {item['function_id']}") 