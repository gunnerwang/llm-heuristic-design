# Comprehensive TSP Evaluation System

This directory contains a comprehensive evaluation system for TSP (Traveling Salesman Problem) optimization using LLM-based methods like EoH (Evolution of Heuristics). The system provides detailed metrics collection, cost analysis, stability testing, and comparative analysis capabilities.

**Key Feature**: The `ComprehensiveProfiler` inherits from the LLM4AD framework's `ProfilerBase` class, ensuring full compatibility with existing EoH methods while providing enhanced functionality.

## Features

### 🔍 Comprehensive Metrics Collection
- **Performance Tracking**: Best/worst/mean scores, convergence analysis, score distributions
- **Cost Analysis**: Token usage, API costs, cost per evaluation, cost efficiency metrics
- **Stability Analysis**: Multiple runs with confidence intervals, coefficient of variation
- **Timing Analysis**: Evaluation times, LLM response times, total experiment duration
- **Solution Diversity**: Analysis of solution space exploration and diversity metrics

### 📊 Enhanced Evaluation Capabilities
- **Stability Testing**: Run each solution multiple times to assess reliability
- **Baseline Comparison**: Compare against simple greedy and advanced heuristic baselines
- **Statistical Analysis**: T-tests, Mann-Whitney U tests for solution comparison
- **Trend Analysis**: Performance trends over generations with linear regression
- **Framework Compatibility**: Full compatibility with LLM4AD's ProfilerBase class
- **Intermediate Solution Tracking**: Automatically captures all EoH optimization intermediate solutions
- **Generation Context**: Tracks which generation each solution belongs to for detailed analysis

### 💰 Cost Management
- **Token Tracking**: Detailed tracking of prompt/completion tokens
- **Cost Calculation**: Accurate cost calculation for different LLM models
- **Budget Optimization**: Tools for cost-aware experiment design
- **Efficiency Metrics**: Cost per score improvement, tokens per evaluation

### 🔬 Experimental Framework
- **Multiple Configurations**: Test different EoH parameter settings
- **Cross-Experiment Comparison**: Compare results across different setups
- **Detailed Reporting**: JSON exports with comprehensive experiment data
- **Visualization Ready**: Data formatted for easy plotting and analysis

## Quick Start

### Basic Usage

```python
from run_comprehensive_tsp_evaluation import run_comprehensive_experiment

# Define your experiment
experiment_config = {
    'name': 'tsp_test_experiment',
    'llm_config': {
        'host': 'api.deepseek.com',
        'key': 'your-api-key',
        'model': 'deepseek-chat',
        'kwargs': {'timeout': 300}
    },
    'method_config': {
        'max_sample_nums': 50,
        'max_generations': 5,
        'pop_size': 8,
        'use_memetic': True,
        'use_reflection': True,
        'debug_mode': False
    }
}

# Run the experiment
results = run_comprehensive_experiment(
    experiment_name=experiment_config['name'],
    llm_config=experiment_config['llm_config'],
    method_config=experiment_config['method_config'],
    num_runs=3
)
```

### Running Complete Comparative Analysis

```bash
cd llm4ad-main/example/tsp_construct
python run_comprehensive_tsp_evaluation.py
```

This will run three different experiment configurations and generate a comprehensive comparison report.

## File Structure

```
tsp_construct/
├── comprehensive_metrics.py           # Core metrics collection system
├── enhanced_llm_wrapper.py           # LLM wrapper with token tracking
├── enhanced_evaluation.py            # Enhanced TSP evaluation with stability analysis
├── run_comprehensive_tsp_evaluation.py # Main experiment runner
├── run_eohmeme.py                     # Basic TSP EoH example
└── README_comprehensive_metrics.md   # This documentation
```

## Key Components

### 1. ComprehensiveProfiler (`comprehensive_metrics.py`)

The heart of the metrics system, providing enhanced functionality while maintaining compatibility with the LLM4AD framework:

```python
from comprehensive_metrics import ComprehensiveProfiler

profiler = ComprehensiveProfiler(
    log_dir="logs",
    track_tokens=True
)

# Log evaluations
profiler.log_evaluation(score=score, generation=gen, evaluation_time=time)

# Log LLM calls
profiler.log_llm_call(tokens_used=1000, cost=0.001, call_time=1.5)

# Generate final metrics
profiler.calculate_final_metrics()
profiler.save_metrics("metrics.json")
```

**Key Features:**
- Inherits from `ProfilerBase` for full LLM4AD compatibility
- Enhanced metrics collection beyond base profiler
- Token usage tracking and cost analysis
- Stability and efficiency metrics
- Cross-experiment comparison capabilities

### 2. Enhanced LLM Wrapper (`enhanced_llm_wrapper.py`)

Provides token tracking and cost analysis:

```python
from enhanced_llm_wrapper import TokenAwareLLMFactory

enhanced_llm = TokenAwareLLMFactory.create_enhanced_llm(
    host='api.deepseek.com',
    key='your-key',
    model='deepseek-chat',
    profiler=profiler
)

# Use like normal LLM, but with automatic tracking
response = enhanced_llm.generate(messages)

# Get detailed usage summary
usage = enhanced_llm.get_usage_summary()
print(f"Total cost: ${usage['cost_analysis']['total_cost_usd']:.6f}")
```

### 3. Enhanced TSP Evaluation (`enhanced_evaluation.py`)

Provides stability analysis and detailed solution metrics with automatic intermediate solution tracking:

```python
from enhanced_evaluation import EnhancedTSPEvaluation

enhanced_task = EnhancedTSPEvaluation(
    profiler=profiler,
    stability_runs=5,
    collect_detailed_metrics=True
)

# Evaluate with stability analysis (for baselines)
result = enhanced_task.evaluate_with_stability_analysis(
    evaluator=your_tsp_solver,
    function_id="test_solution"
)

print(f"Stability score: {result['stability_metrics']['stability_score']:.3f}")

# The evaluate method is automatically called by EoH and captures intermediate solutions
# No additional setup needed - all EoH solutions are automatically tracked!
```

**Key Features:**
- **Automatic Tracking**: Overrides base `evaluate` method to capture all EoH intermediate solutions
- **Generation Context**: Tracks which generation each solution belongs to
- **Dual Mode**: Handles both stability analysis (multiple runs) and single evaluations
- **Comprehensive Analysis**: Provides solution diversity analysis comparing baselines vs EoH solutions
- **No Double Tracking**: Intelligent flag system prevents duplicate entries

## Experiment Configurations

The system includes three pre-configured experiments:

### 1. Standard Configuration
- **Purpose**: Balanced exploration and exploitation
- **Parameters**: 50 samples, 5 generations, pop_size=8
- **Features**: Memetic search, reflection, memory
- **Best for**: General TSP optimization

### 2. Aggressive Configuration  
- **Purpose**: Maximum exploration with higher computational cost
- **Parameters**: 80 samples, 8 generations, pop_size=12
- **Features**: High memetic intensity, large memory
- **Best for**: Complex TSP instances, research experiments

### 3. Conservative Configuration
- **Purpose**: Cost-efficient with minimal LLM usage
- **Parameters**: 30 samples, 3 generations, pop_size=6
- **Features**: No memetic search, no reflection
- **Best for**: Budget-constrained experiments

## Output Structure

### Log Directory Structure
```
comprehensive_logs/
├── experiment_name/
│   ├── run_0/
│   │   ├── comprehensive_metrics.json
│   │   ├── llm_usage.json
│   │   └── evaluation_report.json
│   ├── run_1/
│   └── experiment_comparison.json
└── cross_experiment_analysis/
    └── complete_comparison.json
```

### Key Output Files

#### 1. Comprehensive Metrics (`comprehensive_metrics.json`)
- Performance metrics (best/mean/std scores)
- Token usage and costs
- Timing analysis
- Stability metrics
- Efficiency metrics

#### 2. LLM Usage Log (`llm_usage.json`)
- Detailed call-by-call logs
- Token breakdown (prompt/completion)
- Cost analysis
- Response time statistics

#### 3. Evaluation Report (`evaluation_report.json`)
- Solution diversity analysis
- Performance trends
- Best solutions summary
- Statistical comparisons

#### 4. Cross-Experiment Comparison (`complete_comparison.json`)
- Performance comparison across configurations
- Cost-benefit analysis
- Recommendations for optimal settings

## Baseline Comparisons

The system includes two baseline TSP solvers for comparison:

### 1. Simple Greedy Baseline
```python
def simple_greedy_solver(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Always chooses the nearest unvisited node"""
    distances = distance_matrix[current_node][unvisited_nodes]
    return unvisited_nodes[np.argmin(distances)]
```

### 2. Advanced Baseline (2-step Lookahead)
```python
def nearest_neighbor_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Uses 2-step lookahead for better decision making"""
    # Implementation with lookahead logic
    pass
```

## Cost Analysis

### Supported Models and Pricing

The system includes pricing for major LLM providers:

| Model | Input ($/1K tokens) | Output ($/1K tokens) |
|-------|-------------------|---------------------|
| GPT-4 | $0.030 | $0.060 |
| GPT-4 Turbo | $0.010 | $0.030 |
| GPT-3.5 Turbo | $0.0015 | $0.002 |
| DeepSeek Chat | $0.00014 | $0.00028 |
| Claude-3 Haiku | $0.00025 | $0.00125 |
| Claude-3 Sonnet | $0.003 | $0.015 |

### Cost Estimation

```python
from enhanced_llm_wrapper import estimate_experiment_cost

cost_estimate = estimate_experiment_cost(
    num_generations=5,
    pop_size=8,
    num_evaluations=50,
    model='deepseek-chat',
    tokens_per_call=1000
)

print(f"Estimated cost: ${cost_estimate['cost_estimate']['total_cost_usd']:.6f}")
```

## Advanced Usage

### Custom Baseline Implementation

```python
def create_custom_baseline():
    def custom_tsp_solver(current_node, destination_node, unvisited_nodes, distance_matrix):
        # Your custom TSP heuristic here
        return selected_node
    return custom_tsp_solver

# Use in evaluation
custom_baseline = create_custom_baseline()
result = enhanced_task.evaluate_with_stability_analysis(
    custom_baseline,
    function_id="custom_baseline"
)
```

### Statistical Solution Comparison

```python
# Compare two solutions statistically
comparison = enhanced_task.compare_solutions(
    solution_a_id="eoh_solution",
    solution_b_id="greedy_baseline"
)

print(f"T-test p-value: {comparison['statistical_tests']['t_test']['p_value']:.6f}")
print(f"Effect size: {comparison['practical_significance']['effect_size']:.3f}")
```

### Custom Metrics Tracking

```python
# Add custom solution data
profiler.log_evaluation(
    score=score,
    generation=gen,
    solution_data={
        'tour_length': tour_length,
        'improvement_type': 'memetic',
        'custom_metric': custom_value
    }
)
```

## Performance Metrics Explained

### Stability Score
- **Range**: 0-1 (higher is better)
- **Calculation**: `1 / (1 + coefficient_of_variation)`
- **Interpretation**: Measures consistency across multiple runs

### Efficiency Metrics
- **Score per Dollar**: `best_score / total_cost_usd`
- **Score per Evaluation**: `best_score / total_evaluations`
- **Evaluations per Second**: `total_evaluations / total_time`

### Improvement Metrics
- **Relative to Baseline**: `(eoh_score - baseline_score) / |baseline_score| * 100%`
- **Statistical Significance**: P-values from t-tests and Mann-Whitney U tests

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install scipy numpy
   ```

2. **API Key Issues**
   - Ensure your API key is valid and has sufficient credits
   - Check rate limits for your chosen model

3. **Memory Issues**
   - Reduce `stability_runs` or `num_runs` for large experiments
   - Use conservative configuration for limited resources

4. **Timeout Errors**
   - Increase timeout in LLM kwargs: `{'timeout': 600}`
   - Consider using faster models for initial testing

### Performance Optimization

1. **Reduce Costs**
   - Use DeepSeek models for cost-effective experiments
   - Reduce `max_sample_nums` and `max_generations`
   - Set `stability_runs=1` for quick tests

2. **Improve Reliability**
   - Increase `stability_runs` to 5 or more
   - Use multiple `num_runs` for statistical significance
   - Enable all EoH features for better convergence

## Citation

If you use this comprehensive evaluation system in your research, please cite:

```bibtex
@article{liu2024llm4ad,
  title={LLM4AD: A Platform for Algorithm Design with Large Language Model},
  author={Liu, Fei and Zhang, Rui and Xie, Zhuoliang and Sun, Rui and Li, Kai and Lin, Xi and Wang, Zhenkun and Lu, Zhichao and Zhang, Qingfu},
  journal={arXiv preprint arXiv:2412.17287},
  year={2024}
}
```

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the example output files for expected format
3. Contact the LLM4AD team at http://www.llm4ad.com/contact.html 