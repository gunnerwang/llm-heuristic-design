# Multi-Dimensional Evaluation Metrics System

This system provides a comprehensive multi-dimensional evaluation metrics framework for the LLM4AD project, designed for in-depth analysis and comparison of different algorithms and configurations.

## 🎯 Core Features

### 1. Multi-Dimensional Metrics Collection
- **Cost Metrics**: Token usage, API call costs, time costs
- **Performance Metrics**: Solution quality, convergence speed, success rate
- **Stability Metrics**: Solution stability, coefficient of variation, consistency analysis
- **Efficiency Metrics**: Cost-effectiveness ratio, evaluation efficiency, resource utilization

### 2. Automated Experiment Comparison
- Support for parallel comparative experiments with multiple configurations
- Automatic Pareto frontier analysis generation
- Statistical significance testing
- Cross-experiment consistency validation

### 3. Visualization Dashboard
- Interactive chart displays
- Multi-dimensional comparative analysis
- Trend analysis and convergence visualization
- Export high-quality charts and reports

## 📁 File Structure

```
agv_drone_scheduling/
├── comprehensive_metrics.py           # Core metrics system
├── enhanced_llm_wrapper.py            # Enhanced LLM wrapper
├── enhanced_evaluation.py             # Enhanced evaluator
├── run_eoh_with_comprehensive_metrics.py  # Main execution script
├── visualization_dashboard.py         # Visualization dashboard
└── README_comprehensive_metrics.md    # This documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install required dependencies
pip install numpy matplotlib seaborn pandas scipy scikit-learn
```

### 2. Configure API Keys

Configure your LLM API keys in `run_eoh_with_comprehensive_metrics.py`:

```python
'llm_config': {
    'host': 'api.deepseek.com',
    'key': 'your-api-key-here',  # Replace with your API key
    'model': 'deepseek-chat'
}
```

### 3. Run Experiments

```bash
# Run complete comparative experiments
python run_eoh_with_comprehensive_metrics.py

# Generate visualization reports
python visualization_dashboard.py
```

## 📊 Metrics Details

### 1. Cost-Related Metrics

| Metric Name | Description | Calculation Method |
|-------------|-------------|-------------------|
| **Total Tokens** | Sum of input and output tokens | input_tokens + output_tokens |
| **Total Cost** | API call fees | tokens × price_per_token |
| **Cost Efficiency** | Score obtained per dollar | best_score / total_cost |
| **Token Efficiency** | Score obtained per token | best_score / total_tokens |

### 2. Performance Quality Metrics

| Metric Name | Description | Calculation Method |
|-------------|-------------|-------------------|
| **Best Score** | Highest score across all runs | max(all_scores) |
| **Average Score** | Average score of all valid evaluations | mean(valid_scores) |
| **Score Range** | Difference between highest and lowest scores | max_score - min_score |
| **Relative Improvement** | Improvement percentage relative to baseline | (score - baseline) / baseline × 100% |

### 3. Stability Metrics

| Metric Name | Description | Calculation Method |
|-------------|-------------|-------------------|
| **Coefficient of Variation** | Relative degree of variation | std_score / mean_score |
| **Stability Score** | Stability based on coefficient of variation | 1 / (1 + CV) |
| **Consistency Score** | Consistency based on interquartile range | 1 / (1 + IQR/mean) |
| **Success Rate** | Proportion of successful evaluations | successful_evals / total_evals |

### 4. Convergence Metrics

| Metric Name | Description | Calculation Method |
|-------------|-------------|-------------------|
| **Convergence Generation** | First generation achieving best solution | first_occurrence(best_score) |
| **Improvement Rate** | Proportion of consecutive improvements | improvements / total_generations |
| **Stagnation Count** | Consecutive generations without improvement | consecutive_no_improvement |
| **Convergence Stability** | Variation degree in last few generations | 1 / (1 + var(last_n_generations)) |

## 🔧 Advanced Configuration

### 1. Custom Evaluation Metrics

```python
from comprehensive_metrics import ComprehensiveProfiler

# Create custom profiler
profiler = ComprehensiveProfiler(
    log_dir="custom_logs",
    track_tokens=True
)

# Add custom recording
profiler.record_evaluation(
    score=score,
    evaluation_time=eval_time,
    sample_order=sample_id,
    generation=gen_id,
    success=True
)
```

### 2. Configure Experiment Parameters

```python
# Modify experiment configuration in run_eoh_with_comprehensive_metrics.py
experiments = [
    {
        'name': 'custom_experiment',
        'llm_config': {
            'host': 'your-host',
            'key': 'your-key',
            'model': 'your-model'
        },
        'method_config': {
            'max_sample_nums': 100,  # Maximum sampling number
            'max_generations': 10,   # Maximum generations
            'pop_size': 15,          # Population size
            'stability_runs': 5      # Stability test runs
        }
    }
]
```

### 3. Custom Visualization

```python
from visualization_dashboard import ExperimentVisualizationDashboard

# Create custom dashboard
dashboard = ExperimentVisualizationDashboard("your_logs_dir")
dashboard.load_experiment_data()

# Generate specific charts
dashboard.create_performance_comparison_chart("custom_performance.png")
dashboard.create_cost_analysis_dashboard("custom_cost.png")
```

## 📈 Output Files Description

### 1. Log Directory Structure

```
comprehensive_logs/
├── experiment_name/
│   ├── run_0/
│   │   ├── comprehensive_metrics.json     # Comprehensive metrics
│   │   ├── llm_usage.json                # LLM usage details
│   │   ├── evaluation_report.json        # Evaluation report
│   │   └── detailed_metrics.jsonl        # Detailed logs
│   ├── run_1/
│   └── complete_results.json              # Complete experiment results
└── cross_experiment_analysis/
    └── complete_comparison.json           # Cross-experiment comparison
```

### 2. Visualization Output

```
visualization_output/
├── performance_comparison.png             # Performance comparison chart
├── cost_analysis_dashboard.png           # Cost analysis dashboard
├── stability_analysis.png                # Stability analysis chart
├── convergence_analysis.png              # Convergence analysis chart
├── comprehensive_summary_table.png       # Comprehensive summary table
└── experiment_summary.csv                # Summary data table
```

## 🎯 Use Cases

### 1. Algorithm Performance Comparison
- Compare effects of different LLM models
- Analyze impact of different hyperparameter configurations
- Evaluate algorithm improvement effects

### 2. Cost-Benefit Analysis
- Optimize API usage costs
- Find optimal cost-performance configurations
- Analyze computational resource utilization

### 3. Stability Validation
- Verify algorithm robustness
- Analyze solution consistency
- Evaluate randomness impact

### 4. Convergence Research
- Analyze algorithm convergence characteristics
- Optimize stopping criteria
- Study parameter sensitivity

## 📋 Best Practices

### 1. Experiment Design
- **Multiple Runs**: Run each configuration at least 3-5 times to ensure statistical significance
- **Control Variables**: Change only one parameter at a time to analyze its impact
- **Baseline Comparison**: Always include simple baseline methods for comparison
- **Stability Testing**: Evaluate each solution multiple times to test stability

### 2. Data Analysis
- **Statistical Significance**: Use appropriate statistical testing methods
- **Effect Size**: Focus not only on significance but also on practical effect size
- **Multi-dimensional Analysis**: Consider performance, cost, stability, and other dimensions comprehensively
- **Visualization Validation**: Use charts to validate numerical analysis results

### 3. Result Interpretation
- **Confidence Intervals**: Report confidence intervals rather than point estimates for metrics
- **Relative Comparison**: Use relative improvements rather than absolute values for comparison
- **Contextual Analysis**: Consider trade-offs in different application scenarios
- **Limitation Explanation**: Clearly state experimental limitations and scope of applicability

## 🔍 Troubleshooting

### 1. Common Issues

**Q: What to do when API calls fail?**
A: Check API key configuration, confirm network connection, and review error log details.

**Q: How to handle memory shortage?**
A: Reduce parallel run count, decrease population size, or process experiments in batches.

**Q: What if visualization charts display abnormally?**
A: Check matplotlib font configuration, confirm data format correctness, and update related dependencies.

### 2. Performance Optimization

- **Parallel Evaluation**: Use multiprocessing to accelerate evaluation process
- **Caching Mechanism**: Cache repeated API call results
- **Incremental Saving**: Periodically save intermediate results to avoid data loss
- **Resource Monitoring**: Monitor CPU, memory, and network usage

## 📚 Extension Development

### 1. Adding New Metrics

```python
class CustomMetrics(ComprehensiveMetrics):
    def __init__(self):
        super().__init__()
        self.custom_metric = 0.0
    
    def calculate_custom_metric(self, data):
        # Implement custom metric calculation
        pass
```

### 2. Integrating New LLM

```python
class CustomLLMWrapper(EnhancedLLMWrapper):
    def __init__(self, base_llm, profiler):
        super().__init__(base_llm, profiler)
        # Add functionality specific to new LLM
```

### 3. Extending Visualization

```python
def create_custom_chart(self, data, save_path=None):
    # Implement custom chart
    fig, ax = plt.subplots()
    # ... plotting code
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

## 📜 License

This project follows the license agreement of the LLM4AD project. Please follow relevant citation requirements when using this system.

---

**Note**: This system is still under continuous development, and functionality and APIs may change. Please update regularly to get the latest features and bug fixes. 