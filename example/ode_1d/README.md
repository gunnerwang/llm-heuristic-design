# Comprehensive ODE Discovery Evaluation System

This directory contains a comprehensive evaluation system for ODE (Ordinary Differential Equation) discovery experiments using LLM-based methods like EoH (Evolution of Heuristics) and EoHMeMe (EoH with Memetic and Memory enhancements).

## 🎯 Features

- **Comprehensive Metrics Collection**: Detailed performance, cost, timing, and stability metrics
- **Token-Aware LLM Wrapper**: Track API usage, costs, and response times  
- **Enhanced Evaluation**: Multi-run stability analysis and solution diversity tracking
- **Baseline Comparisons**: Built-in simple and advanced ODE solvers for benchmarking
- **Comparative Analysis**: Compare multiple experiment configurations
- **Rich Reporting**: JSON exports and human-readable summaries
- **Visualization Support**: Performance trends and diversity analysis

## 📁 File Structure

```
ode_1d/
├── run_eoh.py                           # Original EoH runner
├── run_eohmeme.py                       # EoHMeMe runner with all enhancements
├── comprehensive_metrics.py             # Core metrics collection system
├── enhanced_llm_wrapper.py              # Token-aware LLM wrapper
├── enhanced_evaluation.py               # Enhanced ODE evaluation with stability analysis
├── run_comprehensive_ode_evaluation.py  # Main comprehensive experiment runner
├── test_comprehensive_system.py         # Test suite for the system
└── README_comprehensive_metrics.md      # This documentation
```

## 🚀 Quick Start

### 1. Basic EoHMeMe Run

```python
python run_eohmeme.py
```

### 2. Comprehensive Evaluation

```python
python run_comprehensive_ode_evaluation.py
```

### 3. Test System

```python
python test_comprehensive_system.py
```

## 🔧 System Components

### ComprehensiveProfiler

Enhanced profiler that extends the base LLM4AD profiler with:

- **Performance Metrics**: Best/worst/mean scores, convergence tracking
- **Token Metrics**: Usage, costs, API calls with detailed breakdown
- **Timing Metrics**: Evaluation times, LLM response times
- **Stability Metrics**: Success rate, performance variance, robustness scores

```python
from comprehensive_metrics import ComprehensiveProfiler

profiler = ComprehensiveProfiler(log_dir="logs", track_tokens=True)
profiler.log_evaluation(score=0.85, evaluation_time=1.2)
profiler.log_llm_call(tokens_used=100, cost=0.01, call_time=0.5)
profiler.calculate_final_metrics()
```

### Enhanced LLM Wrapper

Token-aware wrapper around LLM APIs with cost tracking:

```python
from enhanced_llm_wrapper import TokenAwareLLMFactory

enhanced_llm = TokenAwareLLMFactory.create_enhanced_llm(
    host='api.deepseek.com',
    key='your-api-key',
    model='deepseek-chat',
    profiler=profiler
)
```

**Supported Models & Pricing**:
- `deepseek-chat`: $0.0014/1K tokens
- `gpt-3.5-turbo`: $0.002/1K tokens  
- `gpt-4`: $0.03/1K tokens
- `claude-3-sonnet`: $0.015/1K tokens

### Enhanced ODE Evaluation

Extended evaluation with stability analysis:

```python
from enhanced_evaluation import EnhancedODEEvaluation

enhanced_task = EnhancedODEEvaluation(
    profiler=profiler,
    stability_runs=3,  # Run each solution 3 times
    collect_detailed_metrics=True
)

# Evaluate with stability analysis
result = enhanced_task.evaluate_with_stability_analysis(
    your_ode_solver,
    function_id="my_solver"
)
```

## 📊 Metrics Collected

### Performance Metrics
- **Best/Worst/Mean Scores**: Statistical summary of all evaluations
- **Convergence Generation**: When the best solution was found
- **Success Rate**: Percentage of successful evaluations
- **Solution Diversity**: Variety in explored solutions

### Resource Efficiency
- **Token Usage**: Total, prompt, and completion tokens
- **API Costs**: Total cost in USD with breakdown
- **Response Times**: Average LLM API response times
- **Evaluation Times**: Time spent on fitness evaluations

### Stability Analysis
- **Multi-run Consistency**: Performance across multiple runs
- **Confidence Intervals**: Statistical confidence in results
- **Robustness Score**: Combined stability metric (0-1)
- **Coefficient of Variation**: Relative variability measure

## 🏃‍♂️ Running Experiments

### Configuration Options

The system supports multiple EoH configurations:

1. **EoH Baseline**: Standard EoH without enhancements
2. **EoH Memetic**: EoH with memetic local search
3. **EoH Full Enhanced (EoHMeMe)**: All enhancements enabled

### Experiment Parameters

```python
method_config = {
    'max_sample_nums': 20,           # Total samples per generation
    'max_generations': 5,            # Number of generations
    'pop_size': 2,                   # Population size
    'num_samplers': 1,               # Parallel samplers
    'num_evaluators': 1,             # Parallel evaluators
    'use_memetic': True,             # Enable memetic enhancement
    'memetic_frequency': 1,          # How often to apply memetic
    'memetic_intensity': 0.5,        # Intensity of memetic search
    'use_hybrid_local_search': True, # Enable hybrid local search
    'hybrid_local_search_method': 'cma-es',  # Local search method
    'use_evolution_memory': True,    # Enable evolution memory
    'memory_capacity': 10,           # Memory size
    'use_reflection': True,          # Enable reflection mechanism
    'reflection_frequency': 1,       # Reflection frequency
    'debug_mode': False              # Debug output
}
```

### Running Comparative Experiments

```python
# Automatically runs multiple configurations and compares them
python run_comprehensive_ode_evaluation.py
```

This will:
1. Test baseline ODE solvers
2. Run EoH variants with different configurations
3. Collect comprehensive metrics for each
4. Generate comparative analysis
5. Export detailed reports

## 📈 Output and Reports

### Generated Files

```
comprehensive_logs/
├── experiment_name/
│   ├── run_0/
│   │   ├── comprehensive_metrics.json    # Detailed metrics
│   │   ├── evaluation_report.json        # Evaluation analysis  
│   │   └── llm_usage.json               # LLM usage logs
│   ├── run_1/ ...
│   ├── experiment_summary.json          # Experiment summary
│   └── comparison_report.json           # Run comparison
├── final_comparison_report.json         # Cross-experiment comparison
└── final_comprehensive_results.json     # Complete results
```

### Report Contents

**Comprehensive Metrics (`comprehensive_metrics.json`)**:
```json
{
  "performance_metrics": {
    "best_score": 0.85,
    "mean_score": 0.72,
    "std_score": 0.08,
    "total_evaluations": 45,
    "success_rate": 0.95
  },
  "token_metrics": {
    "total_tokens": 15420,
    "total_cost_usd": 0.0216,
    "api_calls": 23
  },
  "stability_metrics": {
    "robustness_score": 0.87,
    "performance_variance": 0.0064
  }
}
```

**Evaluation Report (`evaluation_report.json`)**:
```json
{
  "performance_summary": {
    "best_overall_score": 0.85,
    "total_solutions_evaluated": 12
  },
  "diversity_analysis": {
    "diversity_score": 0.34,
    "unique_solutions": 8
  },
  "trend_analysis": {
    "trend_direction": "improving",
    "improvement_rate": 0.05
  }
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_comprehensive_system.py
```

Tests include:
- ✅ Basic component functionality
- ✅ Baseline solver operations  
- ✅ System integration
- ✅ Experiment comparator
- ✅ File operations
- ✅ Mini experiment pipeline

## 🔬 Baseline Solvers

The system includes baseline ODE solvers for comparison:

### Simple Polynomial Solver
```python
def simple_polynomial_solver(x, y):
    # Returns: dy/dx = 0.1*y + 0.05
    return 0.1 * y + 0.05
```

### Advanced Polynomial Solver  
```python
def adaptive_polynomial_solver(x, y):
    # Fits: dy/dx = ax² + bx*y + cy + d
    # Uses least squares for coefficient fitting
```

## 📝 Usage Examples

### Single Experiment Run

```python
from run_comprehensive_ode_evaluation import run_comprehensive_experiment

llm_config = {
    'host': 'api.deepseek.com',
    'key': 'your-api-key',
    'model': 'deepseek-chat'
}

method_config = {
    'max_sample_nums': 20,
    'max_generations': 5,
    'use_memetic': True,
    'use_reflection': True
}

results = run_comprehensive_experiment(
    experiment_name="my_ode_experiment",
    llm_config=llm_config,
    method_config=method_config,
    num_runs=3
)
```

### Custom Evaluation with Stability Analysis

```python
from enhanced_evaluation import EnhancedODEEvaluation
from comprehensive_metrics import ComprehensiveProfiler

profiler = ComprehensiveProfiler(log_dir="custom_logs")
task = EnhancedODEEvaluation(profiler=profiler, stability_runs=5)

def my_ode_solver(x, y):
    # Your ODE solver implementation
    return some_derivative_approximation

# Evaluate with stability analysis
result = task.evaluate_with_stability_analysis(
    my_ode_solver,
    function_id="custom_solver"
)

print(f"Best score: {result['summary']['best_score']}")
print(f"Stability: {result['stability_metrics']['stability_score']}")
```

## 🎛️ Configuration

### Environment Variables
- Set `DEEPSEEK_API_KEY` or modify the key in configuration files
- Adjust `timeout` values based on your network conditions

### Customization Points
- **Baseline Solvers**: Add custom baseline implementations
- **Metrics**: Extend `ComprehensiveProfiler` for custom metrics
- **Evaluation**: Override `EnhancedODEEvaluation` methods
- **Reporting**: Customize report generation and formats

## 🚨 Important Notes

1. **API Keys**: Replace dummy API keys with real ones before running
2. **Costs**: Monitor token usage and costs, especially with larger experiments  
3. **Storage**: Comprehensive logs can be large - clean up periodically
4. **Dependencies**: Requires `numpy`, `scipy`, and LLM4AD framework
5. **Reproducibility**: Results may vary due to LLM non-determinism

## 🤝 Contributing

To extend the system:

1. **Add New Metrics**: Extend `ComprehensiveProfiler`
2. **New LLM Providers**: Add to `TokenAwareLLMFactory`
3. **Enhanced Analysis**: Extend `EnhancedODEEvaluation`
4. **Visualization**: Add plotting capabilities
5. **Tests**: Add tests to `test_comprehensive_system.py`

## 📚 References

- **LLM4AD Framework**: Base framework for LLM-assisted algorithm design
- **EoH Method**: Evolution of Heuristics optimization approach
- **EoHMeMe**: EoH with Memetic and Memory enhancements

---

*This comprehensive evaluation system provides detailed insights into ODE discovery performance, costs, and reliability - enabling informed decisions about algorithm configurations and LLM usage strategies.* 