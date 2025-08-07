# Evolution Reflection in EoHmeme

## Overview

The Evolution Reflection module in EoHmeme is inspired by ReEvo and implements a meta-level learning approach that analyzes the evolutionary process itself to guide future heuristic generation. This reflective mechanism helps to:

1. Identify patterns in successful and unsuccessful evolution steps
2. Compare best and worst performing solutions in the population
3. Detect stagnation periods in the search process
4. Analyze population diversity
5. Generate targeted guidance for different evolutionary operators
6. Adaptively adjust search strategies based on past performance

## How It Works

The reflection mechanism works in three main stages:

### 1. Analysis

During evolution, the `EvolutionReflector` periodically analyzes:

- **Population diversity**: Using semantic differences between algorithms
- **Performance gap**: Analyzing differences between best and worst performing algorithms
- **Algorithmic features**: Extracting patterns that correlate with high or low performance
- **Stagnation detection**: Identifying when evolution stops making progress

### 2. Reflection Generation

Based on the analysis, the reflector generates specific insights and recommendations:

- Identification of algorithmic features that appear in high-performing solutions
- Detection of problematic patterns in low-performing solutions
- Suggestions for increasing diversity when the population converges
- Strategies to break through stagnation periods
- Advice on whether to focus on exploitation or exploration

### 3. Guidance Integration

These reflections are then integrated into the prompts sent to the LLM:

- Each operator (e1, e2, m1, m2, local_search) receives custom guidance
- The LLM is given insights about successful and unsuccessful algorithm features
- Explicit reflection sessions can also generate new ideas directly from the LLM

## Using the Reflection Module

To enable reflection in your EoH run:

```python
method = EoH(
    # ... other parameters ...
    use_reflection=True,           # Enable reflection mechanism
    reflection_frequency=3,        # Perform reflection every 3 generations
)
```

## Reflection Frequency

The `reflection_frequency` parameter in the EoH class controls how often reflection is performed:

- A value of 1 means reflection occurs every generation
- A value of 2 means reflection occurs every other generation
- Higher values reduce the computational overhead of reflection

The EoH class handles the actual timing of reflections - it only calls the reflector's `reflect()` method when `current_generation % reflection_frequency == 0`. This ensures that reflections only happen at the specified intervals, allowing you to balance the benefits of reflection against its computational cost.

## Customizing Reflection

You can adjust how reflection works by modifying these parameters:

- `reflection_frequency`: How often to perform reflection (in generations)
- `use_reflection`: Enable/disable the reflection mechanism

## Integration with Evolution Memory

While the reflection mechanism primarily focuses on analyzing the current population, it can still work with Evolution Memory if available:

- Memory provides historical data about effective operators
- Reflection enhances this with direct analysis of algorithm features
- Together they create a more intelligent and adaptive search process

## Example Reflections

Here are examples of insights the reflection module might generate:

1. "The most effective algorithmic features include: prioritization, sorting, early termination. Consider emphasizing these elements in future algorithm generation."
2. "Features like multiple_conditions and dictionary_usage appear in low-performing algorithms. Consider avoiding or redesigning these aspects."
3. "The best performing algorithm uses prioritization strategies. Consider incorporating priority-based decision making in new algorithms."
4. "There is a substantial performance gap between best and worst algorithms. Focus on understanding what makes the top performers effective and incorporate those patterns."
5. "Evolution appears to be stagnating. Consider introducing new exploration strategies, or applying more aggressive local search."

## Architecture

The reflection module consists of:

- `EvolutionReflector`: Main class that analyzes best and worst performing algorithms
- `EoHPrompt.get_reflection_prompt()`: Generates explicit reflection prompts
- Integration points in EoH for triggering reflections and using their output

## Feature Extraction

The reflection module automatically extracts algorithmic features from both the text descriptions and code of algorithms:

- **Concept features**: Keywords like "priority", "greedy", "heuristic", etc.
- **Pattern features**: Decision-making approaches like conditional rules, prioritization, balancing
- **Code features**: Programming patterns like loops, conditions, data structures, optimization functions

These features are correlated with performance to determine which ones appear more frequently in successful algorithms versus unsuccessful ones.

## Using Reflection Output in Your Own Code

You can access reflection data programmatically through:

```python
# Get current insights
insights = method._reflector._current_insights

# Access specific reflection components
diversity_level = insights.get('diversity', {}).get('level')
perf_diff = insights.get('performance_difference', {}).get('significance')
effective_features = insights.get('pattern_insights', {}).get('effective_features', [])
ineffective_features = insights.get('pattern_insights', {}).get('ineffective_features', [])
reflections = insights.get('reflections', [])
```

This allows you to build your own custom mechanisms on top of the reflection system. 