# Evolution Path Memory

## Overview

Evolution Path Memory is a feature that enhances the EOHMEME (Evolution of Heuristics) algorithm by:

1. Recording and analyzing successful evolution paths 
2. Extracting effective evolution patterns
3. Guiding future heuristic generation based on past successes

## How It Works

The Evolution Path Memory system keeps track of successful optimization steps, including:
- Which operators (e1, e2, m1, m2, local_search) tend to be most effective
- What types of algorithmic changes typically lead to improvements
- Successful examples of parent-child algorithm transformations

This information is used to enhance the prompts sent to the LLM, providing additional context and guidance for generating new algorithms.

## Benefits

- **Improved Efficiency**: Leverages patterns from past successes to guide future exploration
- **Better Adaptivity**: Dynamically adjusts to the characteristics of the problem being solved
- **Knowledge Accumulation**: Builds up knowledge across multiple runs of the optimization process
- **Reduced Exploration Cost**: Focuses search in promising areas of the solution space

## Implementation

The feature is implemented through several components:

1. **EvolutionPathMemory Class**: Core implementation that tracks and analyzes evolution paths
2. **Prompt Enhancement**: Updated prompt templates that incorporate insights from memory
3. **Success Tracking**: Recording of which evolution steps improved performance

## Usage

To enable Evolution Path Memory in your optimization process, set `use_evolution_memory=True` when initializing the EoH class:

```python
optimizer = EoH(
    llm=my_llm,
    evaluation=my_evaluation,
    use_evolution_memory=True,  # Enable evolution path memory
    memory_capacity=100,        # Set capacity (optional, default=100)
    # ... other parameters
)
```

The memory is automatically persisted to disk (in the log directory) and will be loaded in future runs if available.

## Key Files

- `evolution_memory.py`: Core implementation of the evolution path memory
- `prompt.py`: Updated prompt templates that incorporate memory guidance
- `eohmeme.py`: Integration with the main EoH implementation

## Future Enhancements

Potential future enhancements include:
- More sophisticated pattern extraction techniques
- Multi-objective memory guidance
- Cross-problem transfer learning
- Real-time adaptation of operator probabilities based on success rates 