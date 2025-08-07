# EOH Memetic: Evolutionary Optimization with Local Search

This module enhances the original EOH (Evolution of Heuristics) algorithm by incorporating concepts from Memetic Algorithms.

## Overview

Memetic Algorithms are hybrid optimization techniques that combine evolutionary algorithms with local search methods. This implementation extends EOH by adding a local search phase to refine promising solutions.

## Key Features

- All original EOH operators (E1, E2, M1, M2)
- New local search capability that periodically applies focused improvements to promising individuals
- Configurable frequency and intensity of local search operations

## How It Works

1. The algorithm performs regular EOH evolution using operators E1, E2, M1, and M2
2. At regular intervals (controlled by `memetic_frequency`), it selects a portion of the population (controlled by `memetic_intensity`) for local improvement
3. The selected individuals undergo focused local search to refine their solution quality
4. These improved individuals are then evaluated and compete for survival in the population

## Evolution Path Memory

EOHMEME now includes an Evolution Path Memory feature that:

- Records and analyzes successful evolution paths
- Extracts effective evolution patterns from past optimizations
- Guides future heuristic generation based on successful patterns

To enable this feature, set `use_evolution_memory=True` when initializing:

```python
optimizer = EoH(
    llm=my_llm,
    evaluation=my_evaluation,
    use_evolution_memory=True,
    # ... other parameters
)
```

For more details about this feature, see [evolution_memory_readme.md](./evolution_memory_readme.md).

## Usage

```python
from llm4ad.method.eohmeme import EoH

eoh = EoH(
    llm=your_llm,
    evaluation=your_evaluator,
    # Memetic algorithm parameters
    use_memetic=True,        # Enable memetic algorithm
    memetic_frequency=2,     # Apply local search every 2 generations
    memetic_intensity=0.3,   # Apply to top 30% of the population
    # Other standard EOH parameters
    max_generations=10,
    max_sample_nums=100,
    pop_size=5,
    # ...
)

best_function = eoh.run()
```

## Parameters for Memetic Algorithm

- `use_memetic` (bool): Enable or disable memetic algorithm features (default: True)
- `memetic_frequency` (int): How often to apply local search (every N generations) (default: 2)
- `memetic_intensity` (float): Percentage of population to undergo local search (default: 0.3)

## References

- Original EOH Paper: Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model." ICML 2024.
- Memetic Algorithms: Moscato, P. "On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts: Towards Memetic Algorithms". Caltech Concurrent Computation Program, C3P Report, 1989. 