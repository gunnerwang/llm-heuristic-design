from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np

from ...base import *


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._generation = generation

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    def register_function(self, func: Function):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.score is None:
            return
        # if the score is None, we still put it into the population,
        # we set the score to '-inf'
        if func.score is None:
            func.score = float('-inf')
        try:
            self._lock.acquire()
            if self.has_duplicate_function(func):
                func.score = float('-inf')
            # register to next_gen
            self._next_gen_pop.append(func)
            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size:
                pop = self._population + self._next_gen_pop
                pop = sorted(pop, key=lambda f: f.score, reverse=True)
                self._population = pop[:self._pop_size]
                self._next_gen_pop = []
                self._generation += 1
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if str(f) == str(func) or func.score == f.score:
                return True
        for f in self._next_gen_pop:
            if str(f) == str(func) or func.score == f.score:
                return True
        return False

    def selection(self) -> Function:
        funcs = [f for f in self._population if not math.isinf(f.score) and f.score is not None]
        
        # If no valid functions, try to include those with infinite scores
        if not funcs and self._population:
            print("Warning: No valid functions with finite scores found in population. Using all available functions.")
            funcs = self._population.copy()
        
        # If still no functions, raise a more descriptive error
        if not funcs:
            raise ValueError("Cannot perform selection on empty population. Ensure population is initialized properly.")
        
        # Sort by score (highest first)
        sorted_funcs = sorted(funcs, key=lambda f: f.score if f.score is not None else float('-inf'), reverse=True)
        
        # Calculate selection probabilities (higher scores = higher probability)
        p = [1 / (r + len(sorted_funcs)) for r in range(len(sorted_funcs))]
        p = np.array(p)
        p = p / np.sum(p)
        
        try:
            # Perform selection
            return np.random.choice(sorted_funcs, p=p)
        except Exception as e:
            # Fallback to deterministic selection if probabilistic fails
            print(f"Warning: Random selection failed ({str(e)}). Using best function instead.")
            return sorted_funcs[0]
