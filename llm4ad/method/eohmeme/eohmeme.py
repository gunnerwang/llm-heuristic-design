# Module Name: EoH
# Last Revision: 2025/2/16
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang.
#       "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model."
#       In Forty-first International Conference on Machine Learning (ICML). 2024.
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import concurrent.futures
import time
import traceback
from threading import Thread
from typing import Optional, Literal
import math
import os
import datetime
import numpy as np
import random
from scipy import optimize

from .population import Population
from .profiler import EoHProfiler
from .prompt import EoHPrompt
from .sampler import EoHSampler
from .evolution_memory import EvolutionPathMemory
from .reflection import EvolutionReflector
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase


class EoH:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: Optional[int] = 10,
                 max_sample_nums: Optional[int] = 100,
                 pop_size: Optional[int] = 5,
                 selection_num=2,
                 use_e2_operator: bool = True,
                 use_m1_operator: bool = True,
                 use_m2_operator: bool = True,
                 use_memetic: bool = True,
                 memetic_frequency: int = 2,
                 memetic_intensity: float = 0.3,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 *,
                 resume_mode: bool = False,
                 initial_sample_nums_max: int = 50,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 use_evolution_memory: bool = False,
                 memory_capacity: int = 100,
                 use_hybrid_local_search: bool = True,
                 hybrid_local_search_method: str = 'cma-es',
                 use_reflection: bool = True,
                 reflection_frequency: int = 3,
                 reflector_llm: Optional[LLM] = None,
                 **kwargs):
        """Evolutionary of Heuristics.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            use_e2_operator : if use e2 operator.
            use_m1_operator : if use m1 operator.
            use_m2_operator : if use m2 operator.
            use_memetic     : if use memetic algorithm (local search).
            memetic_frequency: how often to apply local search (every N generations).
            memetic_intensity: percentage of the population to undergo local search.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            initial_sample_nums_max     : maximum samples restriction during initialization.
            use_evolution_memory        : if use evolution path memory to guide the generation process.
            memory_capacity             : maximum capacity of the evolution path memory.
            use_hybrid_local_search     : whether to use hybrid local search that combines LLM with classical optimization
            hybrid_local_search_method  : method for classical optimization ('cma-es', 'nelder-mead', 'powell')
            use_reflection              : whether to use reflective mechanisms for heuristic generation
            reflection_frequency        : how often to perform reflection (every N generations)
            reflector_llm               : an instance of 'llm4ad.base.LLM' for reflection analysis. If None, the main LLM will be used.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        self._use_e2_operator = use_e2_operator
        self._use_m1_operator = use_m1_operator
        self._use_m2_operator = use_m2_operator
        self._use_memetic = use_memetic
        self._memetic_frequency = memetic_frequency
        self._memetic_intensity = memetic_intensity
        self._use_evolution_memory = use_evolution_memory
        self._memory_capacity = memory_capacity
        self._use_hybrid_local_search = use_hybrid_local_search
        self._hybrid_local_search_method = hybrid_local_search_method
        self._use_reflection = use_reflection
        self._reflection_frequency = reflection_frequency

        # samplers and evaluators
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._initial_sample_nums_max = initial_sample_nums_max
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # adjust population size
        self._adjust_pop_size()

        # population, sampler, and evaluator
        self._population = Population(pop_size=self._pop_size)
        self._sampler = EoHSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler
        
        # evolution path memory
        if self._use_evolution_memory:
            # Determine the log directory for evolution memory
            if self._profiler and hasattr(self._profiler, '_log_dir'):
                memory_log_dir = self._profiler._log_dir
            else:
                # Create a dedicated log directory for evolution memory
                memory_log_dir = os.path.join(
                    os.getcwd(), 
                    'evolution_memory_logs', 
                    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                )
                os.makedirs(memory_log_dir, exist_ok=True)
                
            if self._debug_mode:
                print(f"Setting up evolution memory with log directory: {memory_log_dir}")
            
            try:
                self._evolution_memory = EvolutionPathMemory(
                    log_dir=memory_log_dir,
                    memory_capacity=self._memory_capacity
                )
                
                # Load memory from disk if available
                self._evolution_memory.load_memory()
                
                print(f"Evolution memory initialized with capacity: {self._memory_capacity}")
            except Exception as e:
                print(f"Error initializing evolution memory: {str(e)}")
                traceback.print_exc()
                self._evolution_memory = None
                self._use_evolution_memory = False
                print("Disabled evolution memory due to initialization error")
        else:
            self._evolution_memory = None
            
        # Initialize reflection mechanism
        if self._use_reflection:
            try:
                if self._profiler and hasattr(self._profiler, '_log_dir'):
                    reflection_log_dir = self._profiler._log_dir
                else:
                    reflection_log_dir = os.path.join(
                        os.getcwd(), 
                        'reflection_logs', 
                        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    )
                    os.makedirs(reflection_log_dir, exist_ok=True)
                
                # Initialize the reflector with a dedicated reflection LLM
                # If no specific reflector_llm is provided, we use the same LLM instance
                reflection_llm = reflector_llm if reflector_llm is not None else llm
                print(f"Initializing reflector with LLM: {type(reflection_llm).__name__}")
                
                # Create adapter for the LLM interface
                class LLMAdapter:
                    def __init__(self, llm: LLM):
                        self.llm = llm
                        print(f"Created LLM adapter with {type(llm).__name__}")
                        
                    def get_thought_and_function(self, prompt):
                        try:
                            print("LLMAdapter: Sending prompt to LLM")
                            # Use the draw_sample method from the LLM interface
                            response = self.llm.draw_sample(prompt)
                            print(f"LLMAdapter: Got response ({len(response) if response else 0} chars)")
                            # Return the response as the thought, with no function
                            return response, None
                        except Exception as e:
                            print(f"LLMAdapter error: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            return None, None
                
                # Create the adapter for the reflection LLM
                reflector = LLMAdapter(reflection_llm)
                
                self._reflector = EvolutionReflector(
                    memory=self._evolution_memory,
                    log_dir=reflection_log_dir,
                    llm_reflector=reflector    # LLM adapter for reflection
                )
                
                print(f"Reflection mechanism initialized with frequency: {self._reflection_frequency}")
            except Exception as e:
                print(f"Error initializing reflection mechanism: {str(e)}")
                traceback.print_exc()
                self._reflector = None
                self._use_reflection = False
                print("Disabled reflection due to initialization error")
        else:
            self._reflector = None

        # statistics
        self._tot_sample_nums = 0

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = max(
            self._initial_sample_nums_max,
            2 * self._pop_size
        )

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: necessary

    def _adjust_pop_size(self):
        # adjust population size
        if self._max_sample_nums >= 10000:
            if self._pop_size is None:
                self._pop_size = 40
            elif abs(self._pop_size - 40) > 20:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 40.')
        elif self._max_sample_nums >= 1000:
            if self._pop_size is None:
                self._pop_size = 20
            elif abs(self._pop_size - 20) > 10:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 20.')
        elif self._max_sample_nums >= 200:
            if self._pop_size is None:
                self._pop_size = 10
            elif abs(self._pop_size - 10) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 10.')
        else:
            if self._pop_size is None:
                self._pop_size = 5
            elif abs(self._pop_size - 5) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 5.')

    def _sample_evaluate_register(self, prompt, parent_funcs=None, operator_type=None):
        """Perform following steps:
        1. Sample an algorithm using the given prompt.
        2. Evaluate it by submitting to the process/thread pool, and get the results.
        3. Add the function to the population and register it to the profiler.
        4. Record evolution step in memory if evolution memory is enabled.
        """
        sample_start = time.time()
        try:
            thought, func = self._sampler.get_thought_and_function(prompt)
            sample_time = time.time() - sample_start
            
            if thought is None or func is None:
                print(f"Warning: Sampler returned None for thought or function with operator {operator_type}")
                return
            
            # Convert to Program instance
            program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
            if program is None:
                print(f"Warning: Failed to convert function to program with operator {operator_type}")
                return
            
            # Evaluate
            try:
                score, eval_time = self._evaluation_executor.submit(
                    self._evaluator.evaluate_program_record_time,
                    program
                ).result()
            except Exception as e:
                print(f"Error evaluating function: {str(e)}")
                return
            
            # Handle invalid scores (None, infinity, NaN)
            if score is None or math.isinf(score) or math.isnan(score):
                print(f"Warning: Function received {score} score, not registering")
                # Still count it in the sample total
                if self._profiler is not None:
                    self._tot_sample_nums += 1
                return
            
            # Register to profiler
            func.score = score
            func.evaluate_time = eval_time
            func.algorithm = thought
            func.sample_time = sample_time
            if self._profiler is not None:
                self._profiler.register_function(func)
                if isinstance(self._profiler, EoHProfiler):
                    self._profiler.register_population(self._population)
                self._tot_sample_nums += 1

            # Get the best score before registering the new function
            previous_best_score = max([f.score for f in self._population.population]) if self._population.population else float('-inf')
            
            # Determine if this is an improvement over parent functions
            parent_improvement = False
            if parent_funcs:
                # Check for valid parent scores (not None, not Infinity, not NaN)
                valid_parent_scores = [f.score for f in parent_funcs 
                                      if f.score is not None and not math.isinf(f.score) and not math.isnan(f.score)]
                
                # Only consider improvement if child has valid score and there are valid parent scores
                if valid_parent_scores and not math.isinf(score) and not math.isnan(score):
                    best_parent_score = max(valid_parent_scores)
                    # In this task, higher scores are better (even if negative)
                    parent_improvement = score > best_parent_score
                    if parent_improvement:
                        improvement = score - best_parent_score
                        print(f"New function improves over parent: {best_parent_score:.2f} -> {score:.2f} (improvement: {improvement:.2f})")
            
            # Register to the population
            self._population.register_function(func)
            
            # Determine if this is an improvement over the population's best
            valid_score = not math.isinf(score) and not math.isnan(score) and score is not None
            valid_previous = not math.isinf(previous_best_score) and not math.isnan(previous_best_score)
            
            # For this optimization task (higher is better, even with negative scores)
            population_improvement = valid_score and valid_previous and score > previous_best_score
            
            # Record evolution step in memory if evolution memory is enabled
            if self._evolution_memory and parent_funcs and operator_type and valid_score:
                try:
                    # Mark as successful if it improved over parents
                    success = parent_improvement
                    
                    # Record the evolution step (filtering will happen in evolution_memory)
                    self._evolution_memory.record_evolution_step(parent_funcs, func, operator_type, success)
                except Exception as e:
                    print(f"Error recording evolution step in memory: {str(e)}")
                    traceback.print_exc()
                    # Continue despite memory recording errors
            
            # Log success
            msg = f"Generated and evaluated function with {operator_type} operator, score: {score}"
            if population_improvement:
                msg += f" (new best, previous: {previous_best_score})"
            print(msg)
            
        except Exception as e:
            print(f"Error in _sample_evaluate_register: {str(e)}")
            traceback.print_exc()
            # Still count as a sample attempt
            if self._profiler is not None:
                self._tot_sample_nums += 1

    def _continue_loop(self) -> bool:
        if self._max_generations is None and self._max_sample_nums is None:
            return True
        elif self._max_generations is not None and self._max_sample_nums is None:
            return self._population.generation < self._max_generations
        elif self._max_generations is None and self._max_sample_nums is not None:
            return self._tot_sample_nums < self._max_sample_nums
        else:
            return (self._population.generation < self._max_generations
                    and self._tot_sample_nums < self._max_sample_nums)

    def _iteratively_use_eoh_operator(self):
        last_save_time = time.time()
        save_interval = 300  # Save every 5 minutes
        
        while self._continue_loop():
            try:
                current_time = time.time()
                
                # Periodically save memory if it's enabled
                if (self._evolution_memory and 
                    self._evolution_memory._log_dir and 
                    current_time - last_save_time > save_interval):
                    print(f"Performing periodic memory save after {current_time - last_save_time:.1f} seconds")
                    self._evolution_memory._save_memory()
                    last_save_time = current_time
                
                # Run reflection analysis if enabled
                if (self._use_reflection and 
                    self._reflector and 
                    self._population.generation > 0 and 
                    self._population.generation % self._reflection_frequency == 0):
                    try:
                        print(f"Performing reflection at generation {self._population.generation}")
                        insights = self._reflector.reflect(self._population.population, self._population.generation)
                        if insights and "reflections" in insights and insights["reflections"]:
                            print(f"Reflection insights:")
                            for i, reflection in enumerate(insights["reflections"], 1):
                                print(f"  {i}. {reflection}")
                            
                            # LLM reflections are now handled within the reflector
                            if "llm_reflections" in insights and insights["llm_reflections"]:
                                print(f"LLM reflection insights:")
                                for i, reflection in enumerate(insights["llm_reflections"], 1):
                                    print(f"  {i}. {reflection}")
                    except Exception as e:
                        print(f"Error during reflection: {str(e)}")
                        traceback.print_exc()
                
                # get a new func using e1
                try:
                    # Try to select individuals, with error handling
                    individuals = []
                    try:
                        for _ in range(self._selection_num):
                            individuals.append(self._population.selection())
                    except Exception as e:
                        print(f"Error during selection for E1: {str(e)}")
                        # If we couldn't select enough individuals, use what we have or continue
                        if not individuals:
                            print("Couldn't select any individuals for E1, skipping")
                            continue
                        else:
                            print(f"Using {len(individuals)} individuals for E1 instead of {self._selection_num}")
                    
                    prompt = EoHPrompt.get_prompt_e1(
                        self._task_description_str, 
                        individuals, 
                        self._function_to_evolve, 
                        self._evolution_memory,
                        self._reflector
                    )
                    if self._debug_mode:
                        print(f'E1 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt, individuals, "e1")
                    if not self._continue_loop():
                        break
                except Exception as e:
                    print(f"Error in E1 operator: {str(e)}")
                    traceback.print_exc()
                    # Continue with next operator instead of breaking

                # get a new func using e2
                if self._use_e2_operator:
                    try:
                        indivs = [self._population.selection() for _ in range(self._selection_num)]
                        prompt = EoHPrompt.get_prompt_e2(
                            self._task_description_str, 
                            indivs, 
                            self._function_to_evolve, 
                            self._evolution_memory,
                            self._reflector
                        )
                        if self._debug_mode:
                            print(f'E2 Prompt: {prompt}')
                        self._sample_evaluate_register(prompt, indivs, "e2")
                        if not self._continue_loop():
                            break
                    except Exception as e:
                        print(f"Error in E2 operator: {str(e)}")
                        traceback.print_exc()
                        # Continue with next operator instead of breaking

                # get a new func using m1
                if self._use_m1_operator:
                    try:
                        indiv = self._population.selection()
                        prompt = EoHPrompt.get_prompt_m1(
                            self._task_description_str, 
                            indiv, 
                            self._function_to_evolve, 
                            self._evolution_memory,
                            self._reflector
                        )
                        if self._debug_mode:
                            print(f'M1 Prompt: {prompt}')
                        self._sample_evaluate_register(prompt, [indiv], "m1")
                        if not self._continue_loop():
                            break
                    except Exception as e:
                        print(f"Error in M1 operator: {str(e)}")
                        traceback.print_exc()
                        # Continue with next operator instead of breaking

                # get a new func using m2
                if self._use_m2_operator:
                    try:
                        indiv = self._population.selection()
                        prompt = EoHPrompt.get_prompt_m2(
                            self._task_description_str, 
                            indiv, 
                            self._function_to_evolve, 
                            self._evolution_memory,
                            self._reflector
                        )
                        if self._debug_mode:
                            print(f'M2 Prompt: {prompt}')
                        self._sample_evaluate_register(prompt, [indiv], "m2")
                        if not self._continue_loop():
                            break
                    except Exception as e:
                        print(f"Error in M2 operator: {str(e)}")
                        traceback.print_exc()
                        # Continue with next operator instead of breaking

                # Apply local search (memetic algorithm) if enabled and it's time
                if (self._use_memetic and
                        self._population.generation > 0 and
                        self._population.generation % self._memetic_frequency == 0):
                    try:
                        self._apply_local_search()
                    except Exception as e:
                        print(f"Error in local search: {str(e)}")
                        traceback.print_exc()
                        # Continue with next iteration instead of breaking

            except KeyboardInterrupt:
                print("Evolution interrupted by user")
                break
            except Exception as e:
                print(f"Critical error in evolution loop: {str(e)}")
                traceback.print_exc()
                # Only break for critical errors in the main loop
                # Add a small delay to avoid tight error loops
                time.sleep(1)

    def _iteratively_init_population(self):
        """Initialize the population with random samples."""
        attempts = 0
        max_attempts = max(self._initial_sample_nums_max * 2, 100)  # Set a reasonable limit
        
        while len(self._population) < self._pop_size and self._tot_sample_nums < self._initial_sample_nums_max and attempts < max_attempts:
            attempts += 1
            try:
                # random sample
                prompt = EoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve, self._evolution_memory)
                if self._debug_mode:
                    print(f'I1 Prompt: {prompt}')
                self._sample_evaluate_register(prompt, None, "i1")
                
                # Print progress update periodically
                if attempts % 5 == 0:
                    print(f"Population initialization: {len(self._population)}/{self._pop_size} individuals, {attempts} attempts")
                    
            except KeyboardInterrupt:
                print("Population initialization interrupted by user")
                break
            except Exception as e:
                print(f"Error during population initialization: {str(e)}")
                traceback.print_exc()
                # Add a small delay to avoid tight error loops
                time.sleep(0.5)
                
        if len(self._population) < self._pop_size:
            print(f"Warning: Could only initialize {len(self._population)}/{self._pop_size} individuals after {attempts} attempts")
        else:
            print(f"Successfully initialized population with {len(self._population)} individuals")

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """Multi-threaded sampling."""
        threads = []
        for _ in range(self._num_samplers):
            t = Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    def _apply_local_search(self):
        """Apply local search (memetic algorithm) to a subset of the population."""
        if not self._population.population:
            return
            
        # Sort population by fitness
        sorted_pop = sorted(
            self._population.population, 
            key=lambda f: f.score if f.score is not None else float('-inf'), 
            reverse=True
        )
        
        # Determine how many individuals to improve
        num_to_improve = max(1, round(len(sorted_pop) * self._memetic_intensity))
        individuals_to_improve = sorted_pop[:num_to_improve]
        
        # Apply local search to selected individuals
        for indiv in individuals_to_improve:
            if indiv.score is None or math.isinf(indiv.score):
                continue
            
            # Choose between LLM-only or hybrid local search
            if self._use_hybrid_local_search and random.random() > 0.5:  # 50% chance to use hybrid approach
                self._apply_hybrid_local_search(indiv)
            else:
                # Traditional LLM-based local search                
                # Generate local search prompt
                prompt = EoHPrompt.get_prompt_local_search(
                    self._task_description_str, 
                    indiv, 
                    self._function_to_evolve,
                    indiv.score,
                    self._evolution_memory,
                    self._reflector
                )
                
                if self._debug_mode:
                    print(f'LLM Local Search Prompt for individual with score {indiv.score}: {prompt}')
                    
                # Sample, evaluate, and register the improved individual
                self._sample_evaluate_register(prompt, [indiv], "local_search")
            
            # Check if we should continue
            if not self._continue_loop():
                break
                
    def _apply_hybrid_local_search(self, indiv: Function):
        """Apply hybrid local search using both LLM guidance and classical optimization algorithms.
        
        Args:
            indiv: The individual to improve
        """
        if self._debug_mode:
            print(f"Applying hybrid local search to individual with score {indiv.score} using {self._hybrid_local_search_method}")
        
        # Step 1: Use LLM to identify parameters or code sections that could be optimized
        parameter_extraction_prompt = EoHPrompt.get_prompt_parameter_extraction(
            self._task_description_str,
            indiv,
            self._function_to_evolve
        )
        
        thought, parameters_func = self._sampler.get_thought_and_function(parameter_extraction_prompt)
        if thought is None or parameters_func is None:
            print("Failed to extract parameters for hybrid local search")
            return
            
        # Step 2: Extract numerical parameters from the optimizable code
        try:
            # Convert parameters into a numerical vector for optimization
            # This is a simplified approach - in practice, we would need more sophisticated parameter extraction
            param_values = self._extract_numerical_parameters(parameters_func)
            
            if not param_values or len(param_values) == 0:
                print("No numerical parameters found for optimization")
                # Fall back to standard LLM-based local search
                prompt = EoHPrompt.get_prompt_local_search(
                    self._task_description_str, indiv, self._function_to_evolve, indiv.score, self._evolution_memory
                )
                self._sample_evaluate_register(prompt, [indiv], "local_search")
                return
                
            # Step 3: Define the objective function for optimization
            def objective_function(params):
                # Create a new function with the updated parameters
                new_func = self._create_function_with_parameters(indiv, parameters_func, params)
                
                # Convert to Program instance
                program = TextFunctionProgramConverter.function_to_program(new_func, self._template_program)
                if program is None:
                    return float('inf')  # Return worst possible score if conversion fails
                
                # Evaluate
                try:
                    score, _ = self._evaluator.evaluate_program_record_time(program)
                    return -score  # Negative because optimization algorithms minimize
                except Exception as e:
                    print(f"Error in objective function: {str(e)}")
                    return float('inf')
            
            # Step 4: Apply classical optimization algorithm
            initial_params = np.array(param_values)
            best_params = None
            
            try:
                if self._hybrid_local_search_method == 'cma-es':
                    # CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
                    import cma
                    es = cma.CMAEvolutionStrategy(initial_params, 0.5)
                    es.optimize(objective_function, iterations=10)
                    best_params = es.result.xbest
                    
                elif self._hybrid_local_search_method == 'nelder-mead':
                    # Nelder-Mead simplex algorithm
                    result = optimize.minimize(
                        objective_function, 
                        initial_params, 
                        method='Nelder-Mead',
                        options={'maxiter': 20, 'disp': self._debug_mode}
                    )
                    best_params = result.x
                    
                elif self._hybrid_local_search_method == 'powell':
                    # Powell's method
                    result = optimize.minimize(
                        objective_function, 
                        initial_params, 
                        method='Powell',
                        options={'maxiter': 20, 'disp': self._debug_mode}
                    )
                    best_params = result.x
                else:
                    # Random search as fallback
                    best_score = float('inf')
                    best_params = initial_params
                    
                    for _ in range(10):
                        # Random perturbation
                        params = initial_params + np.random.normal(0, 0.2, size=len(initial_params))
                        score = objective_function(params)
                        
                        if score < best_score:
                            best_score = score
                            best_params = params
            
            except Exception as e:
                print(f"Error during classical optimization: {str(e)}")
                traceback.print_exc()
                # Fall back to standard LLM-based local search
                prompt = EoHPrompt.get_prompt_local_search(
                    self._task_description_str, indiv, self._function_to_evolve, indiv.score, self._evolution_memory
                )
                self._sample_evaluate_register(prompt, [indiv], "local_search")
                return
            
            # Step 5: Create and evaluate the optimized function
            if best_params is not None:
                optimized_func = self._create_function_with_parameters(indiv, parameters_func, best_params)
                
                # Add metadata
                optimized_func.algorithm = f"{indiv.algorithm} (hybrid optimized)"
                
                # Evaluate and register
                program = TextFunctionProgramConverter.function_to_program(optimized_func, self._template_program)
                if program:
                    try:
                        score, eval_time = self._evaluator.evaluate_program_record_time(program)
                        
                        # Register only if better than original
                        if score > indiv.score:
                            optimized_func.score = score
                            optimized_func.evaluate_time = eval_time
                            optimized_func.sample_time = 0  # Not sampled from LLM
                            
                            if self._profiler is not None:
                                self._profiler.register_function(optimized_func)
                                self._tot_sample_nums += 1
                                
                            # Get previous best score before registering the new function
                            previous_best_score = max([f.score for f in self._population.population]) if self._population.population else float('-inf')
                            
                            # Register to the population
                            self._population.register_function(optimized_func)
                            
                            # Record in evolution memory - ALWAYS mark as successful if it improved the individual 
                            if self._evolution_memory:
                                success = True  # This is definitely a successful improvement over the parent
                                try:
                                    self._evolution_memory.record_evolution_step(
                                        [indiv], optimized_func, "hybrid_local_search", success
                                    )
                                    print(f"Recorded hybrid local search evolution step in memory: {indiv.score} -> {score}")
                                except Exception as e:
                                    print(f"Error recording evolution step: {str(e)}")
                                    traceback.print_exc()
                                
                            if self._debug_mode:
                                print(f"Hybrid local search improved score from {indiv.score} to {score}")
                        else:
                            if self._debug_mode:
                                print(f"Hybrid local search did not improve score: {indiv.score} -> {score}")
                    except Exception as e:
                        print(f"Error evaluating optimized function: {str(e)}")
            
        except Exception as e:
            print(f"Error in hybrid local search: {str(e)}")
            traceback.print_exc()
            # Fall back to standard LLM-based local search as a backup
            prompt = EoHPrompt.get_prompt_local_search(
                self._task_description_str, indiv, self._function_to_evolve, indiv.score, self._evolution_memory
            )
            self._sample_evaluate_register(prompt, [indiv], "local_search")
    
    def _extract_numerical_parameters(self, func: Function) -> list:
        """Extract numerical parameters from a function.
        
        Args:
            func: The function to extract parameters from
            
        Returns:
            List of numerical parameter values
        """
        # This is a simplified implementation - in practice, would need more sophisticated code analysis
        try:
            import ast
            
            # Parse the function body to find numerical constants
            tree = ast.parse(func.body)
            
            # Collect all numerical constants
            parameters = []
            
            class NumConstVisitor(ast.NodeVisitor):
                def visit_Constant(self, node):
                    if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                        parameters.append(node.value)
                    self.generic_visit(node)
                    
                # For Python 3.7 and earlier which used Num instead of Constant
                def visit_Num(self, node):
                    parameters.append(node.n)
                    self.generic_visit(node)
            
            NumConstVisitor().visit(tree)
            return parameters
            
        except Exception as e:
            print(f"Error extracting parameters: {str(e)}")
            return []
            
    def _create_function_with_parameters(self, original_func: Function, params_func: Function, params: list) -> Function:
        """Create a new function by replacing numerical parameters in the original function.
        
        Args:
            original_func: The original function
            params_func: Function with identified parameters
            params: New parameter values
            
        Returns:
            A new function with updated parameters
        """
        # This is a simplified implementation - in practice, would need more sophisticated code modification
        try:
            import ast
            import astor
            
            # Parse the function body
            tree = ast.parse(original_func.body)
            
            # Replace numerical constants
            param_index = 0
            
            class ReplaceNumConstants(ast.NodeTransformer):
                def visit_Constant(self, node):
                    nonlocal param_index
                    if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                        if param_index < len(params):
                            node.value = float(params[param_index])
                            param_index += 1
                    return node
                    
                # For Python 3.7 and earlier which used Num instead of Constant
                def visit_Num(self, node):
                    nonlocal param_index
                    if param_index < len(params):
                        return ast.Constant(value=float(params[param_index]), kind=None)
                    return node
            
            # Apply the transformation
            new_tree = ReplaceNumConstants().visit(tree)
            
            # Convert back to source code
            new_body = astor.to_source(new_tree)
            
            # Create a new function
            new_func = Function(
                name=original_func.name,
                args=original_func.args,
                returns=original_func.returns,
                body=new_body,
                docstring=original_func.docstring
            )
            
            # Copy metadata
            new_func.algorithm = original_func.algorithm
            
            return new_func
            
        except Exception as e:
            print(f"Error creating function with new parameters: {str(e)}")
            return original_func

    def run(self):
        """Run the evolution until termination conditions are met."""
        start_time = time.time()
        last_progress_log = time.time()
        progress_interval = 60  # Log progress every minute
        
        print(f"Starting evolution with parameters:")
        print(f"- Population size: {self._pop_size}")
        print(f"- Max generations: {self._max_generations}")
        print(f"- Max sample numbers: {self._max_sample_nums}")
        print(f"- Evolution memory enabled: {self._use_evolution_memory}")
        print(f"- Reflection enabled: {self._use_reflection}")
        
        if not self._resume_mode:
            print("Initializing population...")
            self._iteratively_init_population()
        else:
            print("Resuming from existing population")

        print("Starting main evolution loop...")
        
        try:
            self._iteratively_use_eoh_operator()
        except Exception as e:
            print(f"Critical error in main evolution loop: {str(e)}")
            traceback.print_exc()
        
        # Calculate statistics
        total_time = time.time() - start_time
        generations_completed = self._population.generation
        samples_evaluated = self._tot_sample_nums
        
        print(f"\nEvolution completed:")
        print(f"- Total runtime: {total_time:.1f} seconds")
        print(f"- Generations completed: {generations_completed}")
        print(f"- Samples evaluated: {samples_evaluated}")
        
        # Get the final best algorithm
        if self._population.population:
            valid_funcs = [f for f in self._population.population if f.score is not None and not math.isinf(f.score)]
            if valid_funcs:
                best_func = max(valid_funcs, key=lambda f: f.score)
                print(f'Best score: {best_func.score}')
                print(f'Best algorithm: {best_func.algorithm}')
                print(f'Best function: {best_func}')
            else:
                print("No valid functions with non-infinite scores found in final population")
        else:
            print("Warning: Empty population at end of evolution")
        
        # Save evolution memory to disk if enabled
        if self._evolution_memory:
            evolution_paths_count = len(self._evolution_memory._evolution_paths) if hasattr(self._evolution_memory, '_evolution_paths') else 0
            print(f'Evolution memory: {evolution_paths_count} paths recorded')
            
            try:
                successful_patterns = self._evolution_memory.get_most_successful_patterns()
                print(f'Most successful patterns: {successful_patterns}')
                
                operator_rates = self._evolution_memory.get_operator_success_rates()
                print(f'Operator success rates: {operator_rates}')
                
                # Print the top evolution paths for reference
                top_paths = self._evolution_memory.get_successful_evolution_paths(top_n=3)
                if top_paths:
                    print("\nTop improvement evolution paths:")
                    for i, path in enumerate(top_paths, 1):
                        parent_score = max(path.get('parent_scores', [0]))
                        child_score = path.get('child_score', 0)
                        operator = path.get('operator', 'unknown')
                        improvement = path.get('improvement', 0)
                        print(f"{i}. {operator}: {parent_score} -> {child_score} (improvement: {improvement:.2f})")
            except Exception as e:
                print(f"Error getting evolution memory statistics: {str(e)}")
                traceback.print_exc()
            
            # Force final save to ensure latest state is persisted
            if hasattr(self._evolution_memory, '_log_dir') and self._evolution_memory._log_dir:
                try:
                    print("Saving final evolution memory state...")
                    self._evolution_memory._save_memory()
                    print(f"Memory saved to {self._evolution_memory._memory_dir}")
                except Exception as e:
                    print(f"Error saving final memory state: {str(e)}")
                    traceback.print_exc()
                    
        # Print reflection statistics if enabled
        if self._use_reflection and self._reflector:
            try:
                print("\nReflection statistics:")
                reflection_count = len(self._reflector._reflections)
                print(f"- Total reflections performed: {reflection_count}")
                
                if reflection_count > 0:
                    # Print diversity history
                    diversity_history = self._reflector._diversity_history
                    if diversity_history:
                        avg_diversity = sum(diversity_history) / len(diversity_history)
                        print(f"- Average population diversity: {avg_diversity:.2f}")
                        print(f"- Diversity trend: {diversity_history[-3:] if len(diversity_history) >= 3 else diversity_history}")
                    
                    # Print stagnation periods
                    stagnation_periods = self._reflector._stagnation_periods
                    if stagnation_periods:
                        print(f"- Stagnation periods: {len(stagnation_periods)}")
                        total_stagnation = sum(period.get('duration', 0) for period in stagnation_periods)
                        print(f"- Total generations in stagnation: {total_stagnation}")
                    
                    # Print most recent reflections
                    recent_reflections = self._reflector._reflections[-1].get('reflections', []) if self._reflector._reflections else []
                    if recent_reflections:
                        print("\nFinal reflections:")
                        for i, reflection in enumerate(recent_reflections, 1):
                            print(f"  {i}. {reflection}")
                
                # Example of how reflection impacted evolution
                print("\nReflection impact example:")
                if hasattr(self._reflector, '_current_insights') and self._reflector._current_insights:
                    insights = self._reflector._current_insights
                    if 'pattern_insights' in insights and 'top_operators' in insights['pattern_insights']:
                        top_ops = insights['pattern_insights']['top_operators']
                        if top_ops:
                            print(f"- Identified most effective operators: {', '.join(top_ops)}")
                            print(f"  This guided the algorithm to favor these operators in later generations.")
                    
                    if 'stagnation' in insights and insights['stagnation'].get('status') == 'stagnating':
                        print(f"- Detected stagnation after {insights['stagnation'].get('duration', 0)} reflections")
                        print(f"  Prompted exploration of novel approaches to break through performance plateaus.")
                else:
                    print("- No specific reflection insights recorded")
            except Exception as e:
                print(f"Error printing reflection statistics: {str(e)}")
                traceback.print_exc()

        return self._population.population
