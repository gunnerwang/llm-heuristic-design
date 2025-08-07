from __future__ import annotations

import copy
from typing import List, Dict, Optional

from ...base import *
from .evolution_memory import EvolutionPathMemory


class EoHPrompt:
    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': cls.get_system_prompt()},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function, memory: Optional[EvolutionPathMemory] = None, reflector=None):
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        
        # Add reflection guidance if available
        if reflector is not None:
            reflection_guidance = reflector.get_reflection_prompt(task_prompt, "i1")
            if reflection_guidance.strip():
                # Here, the reflection already contains the task prompt and explicit guidance
                prompt_content = f'''{reflection_guidance}

Based on the guidance above, please:
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}. 
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
                return prompt_content
        
        # Add memory guidance if available
        if memory is not None:
            memory_guidance = memory.get_guidance_prompt(task_prompt, "i1")
            if memory_guidance.strip():
                return memory_guidance
        
        # create prompt content
        prompt_content = f'''{task_prompt}
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}. 
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function, memory: Optional[EvolutionPathMemory] = None, reflector=None):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        
        # Add reflection guidance if available
        if reflector is not None:
            reflection_guidance = reflector.get_reflection_prompt(task_prompt, "e1")
            if reflection_guidance.strip():
                # create prompt content with reflection - the reflection already contains task prompt
                prompt_content = f'''{reflection_guidance}

EXISTING ALGORITHMS TO CONSIDER:
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}

Using the evolution guidance above, create a new algorithm that has a totally different form from the given ones.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
                return prompt_content
        
        # Add memory guidance if available
        if memory is not None:
            try:
                memory_guidance = memory.get_guidance_prompt(task_prompt, "e1")
                # Only use memory guidance if it's not empty
                if memory_guidance.strip():
                    # create prompt content
                    prompt_content = f'''{memory_guidance}

I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones. 
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
                    return prompt_content
            except Exception as e:
                print(f"Error generating prompt with memory guidance: {str(e)}")
                # Fall back to non-memory prompt
        
        # create prompt content without memory guidance
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones. 
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function, memory: Optional[EvolutionPathMemory] = None, reflector=None):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        
        # Add reflection guidance if available
        if reflector is not None:
            reflection_guidance = reflector.get_reflection_prompt(task_prompt, "e2")
            if reflection_guidance.strip():
                # create prompt content with reflection
                prompt_content = f'''{reflection_guidance}

EXISTING ALGORITHMS TO CONSIDER:
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}

Following the evolution guidance above, create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided algorithms. 
2. Secondly, based on the backbone idea describe your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
                return prompt_content
        
        # Add memory guidance if available
        if memory is not None:
            try:
                memory_guidance = memory.get_guidance_prompt(task_prompt, "e2")
                # Only use memory guidance if it's not empty
                if memory_guidance.strip():
                    # create prompt content
                    prompt_content = f'''{memory_guidance}

I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided algorithms. 
2. Secondly, based on the backbone idea describe your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
                    return prompt_content
            except Exception as e:
                print(f"Error generating E2 prompt with memory guidance: {str(e)}")
                # Fall back to non-memory prompt
        
        # create prompt content without memory guidance
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided algorithms. 
2. Secondly, based on the backbone idea describe your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function, memory: Optional[EvolutionPathMemory] = None, reflector=None):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''

        # Add reflection guidance if available
        if reflector is not None:
            reflection_guidance = reflector.get_reflection_prompt(task_prompt, "m1")
            if reflection_guidance.strip():
                prompt_content = f'''{reflection_guidance}

ALGORITHM TO MODIFY:
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}

Using the evolution guidance above, create a new algorithm that has a different form but can be a modified version of the algorithm provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
                return prompt_content

        # Add memory guidance if available
        if memory is not None:
            try:
                memory_guidance = memory.get_guidance_prompt(task_prompt, "m1")
                # Only use memory guidance if it's not empty
                if memory_guidance.strip():
                    # create prompt content
                    prompt_content = f'''{memory_guidance}

I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
                    return prompt_content
            except Exception as e:
                print(f"Error generating M1 prompt with memory guidance: {str(e)}")
                # Fall back to non-memory prompt

        # create prompt content without memory guidance
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function, memory: Optional[EvolutionPathMemory] = None, reflector=None):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        
        # Add reflection guidance if available
        if reflector is not None:
            reflection_guidance = reflector.get_reflection_prompt(task_prompt, "m2")
            if reflection_guidance.strip():
                prompt_content = f'''{reflection_guidance}

ALGORITHM PARAMETERS TO MODIFY:
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}

Following the evolution guidance above, identify the main algorithm parameters and create a new algorithm that has different parameter settings of the algorithm provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
                return prompt_content
        
        # Add memory guidance if available
        if memory is not None:
            try:
                memory_guidance = memory.get_guidance_prompt(task_prompt, "m2")
                # Only use memory guidance if it's not empty
                if memory_guidance.strip():
                    # create prompt content
                    prompt_content = f'''{memory_guidance}

I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
                    return prompt_content
            except Exception as e:
                print(f"Error generating M2 prompt with memory guidance: {str(e)}")
                # Fall back to non-memory prompt
        
        # create prompt content without memory guidance
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_local_search(cls, task_prompt: str, indi: Function, template_function: Function, score: float, memory: Optional[EvolutionPathMemory] = None, reflector=None):
        """Memetic algorithm local search prompt.
        This prompt guides the LLM to perform a focused improvement (local search) on a specific individual,
        providing its current score to help the LLM better understand what needs to be improved.
        """
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        
        # Add reflection guidance if available
        if reflector is not None:
            reflection_guidance = reflector.get_reflection_prompt(task_prompt, "local_search")
            if reflection_guidance.strip():
                prompt_content = f'''{reflection_guidance}

ALGORITHM TO OPTIMIZE:
I have one algorithm with its code and current performance score ({score}) as follows:
Algorithm description:
{indi.algorithm}
Code:
{str(indi)}

Using the evolution guidance above, perform a focused local improvement on this algorithm without changing its core structure. Optimize specific parts 
that could improve performance while maintaining the algorithm's fundamental approach.
1. First, identify specific bottlenecks or areas for optimization in the current implementation.
2. Second, describe your improved algorithm in one sentence. The description must be inside within boxed {{}}.
3. Third, implement the following Python function with your optimizations:
{str(temp_func)}
Do not give additional explanations.'''
                return prompt_content
        
        # Add memory guidance if available
        if memory is not None:
            try:
                memory_guidance = memory.get_guidance_prompt(task_prompt, "local_search")
                # Only use memory guidance if it's not empty
                if memory_guidance.strip():
                    # create prompt content
                    prompt_content = f'''{memory_guidance}

I have one algorithm with its code and current performance score ({score}) as follows:
Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please perform a focused local improvement on this algorithm without changing its core structure. Optimize specific parts 
that could improve performance while maintaining the algorithm's fundamental approach.
1. First, identify specific bottlenecks or areas for optimization in the current implementation.
2. Second, describe your improved algorithm in one sentence. The description must be inside within boxed {{}}.
3. Third, implement the following Python function with your optimizations:
{str(temp_func)}
Do not give additional explanations.'''
                    return prompt_content
            except Exception as e:
                print(f"Error generating local search prompt with memory guidance: {str(e)}")
                # Fall back to non-memory prompt
        
        # create prompt content without memory guidance
        prompt_content = f'''{task_prompt}
I have one algorithm with its code and current performance score ({score}) as follows:
Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please perform a focused local improvement on this algorithm without changing its core structure. Optimize specific parts 
that could improve performance while maintaining the algorithm's fundamental approach.
1. First, identify specific bottlenecks or areas for optimization in the current implementation.
2. Second, describe your improved algorithm in one sentence. The description must be inside within boxed {{}}.
3. Third, implement the following Python function with your optimizations:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_parameter_extraction(cls, task_prompt: str, indi: Function, template_function: Function) -> str:
        """Generate a prompt for parameter extraction for hybrid local search.
        
        This prompt guides the LLM to identify and extract numerical parameters that could be optimized
        through classical optimization algorithms.
        
        Args:
            task_prompt: The original task description
            indi: The individual function to optimize
            template_function: The template function for the task
            
        Returns:
            A prompt string for parameter extraction
        """
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        
        prompt_content = f'''{task_prompt}
I have one algorithm that I want to optimize using classical numerical optimization techniques:
Algorithm description:
{indi.algorithm}
Code:
{str(indi)}

I need you to identify numerical parameters in this algorithm that could be optimized.
These parameters might include:
1. Numerical constants that affect algorithm behavior (weights, thresholds, coefficients)
2. Parameters that control convergence or termination
3. Parameters that balance exploration vs. exploitation
4. Other numerical values that significantly impact performance

Please do the following:
1. First, identify and list the key numerical parameters in the algorithm that could be optimized.
2. Second, describe how these parameters affect the algorithm's behavior in one sentence. The description must be inside within boxed {{}}.
3. Third, implement the function with the SAME code but add explanatory comments before each important numerical parameter.
   Format each comment as: # PARAM: description of what this parameter controls

{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content
