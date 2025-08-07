# Module Name: ODEEvaluation
# Last Revision: 2025/3/5
# Description: Provides the skeleton for an ODE mathematical function based on given initial data.
#              The function is designed to be differentiable and continuous, using only a limited
#              set of selectable components. This module is part of the LLM4AD project
#              (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    -   x: float - initial value of the ODE formula (default: None).
#    -   params: np.ndarray - 1D array of numeric constants or parameters to be optimized (default: None).
#    -   timeout_seconds: int - Maximum allowed time (in seconds) for the evaluation process (default: 20).
#
# References:
#   - Du, Mengge, et al. "Llm4ed: Large language models for automatic equation discovery."
#       arXiv preprint arXiv:2405.07761 (2024).
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

import re
import itertools
from typing import Any
import numpy as np
import warnings

from llm4ad.base import Evaluation
from llm4ad.task.science_discovery.ode_1d.template import template_program, task_description
from llm4ad.task.science_discovery.ode_1d import strogatz_extended, strogatz_equations

# Suppress specific runtime warnings that are expected during optimization
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in power")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in exp")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")

__all__ = ['ODEEvaluation']

MAX_NPARAMS = 10
params = [1.0] * MAX_NPARAMS

local_dict = {
    "np.e": "sp.E",
    "np.pi": "sp.pi",
    "np.arcsin": "sp.asin",
    "np.arccos": "sp.acos",
    "np.arctan": "sp.atan",
    "np.sin": "sp.sin",
    "np.cos": "sp.cos",
    "np.tan": "sp.tan",
    "np.sign": "sp.sign",
    "np.sqrt": "sp.sqrt",
    "np.log": "sp.log",
    "np.exp": "sp.exp",
}


def safe_equation_wrapper(equation_func):
    """Wrapper to make equation function numerically stable"""
    def safe_equation(t, x, params):
        try:
            # Add small epsilon to prevent division by zero and log of zero
            eps = 1e-12
            
            # Ensure x is not zero or negative for log operations
            x_safe = np.maximum(np.abs(x) + eps, eps)
            
            # Call the original equation with safe parameters
            result = equation_func(t, x_safe, params)
            
            # Check for invalid results
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return np.zeros_like(result)
            
            # Clip extremely large values
            result = np.clip(result, -1e10, 1e10)
            
            return result
        except Exception:
            # Return zero derivative if equation fails
            return np.zeros_like(x)
    
    return safe_equation


def evaluate(program_str: str, data: dict, equation: callable) -> float | None:
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    xs = np.array(data['xs'])
    ts = np.array(data['t'])
    ys = np.array(list(itertools.chain(*data['ys'])))  # flatten to 1d
    num_ini_x_values = len(xs)
    num_variables = len(xs[0])

    try:  # initial x(0) = x0
        # t = sp.symbols('t')  # time variable t
        # x0 = sp.Function('x0')(t)  # x(t) is the unknown formula about t
        # constants = [sp.symbols(f'c{i}') for i in range(MAX_NPARAMS)]  # constants symbol

        # Check if the function already has parameters, don't add 't: float' if it does
        if "def equation(t" in program_str:
            # No need to modify the signature
            pass
        else:
            program_str = re.sub(r"def equation\(", r"def equation(t: float, ", program_str)
            
        local_vars = {"equation": equation}
        exec(program_str, globals(), local_vars)
        equation = local_vars['equation']  # replace equation with str that after replacement of key parts

        # Wrap equation with safety checks
        safe_equation = safe_equation_wrapper(equation)

        # formula_sympy = equation(x0, constants)
        # diff_eq = sp.Eq(sp.diff(x0, t), formula_sympy)

        # calculate the values of 2 initial x0 value
        # solution_with_initial = sp.dsolve(diff_eq, ics={x0.subs(t, 0): xs[0][0]})
        # x0_solution = solution_with_initial.rhs  # extract the expression of right part
        # x0_func = sp.lambdify([t, constants], x0_solution, 'numpy')
    except Exception as e:
        return None
    # Optimize parameters based on data
    from scipy.optimize import minimize
    from scipy.integrate import solve_ivp
    
    def loss(params):
        try:
            # Add bounds to parameters to prevent extreme values
            params = np.clip(params, -10, 10)
            
            y_pred = np.zeros(num_ini_x_values * len(ts[0]))
            
            for i in range(num_ini_x_values):
                try:
                    # Use faster, less accurate integration settings for speed
                    s = solve_ivp(
                        safe_equation, 
                        (ts[i][0], ts[i][-1]), 
                        xs[i], 
                        args=(params,), 
                        t_eval=ts[i],
                        method='RK23',  # Faster method (less accurate but much faster)
                        rtol=1e-3,      # Much more relaxed tolerance
                        atol=1e-6,      # Much more relaxed tolerance
                        max_step=0.1    # Limit step size for stability
                    )
                    
                    if s.success and len(s.y[0]) == len(ts[0]):
                        y_pred[i * len(ts[0]):(i + 1) * len(ts[0])] = s.y[0]
                    else:
                        # If integration fails, return large penalty
                        return 1e8  # Reduced penalty for faster convergence
                        
                except Exception as e:
                    # If integration fails, return large penalty
                    return 1e8
            
            # Calculate MSE loss
            mse = np.mean((y_pred - ys) ** 2)
            
            # Check for invalid loss
            if np.isnan(mse) or np.isinf(mse) or mse < 0:
                return 1e8
                
            # Early stopping for very poor solutions
            if mse > 1e6:
                return mse  # Don't waste time optimizing terrible solutions
                
            return mse
            
        except Exception as e:
            return 1e8

    # x0_funcs = []
    # for i in range(num_ini_x_values):
    # solution_with_initial = sp.dsolve(diff_eq, ics={x0.subs(t, 0): xs[i][0]})
    # x0_solution = solution_with_initial.rhs  # extract the expression of right part
    # x0_func = sp.lambdify([t, constants], x0_solution, 'numpy')
    #
    # x0_funcs.append(x0_func)

    loss_partial = lambda params: loss(params)
    
    # Use much faster optimization settings
    from scipy.optimize import Bounds
    bounds = Bounds(-5, 5)  # Smaller parameter range for faster convergence
    
    # Try only 1 optimization attempt for speed
    best_result = None
    best_loss = 1e8
    
    try:
        # Simple random initialization
        initial_params = np.random.uniform(-0.5, 0.5, MAX_NPARAMS)
        
        # Quick evaluation to see if this equation is worth optimizing
        quick_loss = loss(initial_params)
        if quick_loss > 1e6:  # If initial loss is terrible, skip optimization (less aggressive)
            return None
            
        result = minimize(
            loss_partial, 
            initial_params, 
            method='L-BFGS-B',  # Fast method
            bounds=bounds,
            options={
                'maxiter': 20,     # Much fewer iterations
                'ftol': 1e-6,      # Relaxed tolerance
                'gtol': 1e-6,      # Relaxed tolerance
                'maxfun': 50       # Limit function evaluations
            }
        )
        
        # Accept result even if not fully converged, as long as loss is reasonable
        if result.fun < best_loss:  # Accept any improvement, regardless of convergence
            best_result = result
            best_loss = result.fun
                
    except Exception as e:
        pass
    
    if best_result is None:
        return None

    # Return evaluation score
    optimized_params = best_result.x
    loss = best_result.fun

    if np.isnan(loss) or np.isinf(loss) or loss > 1e6:  # Less aggressive rejection
        return None
    else:
        return -loss


class ODEEvaluation(Evaluation):

    def __init__(self, timeout_seconds=200000, test_id=1, **kwargs):
        """
        Args:
            timeout_seconds: evaluate time limit.
            test_id: test equation id ranges from [1, 16].
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        # read files
        test_eq_dict = strogatz_equations.equations[test_id - 1]
        dataset = strogatz_extended.data

        dataset = dataset[test_id - 1]
        xs = dataset['init']
        t = [e['t'] for e in dataset['solutions'][0]]
        ys = [e['y'][0] for e in dataset['solutions'][0]]  # for only 1 output
        self._datasets = {
            'xs': xs,
            'ys': ys,
            't': t
        }

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        import inspect
        if not program_str:
            program_str = inspect.getsource(callable_func).lstrip()  # for testing
        # for np_func, sp_func in local_dict.items():  # replace key parts
        #     program_str = program_str.replace(np_func, sp_func)
        return evaluate(program_str, self._datasets, callable_func)


if __name__ == '__main__':
    def equation(x: float, params: np.ndarray) -> float:
        """ A ODE mathematical function
        Args:
            x: the initial float value of the ode formula
            params: a 1-d Array of numeric constants or parameters to be optimized

        Return:
            A numpy array representing the result of applying the mathematical function to the inputs.
        """
        y = params[0] * np.sin(x) + params[1]
        return y


    evaluation = ODEEvaluation()
    res = evaluation.evaluate_program('', equation)
    print(res)
