"""
Enhanced LLM Wrapper for TSP Optimization
========================================

This module provides enhanced LLM functionality with comprehensive tracking for TSP optimization.

Features:
- Token usage tracking with detailed breakdown
- Cost calculation for different models
- Response time monitoring
- Enhanced error handling and retry logic
- Integration with comprehensive metrics system
- Detailed logging of all LLM interactions

Usage:
    enhanced_llm = TokenAwareLLMFactory.create_enhanced_llm(
        host='api.deepseek.com',
        key='your-key',
        model='deepseek-chat',
        profiler=profiler
    )
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass, asdict


# Token cost mapping for different models (prices per 1K tokens)
MODEL_COSTS = {
    'gpt-4': {'input': 0.03, 'output': 0.06},
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
    'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
    'deepseek-chat': {'input': 0.00014, 'output': 0.00028},
    'deepseek-coder': {'input': 0.00014, 'output': 0.00028},
    'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
    'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
    'claude-3-opus': {'input': 0.015, 'output': 0.075},
    # Add more models as needed
}


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call"""
    timestamp: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost_usd: float
    success: bool
    error_message: str = None
    response_content: str = None


class EnhancedLLMWrapper:
    """
    Enhanced wrapper around LLM API with comprehensive tracking
    """
    
    def __init__(self, base_llm, profiler=None, model_name: str = None):
        self.base_llm = base_llm
        self.profiler = profiler
        self.model_name = model_name or getattr(base_llm, 'model', 'unknown')
        
        # Initialize tracking
        self.call_records: List[LLMCallRecord] = []
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_time = 0.0
        self.failed_calls = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.model_name}")
        
    def __getattr__(self, name):
        """Forward any attribute access to the base LLM if not found in wrapper"""
        if hasattr(self.base_llm, name):
            attr = getattr(self.base_llm, name)
            if callable(attr):
                # If it's a method, wrap it to potentially add tracking
                def wrapper(*args, **kwargs):
                    return attr(*args, **kwargs)
                return wrapper
            else:
                # If it's a property, return it directly
                return attr
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}' and neither does the base LLM")
        
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage and model pricing"""
        if self.model_name not in MODEL_COSTS:
            # Use a default cost if model not found
            return (prompt_tokens + completion_tokens) * 0.0001  # Default rate
            
        costs = MODEL_COSTS[self.model_name]
        input_cost = (prompt_tokens / 1000) * costs['input']
        output_cost = (completion_tokens / 1000) * costs['output']
        return input_cost + output_cost
        
    def _extract_token_info(self, response) -> Dict[str, int]:
        """Extract token information from response"""
        # Try to extract token info from response
        # This implementation depends on the specific LLM API format
        token_info = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        # Handle different response formats
        if hasattr(response, 'usage'):
            # OpenAI-style response
            usage = response.usage
            token_info.update({
                'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(usage, 'completion_tokens', 0),
                'total_tokens': getattr(usage, 'total_tokens', 0)
            })
        elif isinstance(response, dict):
            # Dictionary response
            usage = response.get('usage', {})
            token_info.update({
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            })
        else:
            # Estimate tokens if not available (rough estimation)
            content = str(response)
            estimated_tokens = len(content.split()) * 1.3  # Rough estimation
            token_info['total_tokens'] = int(estimated_tokens)
            token_info['completion_tokens'] = int(estimated_tokens)
            
        return token_info
        
    def _extract_response_content(self, response) -> str:
        """Extract text content from response"""
        if hasattr(response, 'choices') and response.choices:
            # OpenAI-style response
            return response.choices[0].message.content
        elif isinstance(response, dict):
            # Dictionary response
            choices = response.get('choices', [])
            if choices:
                return choices[0].get('message', {}).get('content', str(response))
            return str(response)
        else:
            return str(response)
            
    def generate(self, messages, **kwargs):
        """Enhanced generate method with comprehensive tracking"""
        start_time = time.time()
        success = False
        error_message = None
        response = None
        
        try:
            # Make the actual LLM call
            response = self.base_llm.generate(messages, **kwargs)
            success = True
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"LLM call failed: {error_message}")
            self.failed_calls += 1
            raise
            
        finally:
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extract token information
            if success and response:
                token_info = self._extract_token_info(response)
                response_content = self._extract_response_content(response)
            else:
                token_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                response_content = None
                
            # Calculate cost
            cost = self._calculate_cost(token_info['prompt_tokens'], token_info['completion_tokens'])
            
            # Create call record
            call_record = LLMCallRecord(
                timestamp=start_time,
                model=self.model_name,
                prompt_tokens=token_info['prompt_tokens'],
                completion_tokens=token_info['completion_tokens'],
                total_tokens=token_info['total_tokens'],
                response_time=response_time,
                cost_usd=cost,
                success=success,
                error_message=error_message,
                response_content=response_content
            )
            
            # Update tracking
            self.call_records.append(call_record)
            self.total_calls += 1
            self.total_tokens += token_info['total_tokens']
            self.total_cost += cost
            self.total_time += response_time
            
            # Log to profiler if available
            if self.profiler:
                self.profiler.log_llm_call(
                    tokens_used=token_info['total_tokens'],
                    cost=cost,
                    call_time=response_time,
                    prompt_tokens=token_info['prompt_tokens'],
                    completion_tokens=token_info['completion_tokens']
                )
                
            # Log call details
            self.logger.info(f"LLM call completed: {token_info['total_tokens']} tokens, "
                           f"${cost:.6f}, {response_time:.3f}s, success={success}")
                           
        return response
        
    def draw_sample(self, prompt, *args, **kwargs):
        """Enhanced draw_sample method with comprehensive tracking"""
        start_time = time.time()
        success = False
        error_message = None
        response = None
        
        try:
            # Make the actual LLM call
            response = self.base_llm.draw_sample(prompt, *args, **kwargs)
            success = True
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"LLM draw_sample call failed: {error_message}")
            self.failed_calls += 1
            raise
            
        finally:
            end_time = time.time()
            response_time = end_time - start_time
            
            # For draw_sample, we need to estimate token usage since we don't get usage info
            if success and response:
                # Estimate token usage based on input and output text
                prompt_text = str(prompt) if isinstance(prompt, str) else str(prompt)
                response_text = str(response)
                
                # Rough estimation: 1 token ≈ 0.75 words
                prompt_tokens = int(len(prompt_text.split()) * 1.3)
                completion_tokens = int(len(response_text.split()) * 1.3)
                total_tokens = prompt_tokens + completion_tokens
                
                token_info = {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                }
            else:
                token_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                
            # Calculate cost
            cost = self._calculate_cost(token_info['prompt_tokens'], token_info['completion_tokens'])
            
            # Create call record
            call_record = LLMCallRecord(
                timestamp=start_time,
                model=self.model_name,
                prompt_tokens=token_info['prompt_tokens'],
                completion_tokens=token_info['completion_tokens'],
                total_tokens=token_info['total_tokens'],
                response_time=response_time,
                cost_usd=cost,
                success=success,
                error_message=error_message,
                response_content=response if success else None
            )
            
            # Update tracking
            self.call_records.append(call_record)
            self.total_calls += 1
            self.total_tokens += token_info['total_tokens']
            self.total_cost += cost
            self.total_time += response_time
            
            # Log to profiler if available
            if self.profiler:
                self.profiler.log_llm_call(
                    tokens_used=token_info['total_tokens'],
                    cost=cost,
                    call_time=response_time,
                    prompt_tokens=token_info['prompt_tokens'],
                    completion_tokens=token_info['completion_tokens']
                )
                
            # Log call details
            self.logger.info(f"LLM draw_sample completed: {token_info['total_tokens']} tokens, "
                           f"${cost:.6f}, {response_time:.3f}s, success={success}")
                           
        return response
        
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary"""
        avg_tokens_per_call = self.total_tokens / max(self.total_calls, 1)
        avg_cost_per_call = self.total_cost / max(self.total_calls, 1)
        avg_response_time = self.total_time / max(self.total_calls, 1)
        success_rate = (self.total_calls - self.failed_calls) / max(self.total_calls, 1)
        
        # Analyze call patterns
        successful_calls = [r for r in self.call_records if r.success]
        
        token_stats = {}
        cost_stats = {}
        time_stats = {}
        
        if successful_calls:
            tokens = [r.total_tokens for r in successful_calls]
            costs = [r.cost_usd for r in successful_calls]
            times = [r.response_time for r in successful_calls]
            
            token_stats = {
                'mean': np.mean(tokens),
                'std': np.std(tokens),
                'min': min(tokens),
                'max': max(tokens),
                'median': np.median(tokens)
            }
            
            cost_stats = {
                'mean': np.mean(costs),
                'std': np.std(costs),
                'min': min(costs),
                'max': max(costs),
                'median': np.median(costs)
            }
            
            time_stats = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': min(times),
                'max': max(times),
                'median': np.median(times)
            }
        
        return {
            'model': self.model_name,
            'total_calls': self.total_calls,
            'successful_calls': len(successful_calls),
            'failed_calls': self.failed_calls,
            'success_rate': success_rate,
            'total_tokens': self.total_tokens,
            'total_cost_usd': self.total_cost,
            'total_time': self.total_time,
            'averages': {
                'tokens_per_call': avg_tokens_per_call,
                'cost_per_call': avg_cost_per_call,
                'response_time': avg_response_time
            },
            'statistics': {
                'tokens': token_stats,
                'costs': cost_stats,
                'response_times': time_stats
            },
            'cost_analysis': {
                'total_cost_usd': self.total_cost,
                'cost_per_token': self.total_cost / max(self.total_tokens, 1),
                'estimated_monthly_cost': self.total_cost * 30,  # Rough estimate
                'cost_breakdown_by_model': {
                    self.model_name: self.total_cost
                }
            }
        }
        
    def export_detailed_log(self, filepath: str):
        """Export detailed call logs to file"""
        detailed_log = {
            'metadata': {
                'model': self.model_name,
                'total_calls': self.total_calls,
                'export_timestamp': time.time()
            },
            'usage_summary': self.get_usage_summary(),
            'call_records': [asdict(record) for record in self.call_records]
        }
        
        with open(filepath, 'w') as f:
            json.dump(detailed_log, f, indent=2)
            
        self.logger.info(f"Detailed log exported to {filepath}")
        
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get detailed cost analysis"""
        usage_summary = self.get_usage_summary()
        
        analysis = {
            'current_session': {
                'total_cost': self.total_cost,
                'cost_per_call': self.total_cost / max(self.total_calls, 1),
                'cost_per_token': self.total_cost / max(self.total_tokens, 1)
            },
            'projections': {
                'cost_per_100_calls': (self.total_cost / max(self.total_calls, 1)) * 100,
                'cost_per_10k_tokens': (self.total_cost / max(self.total_tokens, 1)) * 10000
            },
            'efficiency_metrics': {
                'tokens_per_dollar': self.total_tokens / max(self.total_cost, 1e-8),
                'calls_per_dollar': self.total_calls / max(self.total_cost, 1e-8)
            }
        }
        
        return analysis


class TokenAwareLLMFactory:
    """Factory for creating token-aware LLM instances"""
    
    @staticmethod
    def create_enhanced_llm(host: str, key: str, model: str, profiler=None, **kwargs):
        """Create an enhanced LLM with token tracking"""
        # Import the base LLM class
        from llm4ad.tools.llm.llm_api_https import HttpsApi
        
        # Create base LLM
        base_llm = HttpsApi(
            host=host,
            key=key,
            model=model,
            **kwargs
        )
        
        # Wrap with enhanced functionality
        enhanced_llm = EnhancedLLMWrapper(
            base_llm=base_llm,
            profiler=profiler,
            model_name=model
        )
        
        return enhanced_llm
        
    @staticmethod
    def create_cost_optimized_llm(budget_usd: float, expected_calls: int, **llm_kwargs):
        """Create LLM optimized for a given budget"""
        cost_per_call_budget = budget_usd / expected_calls
        
        # Recommend model based on budget
        recommended_models = []
        for model, costs in MODEL_COSTS.items():
            # Estimate cost per call (rough approximation)
            estimated_cost = (costs['input'] + costs['output']) * 0.5  # Assume 500 tokens average
            if estimated_cost <= cost_per_call_budget:
                recommended_models.append((model, estimated_cost))
                
        if recommended_models:
            # Choose the most capable model within budget
            recommended_models.sort(key=lambda x: x[1], reverse=True)
            chosen_model = recommended_models[0][0]
            
            print(f"Budget-optimized model selection: {chosen_model}")
            print(f"Estimated cost per call: ${recommended_models[0][1]:.6f}")
            
            llm_kwargs['model'] = chosen_model
        
        return TokenAwareLLMFactory.create_enhanced_llm(**llm_kwargs)


def estimate_experiment_cost(num_generations: int, pop_size: int, num_evaluations: int, 
                           model: str = 'deepseek-chat', tokens_per_call: int = 1000) -> Dict:
    """Estimate the cost of running a TSP optimization experiment"""
    
    if model not in MODEL_COSTS:
        raise ValueError(f"Model {model} not found in cost database")
        
    costs = MODEL_COSTS[model]
    
    # Estimate total LLM calls
    # This is a rough estimation based on typical EoH usage patterns
    calls_per_generation = pop_size * 2  # Sampling and potential reflection
    total_calls = num_generations * calls_per_generation
    
    # Calculate costs
    total_tokens = total_calls * tokens_per_call
    cost_per_1k_tokens = (costs['input'] + costs['output']) / 2  # Average input/output
    total_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    estimate = {
        'experiment_parameters': {
            'num_generations': num_generations,
            'pop_size': pop_size,
            'num_evaluations': num_evaluations,
            'model': model,
            'tokens_per_call': tokens_per_call
        },
        'cost_estimate': {
            'total_calls': total_calls,
            'total_tokens': total_tokens,
            'total_cost_usd': total_cost,
            'cost_per_generation': total_cost / num_generations,
            'cost_per_evaluation': total_cost / num_evaluations
        },
        'model_pricing': costs
    }
    
    return estimate 