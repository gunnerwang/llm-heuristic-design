"""
Enhanced LLM Wrapper for ODE Discovery
=====================================

This module provides enhanced LLM functionality with comprehensive token tracking,
cost monitoring, and performance analysis for ODE discovery experiments.

Features:
- Token usage tracking with detailed breakdown
- Cost calculation and monitoring
- API call timing and performance metrics
- Request/response logging and analysis
- Integration with comprehensive profiler
- Support for multiple LLM providers

Usage:
    enhanced_llm = TokenAwareLLMFactory.create_enhanced_llm(
        host='api.deepseek.com',
        key='your-api-key',
        model='deepseek-chat',
        profiler=profiler
    )
"""

import time
import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

# Import base LLM from LLM4AD framework
from llm4ad.tools.llm.llm_api_https import HttpsApi


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call"""
    timestamp: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    response_time: float
    model: str
    success: bool
    error_message: str = None
    prompt_preview: str = ""  # First 100 chars of prompt
    completion_preview: str = ""  # First 100 chars of completion


@dataclass
class LLMUsageStats:
    """Aggregated LLM usage statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_response_time: float = 0.0
    avg_response_time: float = 0.0
    avg_tokens_per_call: float = 0.0
    cost_per_token: float = 0.0
    success_rate: float = 0.0


class EnhancedLLMWrapper:
    """
    Enhanced wrapper around LLM API with comprehensive tracking
    """
    
    def __init__(self, base_llm: HttpsApi, profiler=None, 
                 cost_per_1k_tokens: float = 0.0014):  # DeepSeek pricing
        self.base_llm = base_llm
        self.profiler = profiler
        self.cost_per_1k_tokens = cost_per_1k_tokens
        
        # Framework compatibility attributes
        self.do_auto_trim = getattr(base_llm, 'do_auto_trim', False)
        
        # Tracking data
        self.call_records: List[LLMCallRecord] = []
        self.usage_stats = LLMUsageStats()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Enhanced LLM call with comprehensive tracking
        """
        start_time = time.time()
        
        try:
            # Call the base LLM using its draw_sample method
            response = self.base_llm.draw_sample(prompt, **kwargs)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Estimate token usage (rough approximation)
            prompt_tokens = self._estimate_tokens(prompt)
            completion_tokens = self._estimate_tokens(response)
            total_tokens = prompt_tokens + completion_tokens
            
            # Calculate cost
            cost_usd = (total_tokens / 1000.0) * self.cost_per_1k_tokens
            
            # Create call record
            call_record = LLMCallRecord(
                timestamp=start_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                response_time=response_time,
                model=getattr(self.base_llm, '_model', 'unknown'),
                success=True,
                prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                completion_preview=response[:100] + "..." if len(response) > 100 else response
            )
            
            # Update tracking
            self._update_tracking(call_record)
            
            # Log to profiler if available
            if self.profiler:
                self.profiler.log_llm_call(
                    tokens_used=total_tokens,
                    cost=cost_usd,
                    call_time=response_time,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                
            self.logger.info(f"LLM call successful - Tokens: {total_tokens}, Cost: ${cost_usd:.4f}, Time: {response_time:.2f}s")
            
            return response
            
        except Exception as e:
            # Handle failed calls
            response_time = time.time() - start_time
            
            call_record = LLMCallRecord(
                timestamp=start_time,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                response_time=response_time,
                model=getattr(self.base_llm, '_model', 'unknown'),
                success=False,
                error_message=str(e),
                prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt
            )
            
            self._update_tracking(call_record)
            
            self.logger.error(f"LLM call failed - Error: {str(e)}, Time: {response_time:.2f}s")
            
            # Re-raise the exception
            raise e
            
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (1 token ≈ 4 characters for English)
        This is a simple approximation - actual tokenization may vary
        """
        return max(1, len(text) // 4)
        
    def _update_tracking(self, call_record: LLMCallRecord):
        """Update usage statistics with new call record"""
        self.call_records.append(call_record)
        
        # Update aggregated stats
        self.usage_stats.total_calls += 1
        
        if call_record.success:
            self.usage_stats.successful_calls += 1
            self.usage_stats.total_tokens += call_record.total_tokens
            self.usage_stats.total_prompt_tokens += call_record.prompt_tokens
            self.usage_stats.total_completion_tokens += call_record.completion_tokens
            self.usage_stats.total_cost_usd += call_record.cost_usd
        else:
            self.usage_stats.failed_calls += 1
            
        self.usage_stats.total_response_time += call_record.response_time
        
        # Calculate derived metrics
        if self.usage_stats.total_calls > 0:
            self.usage_stats.avg_response_time = self.usage_stats.total_response_time / self.usage_stats.total_calls
            self.usage_stats.success_rate = self.usage_stats.successful_calls / self.usage_stats.total_calls
            
        if self.usage_stats.successful_calls > 0:
            self.usage_stats.avg_tokens_per_call = self.usage_stats.total_tokens / self.usage_stats.successful_calls
            
        if self.usage_stats.total_tokens > 0:
            self.usage_stats.cost_per_token = self.usage_stats.total_cost_usd / self.usage_stats.total_tokens
            
    def get_usage_summary(self) -> Dict:
        """Get comprehensive usage summary"""
        return {
            'basic_stats': asdict(self.usage_stats),
            'recent_performance': self._get_recent_performance(),
            'cost_breakdown': self._get_cost_breakdown(),
            'performance_trends': self._get_performance_trends()
        }
        
    def _get_recent_performance(self, window_size: int = 10) -> Dict:
        """Get performance metrics for recent calls"""
        if not self.call_records:
            return {}
            
        recent_calls = self.call_records[-window_size:]
        
        successful_recent = [r for r in recent_calls if r.success]
        
        if not successful_recent:
            return {'recent_success_rate': 0.0}
            
        return {
            'recent_success_rate': len(successful_recent) / len(recent_calls),
            'recent_avg_response_time': sum(r.response_time for r in successful_recent) / len(successful_recent),
            'recent_avg_tokens': sum(r.total_tokens for r in successful_recent) / len(successful_recent),
            'recent_cost': sum(r.cost_usd for r in successful_recent)
        }
        
    def _get_cost_breakdown(self) -> Dict:
        """Get detailed cost breakdown"""
        if not self.call_records:
            return {}
            
        successful_calls = [r for r in self.call_records if r.success]
        
        if not successful_calls:
            return {'total_cost': 0.0}
            
        costs = [r.cost_usd for r in successful_calls]
        
        return {
            'total_cost': sum(costs),
            'avg_cost_per_call': sum(costs) / len(costs),
            'max_cost_per_call': max(costs),
            'min_cost_per_call': min(costs),
            'cost_distribution': {
                'p25': sorted(costs)[len(costs)//4] if len(costs) >= 4 else costs[0],
                'p50': sorted(costs)[len(costs)//2] if len(costs) >= 2 else costs[0],
                'p75': sorted(costs)[3*len(costs)//4] if len(costs) >= 4 else costs[-1],
            }
        }
        
    def _get_performance_trends(self) -> Dict:
        """Analyze performance trends over time"""
        if len(self.call_records) < 5:
            return {}
            
        # Split into first half and second half to see trends
        mid = len(self.call_records) // 2
        first_half = [r for r in self.call_records[:mid] if r.success]
        second_half = [r for r in self.call_records[mid:] if r.success]
        
        if not first_half or not second_half:
            return {}
            
        first_avg_time = sum(r.response_time for r in first_half) / len(first_half)
        second_avg_time = sum(r.response_time for r in second_half) / len(second_half)
        
        first_avg_tokens = sum(r.total_tokens for r in first_half) / len(first_half)
        second_avg_tokens = sum(r.total_tokens for r in second_half) / len(second_half)
        
        return {
            'response_time_trend': 'improving' if second_avg_time < first_avg_time else 'degrading',
            'response_time_change': ((second_avg_time - first_avg_time) / first_avg_time) * 100,
            'token_usage_trend': 'increasing' if second_avg_tokens > first_avg_tokens else 'decreasing',
            'token_usage_change': ((second_avg_tokens - first_avg_tokens) / first_avg_tokens) * 100
        }
        
    def export_detailed_log(self, filepath: str):
        """Export detailed call log to JSON file"""
        log_data = {
            'experiment_info': {
                'model': getattr(self.base_llm, '_model', 'unknown'),
                'host': getattr(self.base_llm, '_host', 'unknown'),
                'cost_per_1k_tokens': self.cost_per_1k_tokens,
                'export_timestamp': datetime.now().isoformat()
            },
            'usage_summary': self.get_usage_summary(),
            'detailed_calls': [asdict(record) for record in self.call_records]
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
            
        self.logger.info(f"Detailed LLM log exported to {filepath}")
        
    def reset_tracking(self):
        """Reset all tracking data"""
        self.call_records = []
        self.usage_stats = LLMUsageStats()
        self.logger.info("LLM tracking data reset")

    def draw_sample(self, prompt: str, **kwargs) -> str:
        """
        Draw sample method for compatibility with EoH framework
        This is an alias for __call__ to maintain compatibility with the sampler
        """
        return self.__call__(prompt, **kwargs)
    
    def draw_samples(self, prompts: list, **kwargs) -> list:
        """
        Draw multiple samples method for compatibility with FunSearch/HillClimb framework
        
        Args:
            prompts: List of prompts to process
            **kwargs: Additional arguments passed to individual calls
            
        Returns:
            List of responses corresponding to the input prompts
        """
        responses = []
        for prompt in prompts:
            try:
                response = self.__call__(prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Failed to process prompt: {str(e)}")
                responses.append("")  # Return empty string on failure
        return responses


class TokenAwareLLMFactory:
    """Factory for creating token-aware LLM instances"""
    
    # Pricing data for different models (cost per 1K tokens)
    MODEL_PRICING = {
        'deepseek-chat': 0.0014,
        'gpt-3.5-turbo': 0.002,
        'gpt-4': 0.03,
        'gpt-4-turbo': 0.01,
        'claude-3-sonnet': 0.015,
        'claude-3-haiku': 0.0025
    }
    
    @classmethod
    def create_enhanced_llm(cls, host: str, key: str, model: str, 
                           profiler=None, timeout: int = 60,
                           custom_pricing: float = None) -> EnhancedLLMWrapper:
        """
        Create an enhanced LLM wrapper with comprehensive tracking
        
        Args:
            host: API host endpoint
            key: API key
            model: Model name
            profiler: Optional profiler for integration
            timeout: Request timeout in seconds
            custom_pricing: Custom pricing per 1K tokens
            
        Returns:
            Enhanced LLM wrapper with tracking capabilities
        """
        # Create base LLM
        base_llm = HttpsApi(host=host, key=key, model=model, timeout=timeout)
        
        # Determine pricing
        cost_per_1k_tokens = custom_pricing or cls.MODEL_PRICING.get(model, 0.002)  # Default to GPT-3.5 pricing
        
        # Create enhanced wrapper
        enhanced_llm = EnhancedLLMWrapper(
            base_llm=base_llm, 
            profiler=profiler,
            cost_per_1k_tokens=cost_per_1k_tokens
        )
        
        logging.getLogger(__name__).info(
            f"Created enhanced LLM: {model} on {host} with ${cost_per_1k_tokens:.4f}/1K tokens"
        )
        
        return enhanced_llm
        
    @classmethod
    def get_supported_models(cls) -> Dict[str, float]:
        """Get dictionary of supported models and their pricing"""
        return cls.MODEL_PRICING.copy()
        
    @classmethod
    def estimate_cost(cls, model: str, total_tokens: int) -> float:
        """Estimate cost for given model and token count"""
        cost_per_1k = cls.MODEL_PRICING.get(model, 0.002)
        return (total_tokens / 1000.0) * cost_per_1k 