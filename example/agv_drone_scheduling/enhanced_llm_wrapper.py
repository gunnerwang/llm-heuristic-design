import time
import json
import re
from typing import Optional, Dict, Any, Union, List
from llm4ad.tools.llm.llm_api_https import HttpsApi
from comprehensive_metrics import ComprehensiveProfiler

class EnhancedLLMWrapper:
    """增强版LLM包装器，自动追踪token使用和性能指标"""
    
    def __init__(self, base_llm: HttpsApi, profiler: Optional[ComprehensiveProfiler] = None):
        self.base_llm = base_llm
        self.profiler = profiler
        self.total_calls = 0
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_time = 0.0
        self.call_history = []
        
        # Token计费策略（根据不同模型调整）
        self.pricing = self._get_pricing_info()
        
        # 调试模式：从base_llm复制
        self.debug_mode = getattr(base_llm, 'debug_mode', False)
        self.do_auto_trim = getattr(base_llm, 'do_auto_trim', True)
        
    def _get_pricing_info(self) -> Dict[str, Dict[str, float]]:
        """获取不同模型的计费信息（每1000tokens的价格，单位：美元）"""
        return {
            "deepseek-chat": {"input": 0.0014, "output": 0.0028},
            "gpt-4o": {"input": 0.0050, "output": 0.0150},
            "gpt-4o-mini": {"input": 0.0001, "output": 0.0004},
            "qwen-plus": {"input": 0.0008, "output": 0.0020},
            "claude-3-sonnet": {"input": 0.0030, "output": 0.0150},
            "default": {"input": 0.0010, "output": 0.0030}  # 默认价格
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数量（简单估算，实际应使用tokenizer）"""
        # 简单的估算方法：大约4个字符=1个token（对于英文）
        # 对于中文，大约1.5个字符=1个token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(text) - chinese_chars
        
        estimated_tokens = int(english_chars / 4 + chinese_chars / 1.5)
        return max(estimated_tokens, 1)
    
    def _parse_token_usage_from_response(self, response_data: Dict) -> Dict[str, int]:
        """从API响应中解析token使用情况"""
        usage = {}
        
        # OpenAI格式
        if 'usage' in response_data:
            usage_data = response_data['usage']
            usage = {
                'prompt_tokens': usage_data.get('prompt_tokens', 0),
                'completion_tokens': usage_data.get('completion_tokens', 0),
                'total_tokens': usage_data.get('total_tokens', 0)
            }
        
        # DeepSeek格式
        elif 'usage' in response_data:
            usage_data = response_data['usage']
            usage = {
                'prompt_tokens': usage_data.get('prompt_tokens', 0),
                'completion_tokens': usage_data.get('completion_tokens', 0),
                'total_tokens': usage_data.get('total_tokens', 0)
            }
        
        # 如果没有找到usage信息，进行估算
        if not usage:
            # 这里需要访问原始prompt和response，但由于API限制，我们做简单估算
            usage = {
                'prompt_tokens': 0,  # 无法获取原始prompt
                'completion_tokens': 0,  # 无法获取原始response
                'total_tokens': 0
            }
            
        return usage
    
    def draw_sample(self, prompt: Union[str, Any], *args, **kwargs) -> str:
        """包装的采样方法，自动追踪指标"""
        start_time = time.time()
        
        try:
            # 估算输入token数
            if isinstance(prompt, str):
                estimated_input_tokens = self._estimate_tokens(prompt)
            elif isinstance(prompt, list):
                # 对于消息列表格式
                total_text = ""
                for msg in prompt:
                    if isinstance(msg, dict) and 'content' in msg:
                        total_text += msg['content']
                estimated_input_tokens = self._estimate_tokens(total_text)
            else:
                estimated_input_tokens = 100  # 默认估算
            
            # 调用原始LLM
            response = self.base_llm.draw_sample(prompt, *args, **kwargs)
            
            # 计算时间
            call_time = time.time() - start_time
            
            # 估算输出token数
            estimated_output_tokens = self._estimate_tokens(response)
            estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
            
            # 更新统计信息
            self.total_calls += 1
            self.total_input_tokens += estimated_input_tokens
            self.total_output_tokens += estimated_output_tokens
            self.total_tokens += estimated_total_tokens
            self.total_time += call_time
            
            # 计算成本
            model_name = getattr(self.base_llm, '_model', 'default')
            pricing = self.pricing.get(model_name, self.pricing['default'])
            input_cost = (estimated_input_tokens / 1000) * pricing['input']
            output_cost = (estimated_output_tokens / 1000) * pricing['output']
            total_cost = input_cost + output_cost
            
            # 记录调用历史
            call_record = {
                'timestamp': time.time(),
                'call_number': self.total_calls,
                'prompt_tokens': estimated_input_tokens,
                'completion_tokens': estimated_output_tokens,
                'total_tokens': estimated_total_tokens,
                'call_time': call_time,
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost,
                'model': model_name
            }
            self.call_history.append(call_record)
            
            # 如果有profiler，记录到profiler中
            if self.profiler:
                self.profiler.record_llm_call(
                    prompt_tokens=estimated_input_tokens,
                    completion_tokens=estimated_output_tokens,
                    total_tokens=estimated_total_tokens,
                    call_time=call_time
                )
            
            return response
            
        except Exception as e:
            call_time = time.time() - start_time
            self.total_time += call_time
            
            # 记录失败的调用
            failed_record = {
                'timestamp': time.time(),
                'call_number': self.total_calls + 1,
                'error': str(e),
                'call_time': call_time,
                'success': False
            }
            self.call_history.append(failed_record)
            
            raise e

    def draw_samples(self, prompts: List[Union[str, Any]], *args, **kwargs) -> List[str]:
        """包装的批量采样方法，自动追踪指标"""
        return [self.draw_sample(p, *args, **kwargs) for p in prompts]
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """获取使用情况摘要"""
        model_name = getattr(self.base_llm, '_model', 'default')
        pricing = self.pricing.get(model_name, self.pricing['default'])
        
        total_input_cost = (self.total_input_tokens / 1000) * pricing['input']
        total_output_cost = (self.total_output_tokens / 1000) * pricing['output']
        total_cost = total_input_cost + total_output_cost
        
        successful_calls = len([call for call in self.call_history if call.get('success', True)])
        failed_calls = len([call for call in self.call_history if not call.get('success', True)])
        
        return {
            'total_calls': self.total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': successful_calls / max(self.total_calls, 1),
            'total_tokens': self.total_tokens,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_time': self.total_time,
            'average_time_per_call': self.total_time / max(self.total_calls, 1),
            'tokens_per_second': self.total_tokens / max(self.total_time, 1e-6),
            'cost_analysis': {
                'input_cost_usd': total_input_cost,
                'output_cost_usd': total_output_cost,
                'total_cost_usd': total_cost,
                'cost_per_call': total_cost / max(self.total_calls, 1),
                'tokens_per_dollar': self.total_tokens / max(total_cost, 1e-6)
            },
            'model_info': {
                'model_name': model_name,
                'pricing_per_1k_tokens': pricing
            }
        }
    
    def get_cost_breakdown_by_time(self, time_windows: list = None) -> Dict[str, Any]:
        """按时间窗口分析成本分布"""
        if time_windows is None:
            time_windows = [300, 900, 1800, 3600]  # 5分钟、15分钟、30分钟、1小时
        
        if not self.call_history:
            return {}
        
        start_time = self.call_history[0]['timestamp']
        breakdown = {}
        
        for window in time_windows:
            window_data = {
                'calls': 0,
                'tokens': 0,
                'cost': 0.0,
                'time_spent': 0.0
            }
            
            for call in self.call_history:
                if call.get('success', True) and call['timestamp'] - start_time <= window:
                    window_data['calls'] += 1
                    window_data['tokens'] += call.get('total_tokens', 0)
                    window_data['cost'] += call.get('total_cost', 0)
                    window_data['time_spent'] += call.get('call_time', 0)
            
            breakdown[f'{window}_seconds'] = window_data
        
        return breakdown
    
    def export_detailed_log(self, filename: str):
        """导出详细的调用日志"""
        export_data = {
            'summary': self.get_usage_summary(),
            'cost_breakdown': self.get_cost_breakdown_by_time(),
            'detailed_calls': self.call_history
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_time = 0.0
        self.call_history = []
    
    # 转发其他属性到原始LLM对象
    def __getattr__(self, name):
        return getattr(self.base_llm, name)

class TokenAwareLLMFactory:
    """支持token追踪的LLM工厂"""
    
    @staticmethod
    def create_enhanced_llm(host: str, key: str, model: str, 
                           profiler: Optional[ComprehensiveProfiler] = None,
                           **kwargs) -> EnhancedLLMWrapper:
        """创建增强版LLM"""
        base_llm = HttpsApi(host=host, key=key, model=model, **kwargs)
        return EnhancedLLMWrapper(base_llm, profiler)
    
    @staticmethod
    def create_multiple_llms(configs: list, 
                           profiler: Optional[ComprehensiveProfiler] = None) -> Dict[str, EnhancedLLMWrapper]:
        """创建多个LLM实例用于对比实验"""
        llms = {}
        for i, config in enumerate(configs):
            name = config.get('name', f'llm_{i}')
            llm = TokenAwareLLMFactory.create_enhanced_llm(
                host=config['host'],
                key=config['key'], 
                model=config['model'],
                profiler=profiler,
                **config.get('kwargs', {})
            )
            llms[name] = llm
        return llms

# 使用示例和测试函数
def compare_llm_efficiency():
    """比较不同LLM的效率"""
    # 配置多个LLM
    llm_configs = [
        {
            'name': 'deepseek',
            'host': 'api.deepseek.com',
            'key': 'your-deepseek-key',
            'model': 'deepseek-chat'
        },
        {
            'name': 'gpt4o-mini',
            'host': 'api.openai.com',
            'key': 'your-openai-key', 
            'model': 'gpt-4o-mini'
        }
    ]
    
    # 创建profiler
    profiler = ComprehensiveProfiler(log_dir="llm_comparison_logs")
    
    # 创建LLM实例
    llms = TokenAwareLLMFactory.create_multiple_llms(llm_configs, profiler)
    
    # 测试prompt
    test_prompts = [
        "解释进化算法的基本原理",
        "设计一个调度算法来优化AGV和无人机的协同工作",
        "分析以下代码的时间复杂度：[代码示例]"
    ]
    
    results = {}
    for name, llm in llms.items():
        print(f"Testing {name}...")
        start_time = time.time()
        
        for prompt in test_prompts:
            try:
                response = llm.draw_sample(prompt)
                print(f"Response length: {len(response)}")
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        total_time = time.time() - start_time
        usage_summary = llm.get_usage_summary()
        
        results[name] = {
            'usage_summary': usage_summary,
            'total_test_time': total_time
        }
        
        # 导出详细日志
        llm.export_detailed_log(f"llm_log_{name}.json")
    
    # 生成比较报告
    comparison = {}
    for name, result in results.items():
        summary = result['usage_summary']
        comparison[name] = {
            'cost_per_response': summary['cost_analysis']['total_cost_usd'] / len(test_prompts),
            'time_per_response': summary['average_time_per_call'],
            'tokens_per_response': summary['total_tokens'] / len(test_prompts),
            'efficiency_score': summary['total_tokens'] / max(summary['cost_analysis']['total_cost_usd'], 1e-6)
        }
    
    print("\n=== LLM效率对比 ===")
    for name, metrics in comparison.items():
        print(f"\n{name}:")
        print(f"  平均每次响应成本: ${metrics['cost_per_response']:.6f}")
        print(f"  平均每次响应时间: {metrics['time_per_response']:.2f}秒")
        print(f"  平均每次响应tokens: {metrics['tokens_per_response']:.0f}")
        print(f"  效率分数 (tokens/$): {metrics['efficiency_score']:.0f}")
    
    return results, comparison

if __name__ == "__main__":
    # 运行LLM效率对比测试
    results, comparison = compare_llm_efficiency() 