import sys
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to sys.path to find the modules
sys.path.append('../../')  

from llm4ad.task.optimization.agv_drone_scheduling.evaluation import VehicleSchedulingEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
# from llm4ad.method.eoh import EoH
from llm4ad.method.eohmeme import EoH

def main():
    # Set up the LLM API using environment variables
    
    # Get API configuration from environment variables
    api_key = os.getenv('DEEPSEEK_API_KEY')
    host = os.getenv('DEFAULT_LLM_HOST', 'api.deepseek.com')
    model = os.getenv('DEFAULT_LLM_MODEL', 'deepseek-chat')
    timeout = int(os.getenv('DEFAULT_LLM_TIMEOUT', '300'))
    
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Please create a .env file with your API key.")
    
    llm = HttpsApi(host=host,
                   key=api_key,
                   model=model,
                   timeout=timeout)

    # Alternative configurations (uncomment and modify .env file to use):
    
    # For Qwen API:
    # qwen_key = os.getenv('QWEN_API_KEY')
    # if qwen_key:
    #     llm = HttpsApi(host='dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',  
    #                    key=qwen_key,
    #                    model='qwen-plus',
    #                    timeout=300)
    
    # For OpenAI API:
    # openai_key = os.getenv('OPENAI_API_KEY')
    # if openai_key:
    #     llm = HttpsApi(host='api.openai-proxy.org',  
    #                    key=openai_key,
    #                    model='gpt-4o', 
    #                    timeout=120)

    # Initialize the AGV and drone scheduling task
    task = VehicleSchedulingEvaluation(n_instance=3)  # use fewer instances for testing
    
    # Test the evaluation with a simple scheduler before running EoH
    print("Testing evaluation with a simple scheduler...")
    
    def simple_scheduler(env, vehicle_index, current_node):
        """
        A simple scheduler for testing the AGV and drone environment.
        """
        # First check if vehicle needs charging
        if env.check_action4_valide(vehicle_index):
            return 4
            
        # Determine vehicle type
        is_drone = (env.vehicle_info[vehicle_index, 4] == 1)
        
        # Apply different strategies based on vehicle type
        if is_drone:
            # Drones are faster, prioritize longer routes
            if current_node == 'A':
                if env.check_action0_valide(vehicle_index):
                    return 0  # Take part from A to B
                elif env.check_action2_valide(vehicle_index):
                    return 2  # Take part from A to C
            elif current_node == 'B':
                if env.check_action1_valide(vehicle_index):
                    return 1  # Recycle tray from B to A
            elif current_node == 'C':
                if env.check_action3_valide(vehicle_index):
                    return 3  # Carry part from C to B
        else:
            # AGVs - standard priority
            if current_node == 'A':
                # First try to take parts directly to B
                if env.check_action0_valide(vehicle_index):
                    return 0
                # If B is full or no parts for B, consider C
                elif env.check_action2_valide(vehicle_index):
                    return 2
            elif current_node == 'B':
                if env.check_action1_valide(vehicle_index):
                    return 1  # Recycle tray
            elif current_node == 'C':
                if env.check_action3_valide(vehicle_index):
                    return 3  # Take from C to B
        
        # No valid action
        return -1

    # Test the evaluation with the simple scheduler
    test_score = task.evaluate(simple_scheduler)
    print(f"Test score with simple scheduler: {test_score}")

    # Initialize the EoH method
    method = EoH(llm=llm,
                 profiler=ProfilerBase(log_dir='logs', log_style='complex'),
                 evaluation=task,
                 max_sample_nums=100,     # Increased to allow more solution exploration
                 max_generations=10,      # Keep generation limit the same
                 pop_size=10,             # Slightly larger population to ensure diversity
                 num_samplers=1,
                 num_evaluators=1,
                 use_memetic=True,
                 memetic_frequency=1,
                 memetic_intensity=0.5,
                 use_hybrid_local_search=True,
                 hybrid_local_search_method='cma-es',
                 use_evolution_memory=True,  # Reflection now works without memory
                 memory_capacity=10,
                 use_reflection=True,         # Enable reflection mechanism based on best/worst samples
                 reflection_frequency=1,      # Reflect every 1 generation (EoH controls this frequency)
                 debug_mode=False,
                 )            
    

    # Run the EoH method
    method.run()


if __name__ == '__main__':
    main() 