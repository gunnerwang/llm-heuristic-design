import sys
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to sys.path to find the modules
sys.path.append('../../')  

from llm4ad.task.optimization.agv_scheduling_ori import AGVEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.eoh import EoH



# Check for required environment variables
def check_api_key():
    """Check if API key is available in environment variables."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Please create a .env file with your API key.")
    return api_key

def main():
    # Set up the LLM API
    llm = HttpsApi(host='api.deepseek.com',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key=os.getenv('DEEPSEEK_API_KEY'),  # replace with your API key
                   model='deepseek-reasoner',  # replace with your preferred model
                   timeout=120)  # increased timeout for AGV scheduling

    # Initialize the AGV scheduling task
    task = AGVEvaluation(n_instance=6)  # use fewer instances for testing
    
    # Test the evaluation with a simple scheduler before running EoH
    print("Testing evaluation with a simple scheduler...")
    
    def simple_scheduler(env, agv_index, current_node):
        # """
        # A scheduler that tries to use all valid actions.
        # """
        # # Gather information about the environment state
        # parts_in_a = np.sum(env.A_part_info[:, 0]) > 0  # Are there parts at A?
        # b_has_space = np.sum(env.B_info[:, 0] == 0) > 0  # Is there space at B?
        
        # # Check for finished parts in B
        # b_has_finished_parts = False
        # for i in range(env.B_info.shape[0]):
        #     if env.B_info[i,0] >= 1 and env.AGV_timer[0] >= env.B_info[i,2]:
        #         b_has_finished_parts = True
        #         break
        
        # print(f"DEBUG - AGV {agv_index} at {current_node}:")
        # print(f"  Parts at A: {parts_in_a}, Space at B: {b_has_space}, Finished parts at B: {b_has_finished_parts}")
        # print(f"  Current time: {env.AGV_timer[agv_index]}")
        # print(f"  Action 0 valid: {env.check_action0_valide()}")
        # print(f"  Action 1 valid: {env.check_action1_valide()}")
        # print(f"  Action 2 valid: {env.check_action2_valide()}")
        # print(f"  Action 3 valid: {env.check_action3_valide()}")
        
        # # Try actions in priority order
        # if current_node == 'A':
        #     if env.check_action0_valide() and b_has_space:
        #         print(f"  Selecting action 0: A->B")
        #         return 0
        #     elif env.check_action2_valide() and not b_has_space:
        #         print(f"  Selecting action 2: A->C")
        #         return 2
        # elif current_node == 'B':
        #     if env.check_action1_valide() and b_has_finished_parts:
        #         print(f"  Selecting action 1: B->A")
        #         return 1
        #     elif env.check_action1_valide():
        #         print(f"  Selecting action 1: B->A (no finished parts, but valid)")
        #         return 1
        # elif current_node == 'C':
        #     if env.check_action3_valide() and b_has_space:
        #         print(f"  Selecting action 3: C->B")
        #         return 3
        
        # Fallback to greedy action selection if no direct match
        action = env.select_action_greedy(agv_index, current_node)
        print(f"  Using greedy selection, result: {action}")
        return action
    
    def simple_scheduler_2(env, agv_index, current_node):
        """
        Select the next action for the given AGV.
        
        Args:
            env: The AGV environment containing the current state
            agv_index: Index of the AGV to schedule
            current_node: Current location of the AGV ('A', 'B', or 'C')
            
        Returns:
            Action index (0, 1, 2, or 3) or -1 if no valid action is available
            
            Action 0: Take part and tray from A to B
            Action 1: Recycle tray from B to A
            Action 2: Take part and tray from A to C
            Action 3: Carry part and tray from C to B
        """
        if current_node == 'A':
            if env.check_action0_valide():
                return 0
            elif env.check_action2_valide():
                return 2
        elif current_node == 'B':
            if env.check_action1_valide():
                return 1
        elif current_node == 'C':
            if env.check_action3_valide():
                return 3
        
        return -1

    # Test the evaluation with the simple scheduler
    test_score = task.evaluate(simple_scheduler)
    print(f"Test score with simple scheduler: {test_score}")
    # exit()

    # Initialize the EoH method
    method = EoH(llm=llm,
                 profiler=ProfilerBase(log_dir='logs', log_style='complex'),
                 evaluation=task,
                 max_sample_nums=20,  # Reduced for faster testing
                 max_generations=5,   # Reduced for faster testing
                 pop_size=3,
                 num_samplers=1,
                 num_evaluators=1)

    # Run the EoH method
    method.run()


if __name__ == '__main__':
    main()