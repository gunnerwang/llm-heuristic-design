import sys
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.tsp_construct import TSPEvaluation
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
    llm = HttpsApi(host='api.deepseek.com',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key=os.getenv('DEEPSEEK_API_KEY'),  # your key, e.g., 'sk-abcdefghijklmn'
                   model='deepseek-chat',  # your llm, e.g., 'gpt-3.5-turbo'
                   timeout=60)

    task = TSPEvaluation()

    method = EoH(llm=llm,
                 profiler=ProfilerBase(log_dir='logs', log_style='complex'),
                 evaluation=task,
                 max_sample_nums=20,
                 max_generations=5,
                 pop_size=2,
                 num_samplers=1,
                 num_evaluators=1)

    method.run()


if __name__ == '__main__':
    main()
