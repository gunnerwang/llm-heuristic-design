<h1 align="center">
LLM for Multi-AGV Scheduling Problem
</h1>

# Environment Setup Guide

## Setting up API Keys

For security reasons, API keys should be stored in environment variables rather than hardcoded in the source code.

### Step 1: Install Dependencies

Make sure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Create a .env File

Create a `.env` file in the project root directory (`llm-agv-scheduling/.env`) with the following content:

```env
# API Configuration - Choose one and set the corresponding key
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Alternative providers (uncomment and set if using)
# QWEN_API_KEY=your_qwen_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here

# Default LLM Configuration
DEFAULT_LLM_HOST=api.deepseek.com
DEFAULT_LLM_MODEL=deepseek-chat
DEFAULT_LLM_TIMEOUT=300
```

### Step 3: Replace Placeholder Values

Replace `your_deepseek_api_key_here` with your actual API key from DeepSeek.

### Step 4: Alternative API Providers

If you want to use a different API provider:

#### For Qwen:
- Uncomment the `QWEN_API_KEY` line
- Set `DEFAULT_LLM_HOST=dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation`
- Set `DEFAULT_LLM_MODEL=qwen-plus`

#### For OpenAI:
- Uncomment the `OPENAI_API_KEY` line  
- Set `DEFAULT_LLM_HOST=api.openai-proxy.org`
- Set `DEFAULT_LLM_MODEL=gpt-4o`
- Set `DEFAULT_LLM_TIMEOUT=120`

### Security Notes

- The `.env` file is automatically ignored by git (see `.gitignore`)
- Never commit API keys to version control
- Keep your API keys secure and don't share them

### Usage

Once configured, simply run your scripts as normal:

```bash
python example/agv_drone_scheduling/run_eoh.py
```

The script will automatically load the environment variables from the `.env` file. 