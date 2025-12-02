Setup:
Clone the repo from:
https://github.com/ed-donner/llm_engineering
Fork the repo:
https://github.com/TheInstantiator/llm_engineering.git
Setup git go push changes to the fork on llm_main and pull new changes from the original

Download Ollama
Seems like I have to put Ollama on C drive
ollama.com/search shows all models
Gpt-os (gpt)
Deepseek-r1 (dem chinese)
Llama3.2 (meta)
Gemma (google)
Don't see a grok, but grok tells me grok-2
Not installed
VsCode
Extensions
Python
Jupyter
UV Package Manager
Setup .env with GROK_API_KEY, GEMINI_API_KEY, and OPENAI_KEY
I didn't put any billing info to OPENAI or DEEPSEEK

Notes:
When running jupyter notebooks in the page in the upper right you have to set environment.

OpenAI calls for other APIs
GROK_BASE_URL = "https://api.x.ai/v1"
grok = OpenAI(base_url=GROK_BASE_URL, api_key=grok_api_key)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=gemini_api_key)

## System Prompts 
# System Prompt (system_prompt):

Acts like giving the AI its job description
Tells it to be an NFL injury report analyst
Defines exactly what information to look for (name, injury, status)
Specifies how to format the output (markdown tables)
This stays consistent across all injury report requests

# User Prompt (user_prompt_prefix):

Gives the specific task for this particular website
Tells it to ignore non-injury content
Specifies what to do with the website content that follows
Gets combined with the actual website content
Messages List:

Combines both prompts in the format OpenAI expects
Always has system message first, then user message

Git Flow:
# fetch latest from upstream
git fetch upstream

# ensure you're on llm_main
git checkout llm_main

# merge upstream/main
git merge upstream/main
# or rebase:
# git rebase upstream/main

# push merged changes to your fork
git push origin llm-main
