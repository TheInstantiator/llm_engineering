# 🚀 LLM Engineering Fork

Personal notes and setup guide for the [LLM Engineering](https://github.com/ed-donner/llm_engineering) course.

## 🛠️ Repository Setup

1. **Clone the original repo:**

    ```bash
    git clone https://github.com/ed-donner/llm_engineering
    ```

2. **Fork the repo:**
    <https://github.com/TheInstantiator/llm_engineering.git>

3. **Goal:** Setup git to push changes to the fork on `llm_main` and pull new changes from the original.

## 📦 Tools & Environment

### Essential Software

- **IDE:** 🆚 VS Code (with Python & Jupyter extensions)
- **Language:** 🐍 Python
- **Package Manager:** ⚡ `uv` Package Manager
- **Notebooks:** 📓 Jupyter

### 🦙 Ollama Setup

- Download **Ollama** (Note: May need to install on `C:` drive if on Windows).
- Visit [ollama.com/search](https://ollama.com/search) to find models.

**Target Models:**

- `gpt-os` (GPT)
- `deepseek-r1` (DeepSeek)
- `llama3.2` (Meta)
- `gemma` (Google)
- `grok-2` (xAI - check availability)

## 🔑 Configuration (.env)

Setup your `.env` file with the following keys:

- `GROK_API_KEY`
- `GEMINI_API_KEY`
- `OPENAI_KEY`

> **Note:** Ensure you set the environment correctly when running Jupyter notebooks (upper right kernel selector). No code billing info added for OpenAI or DeepSeek yet.

## 💻 Code Snippets

### OpenAI Compatible Clients (Grok & Gemini)

```python
from openai import OpenAI
import os

# Grok Setup
GROK_BASE_URL = "https://api.x.ai/v1"
grok = OpenAI(base_url=GROK_BASE_URL, api_key=os.getenv("GROK_API_KEY"))

# Gemini Setup
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=os.getenv("GEMINI_API_KEY"))
```

## 🧠 Prompt Engineering Notes

### System Prompt (`system_prompt`)

- **Job Description:** Acts like giving the AI its role (e.g., NFL injury report analyst).
- **Task:** Defines exactly what info to look for.
- **Format:** Specifies output format (e.g., markdown tables).
- *Consistency:* Stays the same across requests.

### User Prompt (`user_prompt_prefix`)

- **Task:** Specific task for the current input.
- **Constraint:** Tells it to ignore non-relevant content.
- **Action:** Specifies what to do with the following content.

### Message Structure

Combines prompts for the API:

1. **System Message** (First)
2. **User Message** (Second)

## 🔄 Git Workflow Cheat Sheet

**Fetch latest from upstream:**

```bash
git fetch upstream
```

**Ensure you're on `llm_main`:**

```bash
git checkout llm_main
```

**Merge upstream changes:**

```bash
git merge upstream/main
# OR
git rebase upstream/main
```

**Push to your fork:**

```bash
git push origin llm-main
```
