# How to Add Your Groq API Key

## Step 1: Get Your Free Groq API Key

1. Go to https://console.groq.com
2. Sign up (free)
3. Create an API key
4. Copy the key (starts with `gsk_...`)

## Step 2: Add Key to the Code

Open `llm_task_decomposer.py` and find this line near the top (around line 28):

```python
DEFAULT_GROQ_API_KEY = "gsk_your_api_key_here"  # TODO: Replace with your actual Groq key
```

Replace `"gsk_your_api_key_here"` with your actual key:

```python
DEFAULT_GROQ_API_KEY = "gsk_abc123xyz..."  # Your real Groq key
```

## Step 3: That's It!

The system will now use Groq's Llama 3.1 70B model for intelligent task decomposition!

## Usage Example

```python
from llm_task_decomposer import LLMTaskDecomposer

# Uses the hardcoded key automatically
decomposer = LLMTaskDecomposer()

result = decomposer.decompose_goal(
    "Prepare for Data Structures exam in 3 days. Focus on Graphs and DP.",
    deadline_days=3
)

print(f"Generated {len(result['sub_tasks'])} tasks")
for task in result['sub_tasks']:
    print(f"- {task['task_name']}")
```

## What You Get with Groq API:

✅ **Intelligent task breakdown** - LLM understands context  
✅ **Better task names** - More specific and relevant  
✅ **Smarter dependencies** - Logical prerequisite chains  
✅ **Adaptive priorities** - Based on actual content  

## Fallback Mode (No API Key):

If you don't add a key or the API fails, the system automatically uses the rule-based fallback which still works great!

Just set:
```python
DEFAULT_GROQ_API_KEY = None
```

---

**That's all you need!** Simple hardcoded key for your prototype. 🚀
