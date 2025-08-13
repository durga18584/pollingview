import requests

# --- Set your OpenRouter API key here ---
OPENROUTER_API_KEY = "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"

def call_llm(prompt, llm_config):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": llm_config["model"],  # e.g., "openai/gpt-3.5-turbo"
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
