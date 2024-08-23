openai_model_list = [
    'gpt-3.5-turbo', 'gpt-4o', 'gpt-4'
]
claude_model_list = [
    'claude-3-5-sonnet-20240620', 'claude-3-opus-20240229',
    'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'
]
gemini_model_list = [
    'gemini-1.5-pro-001'
]
ollama_model_list = [
    'llama3.1:8b-instruct-fp16', 'llama3.1:8b-instruct-q8_0', 'llama3.1:70b-instruct-q8_0',
    'mistral-nemo:12b-instruct-2407-q6_K',
]

per_minute_token_limit = {
    "claude-3-5-sonnet-20240620": 160000,
    "claude-3-opus-20240229": 80000,
    "claude-3-sonnet-20240229": 160000,
    "claude-3-haiku-20240307": 200000,
    "gpt-4o": 800000,
    "llama3.1:8b-instruct-fp16": 1000000000,
    "llama3.1:8b-instruct-q8_0": 1000000000,
    "llama3.1:70b-instruct-q8_0": 1000000000,
    "mistral-nemo:12b-instruct-2407-q6_K": 1000000000,
}
