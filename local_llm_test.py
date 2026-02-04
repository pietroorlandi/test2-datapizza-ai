import os
from datapizza.clients.openai_like import OpenAILikeClient
from dotenv import load_dotenv

load_dotenv()

# Create client for Ollama
client = OpenAILikeClient(
    api_key="",  # Ollama doesn't require an API key
    model="gemma2:2b",  # Use any model you've pulled with Ollama
    system_prompt="You are a helpful assistant.",
    base_url="http://localhost:11434/v1",  # Default Ollama API endpoint
)

# Simple query
response = client.invoke("What is the capital of France?")
print(response.content)
