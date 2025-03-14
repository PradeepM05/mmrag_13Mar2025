import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMInterface:
    def __init__(self, api_key=None, model="gpt-4-turbo"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY in .env file or pass it directly.")
    
    def generate_response(self, prompt, system_message=None, temperature=0.7, max_tokens=1000):
        """Generate a response from the LLM."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(data)
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {e}"
