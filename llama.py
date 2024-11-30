from requests import post as rpost
from langchain_core.language_models import LLM

class LLaMa(LLM):
    def _call(self, prompt, stop = None, run_manager = None, **kwargs):
        return self.call_llama(prompt)
    
    def call_llama(self, prompt):
        headers = {"Content-Type":"application/json"}
        payload = {
            "model": "llama3.1:8b",
            "prompt" : prompt,
            "stream" : False
        }

        response = rpost(
            "http://localhost:11434/api/generate",
            headers=headers,
            json=payload
        )
        return response.json()["response"]
    
    @property
    def _llm_type(self):
        return "llama-3.1-8b"
    
    
