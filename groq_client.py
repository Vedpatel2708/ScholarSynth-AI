import httpx
import os
from typing import List, Dict, Any, AsyncGenerator
import json

class GroqChatCompletionClient:
    # Current supported models as of July 2025
    SUPPORTED_MODELS = {
        "llama-3.3-70b-versatile": {"context": "128K", "type": "production"},
        "llama-3.1-8b-instant": {"context": "128K", "type": "production"},
        "llama3-70b-8192": {"context": "8K", "type": "production"},
        "llama3-8b-8192": {"context": "8K", "type": "production"},
        "gemma2-9b-it": {"context": "8K", "type": "production"},
        "meta-llama/llama-4-maverick-17b-128e-instruct": {"context": "128K", "type": "preview"},
        "meta-llama/llama-4-scout-17b-16e-instruct": {"context": "128K", "type": "preview"},
    }
    
    def __init__(self, model="llama-3.3-70b-versatile", api_key=None):
        # Priority order for getting API key:
        # 1. api_key parameter
        # 2. Environment variable
        # 3. Streamlit session state (if available)
        
        self.api_key = None
        
        if api_key and api_key.strip():
            self.api_key = api_key.strip()
        elif os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY").strip():
            self.api_key = os.getenv("GROQ_API_KEY").strip()
        else:
            # Try to get from streamlit session state
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and 'groq_api_key' in st.session_state:
                    session_key = st.session_state.groq_api_key
                    if session_key and session_key.strip():
                        self.api_key = session_key.strip()
                        # Also set environment variable for consistency
                        os.environ["GROQ_API_KEY"] = self.api_key
            except ImportError:
                pass
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required. Please set it as environment variable or pass directly.")
        
        # Validate API key format
        if not self.api_key.startswith('gsk_'):
            print(f"Warning: API key doesn't start with 'gsk_'. Got: {self.api_key[:10]}...")
            
        # Validate model
        if model not in self.SUPPORTED_MODELS:
            print(f"Warning: Model '{model}' may not be supported. Supported models: {list(self.SUPPORTED_MODELS.keys())}")
            print(f"Using '{model}' anyway, but you may encounter errors.")
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = model
        self.model_info = {"function_calling": True, "vision": False}
        
        print(f"Initialized Groq client with model: {self.model}")
        print(f"API key format check: {'✅ Valid format' if self.api_key.startswith('gsk_') else '⚠️ Unusual format'}")

    @classmethod
    def get_recommended_model(cls, use_case="general"):
        """Get recommended model based on use case"""
        recommendations = {
            "general": "llama-3.3-70b-versatile",  # Best balance of performance and capability
            "fast": "llama-3.1-8b-instant",        # Fastest responses
            "research": "llama-3.3-70b-versatile", # Best for academic/research tasks
            "coding": "meta-llama/llama-4-maverick-17b-128e-instruct", # Good for code tasks
        }
        return recommendations.get(use_case, "llama-3.3-70b-versatile")

    def _format_messages(self, messages):
        """Format AutoGen messages for Groq API"""
        formatted = []
        for m in messages:
            # Handle different message types from AutoGen
            if hasattr(m, 'source'):  # AutoGen message object
                role = "assistant" if m.source != "user" else "user"
                content = str(m.content) if hasattr(m, 'content') else str(m)
            elif isinstance(m, dict):
                # Handle dict-like messages
                role = m.get('role', 'user')
                content = m.get('content', str(m))
            else:
                # Handle string messages
                role = 'user'
                content = str(m)

            # Clean up content
            if isinstance(content, (list, dict)):
                content = json.dumps(content)
            
            formatted.append({
                "role": role,
                "content": str(content)
            })
        return formatted

    async def create(self, messages, **kwargs):
        if not self.api_key:
            raise ValueError("API key is not set")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            formatted_messages = self._format_messages(messages)
        except Exception as e:
            print(f"Message formatting error: {e}")
            # Fallback formatting
            if isinstance(messages, str):
                formatted_messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], dict):
                    formatted_messages = messages
                else:
                    formatted_messages = [{"role": "user", "content": str(messages)}]
            else:
                formatted_messages = [{"role": "user", "content": str(messages)}]

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", None),
            "stream": False
        }

        async with httpx.AsyncClient(timeout=90.0) as client:  # Increased timeout
            try:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                # Return in expected AutoGen format
                return type('Response', (), {
                    'choices': [type('Choice', (), {
                        'message': type('Message', (), {
                            'content': result["choices"][0]["message"]["content"]
                        })()
                    })()]
                })()
                
            except httpx.HTTPStatusError as e:
                error_text = e.response.text
                print(f"HTTP Error: {e.response.status_code} - {error_text}")
                
                # Provide helpful error messages
                if e.response.status_code == 400:
                    if "model" in error_text.lower() and "decommissioned" in error_text.lower():
                        raise ValueError(f"Model '{self.model}' is no longer supported. Please use one of: {list(self.SUPPORTED_MODELS.keys())}")
                    elif "rate limit" in error_text.lower():
                        raise ValueError("Rate limit exceeded. Please wait and try again.")
                elif e.response.status_code == 401:
                    raise ValueError(f"Invalid API key. Please check your GROQ_API_KEY. Current key starts with: {self.api_key[:10] if self.api_key else 'None'}...")
                elif e.response.status_code == 429:
                    raise ValueError("Rate limit exceeded. Please wait and try again.")
                
                raise
            except Exception as e:
                print(f"Error: {str(e)}")
                raise
