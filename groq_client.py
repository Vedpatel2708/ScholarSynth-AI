import httpx
import os
from typing import List, Dict, Any, AsyncGenerator
import json

class GroqChatCompletionClient:
    def __init__(self, model="mixtral-8x7b-32768"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = model
        self.model_info = {"function_calling": True, "vision": False}

    def _format_messages(self, messages):
        """Format AutoGen messages for Groq API"""
        formatted = []
        for m in messages:
            # Handle different message types from AutoGen
            if hasattr(m, 'source'):  # AutoGen message object
                role = "assistant" if m.source != "user" else "user"
                content = str(m.content) if hasattr(m, 'content') else str(m)
            else:
                # Handle dict-like messages
                role = getattr(m, 'role', 'user')
                content = getattr(m, 'content', str(m))

            # Clean up content
            if isinstance(content, (list, dict)):
                content = json.dumps(content)
            
            formatted.append({
                "role": role,
                "content": str(content)
            })
        return formatted

    async def create(self, messages, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            formatted_messages = self._format_messages(messages)
        except Exception as e:
            print(f"Message formatting error: {e}")
            # Fallback formatting
            formatted_messages = [{"role": "user", "content": str(messages)}]

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", 0.2),
            "stream": False
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
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
                print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                print(f"Error: {str(e)}")
                raise