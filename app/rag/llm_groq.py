import os
from groq import Groq


class GroqLLM:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in environment variables.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return chat_completion.choices[0].message.content
