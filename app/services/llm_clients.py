import os
from openai import OpenAI

class LLMClient:
    def __init__(self):
        self.qwen_key = os.getenv("QWEN_API_KEY", "")
        self.qwen_base = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.qwen_model = os.getenv("QWEN_MODEL", "qwen-plus")

        self.qwen_client = OpenAI(api_key=self.qwen_key, base_url=self.qwen_base)

    def chat(self, system_prompt: str, user_prompt: str):
        client = self.qwen_client
        model = self.qwen_model

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content
