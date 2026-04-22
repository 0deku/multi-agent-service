import os
from openai import OpenAI

class LLMClient:
    def __init__(self):
        self.qwen_key = os.getenv("QWEN_API_KEY", "")
        self.qwen_base = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.qwen_model = os.getenv("QWEN_MODEL", "qwen-plus")
        self.request_timeout = float(os.getenv("LLM_TIMEOUT", "20"))
        self.enabled = bool(self.qwen_key)

        self.qwen_client = OpenAI(api_key=self.qwen_key, base_url=self.qwen_base) if self.enabled else None

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.7):
        if not self.enabled:
            raise RuntimeError("QWEN_API_KEY is not configured")

        client = self.qwen_client
        model = self.qwen_model

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            timeout=self.request_timeout,
        )
        return resp.choices[0].message.content
