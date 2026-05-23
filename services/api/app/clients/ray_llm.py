# services/api/app/clients/ray_llm.py
import logging
import backoff
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from services.api.app.config import settings

logger = logging.getLogger(__name__)

class RayLLMClient:
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None

    async def start(self):
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
        )
        logger.info(f"DeepSeek Client initialized. Model: {settings.LLM_MODEL}")

    async def close(self):
        if self.client:
            await self.client.close()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def chat_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        if not self.client:
            raise RuntimeError("Client not initialized.")

        kwargs = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

llm_client = RayLLMClient()