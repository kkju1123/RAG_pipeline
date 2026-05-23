# services/api/app/clients/ray_embed.py
import asyncio
import logging
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    logger.info("加载 BGE-M3 模型，首次约需 1-2 分钟...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    logger.info("BGE-M3 加载完成")
    return model

class RayEmbedClient:

    async def embed_query(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, _get_model)
        vector = await loop.run_in_executor(
            None,
            lambda: model.encode(text, normalize_embeddings=True).tolist()
        )
        return vector

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, _get_model)
        vectors = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, normalize_embeddings=True).tolist()
        )
        return vectors

embed_client = RayEmbedClient()