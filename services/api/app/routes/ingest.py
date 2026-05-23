# services/api/app/routes/ingest.py
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from functools import lru_cache

router = APIRouter()
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embedder():
    logger.info("Loading embedding model...")
    return SentenceTransformer('BAAI/bge-small-en-v1.5')

def get_qdrant():
    return QdrantClient(host="localhost", port=6333)

@router.post("/text")
async def ingest_text_file(file: UploadFile = File(...)):
    """
    Upload a .txt file, chunk it, embed it, and store in Qdrant.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    try:
        content = await file.read()
        text = content.decode("utf-8")

        # 切分：按行，过滤空行
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not lines:
            raise HTTPException(status_code=400, detail="File is empty.")

        # Embedding
        embedder = get_embedder()
        vectors = embedder.encode(lines).tolist()

        # 写入 Qdrant
        client = get_qdrant()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": line,
                    "source": file.filename,
                    "metadata": {"filename": file.filename}
                }
            )
            for line, vec in zip(lines, vectors)
        ]
        client.upsert(collection_name="rag_collection", points=points)

        logger.info(f"Ingested {len(points)} chunks from {file.filename}")
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": len(points)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
