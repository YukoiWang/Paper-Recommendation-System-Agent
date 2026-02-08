"""
Embedding Service
文本向量化服务
"""
from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

from backend.config import settings


class EmbeddingService:
    """Embedding Service for text vectorization"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载embedding模型"""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded embedding model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a default model
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                logger.warning("Using fallback embedding model")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
    
    async def embed_query(self, text: str) -> List[float]:
        """
        向量化查询文本
        
        Args:
            text: 查询文本
            
        Returns:
            向量列表
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量向量化文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            向量列表
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise
    
    async def embed_paper(self, paper: dict) -> List[float]:
        """
        向量化论文（使用title + abstract）
        
        Args:
            paper: 论文字典
            
        Returns:
            向量列表
        """
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        text = f"{title} {abstract}".strip()
        
        return await self.embed_query(text)
