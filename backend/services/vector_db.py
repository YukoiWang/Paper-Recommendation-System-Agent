"""
Vector Database Service
支持ChromaDB和Milvus
"""
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from backend.config import settings


class VectorDBService:
    """Vector Database Service for embedding storage and retrieval"""
    
    def __init__(self):
        self.db_type = settings.VECTOR_DB_TYPE
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize vector database client"""
        if self.db_type == "chroma":
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
                
                self.client = chromadb.PersistentClient(
                    path=settings.CHROMA_PERSIST_DIR,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                self.collection = self.client.get_or_create_collection(
                    name="papers",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("ChromaDB initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self.collection = None
        elif self.db_type == "milvus":
            try:
                from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
                
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT
                )
                # TODO: Create collection if not exists
                logger.info("Milvus initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Milvus: {e}")
    
    async def add_papers(
        self,
        papers: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> bool:
        """
        添加论文向量
        
        Args:
            papers: 论文列表
            embeddings: 对应的向量列表
            
        Returns:
            是否成功
        """
        try:
            if self.db_type == "chroma":
                if self.collection is None:
                    err = RuntimeError(
                        "ChromaDB 未初始化（可能因 sqlite3 版本过旧）。"
                        "请安装: pip install pysqlite3-binary"
                    )
                    logger.error(str(err))
                    raise err
                ids = [p.get("paper_id") or str(i) for i, p in enumerate(papers)]
                documents = [p.get("abstract", "") or p.get("title", "") for p in papers]
                metadatas = [
                    {
                        "title": p.get("title", ""),
                        "venue": p.get("venue", ""),
                        "year": str(p.get("year", "")),
                    }
                    for p in papers
                ]
                
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(papers)} papers to ChromaDB")
                return True
            else:
                # TODO: Implement Milvus add
                logger.warning("Milvus add not implemented yet")
                return False
        except Exception as e:
            logger.error(f"Failed to add papers to vector DB: {e}")
            return False
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回top-k结果
            filters: 过滤条件
            exclude_ids: 排除的paper_id列表
            
        Returns:
            搜索结果列表，包含paper_id和score
        """
        try:
            if self.db_type == "chroma":
                where = None
                if filters:
                    where = {}
                    if "venue" in filters:
                        where["venue"] = filters["venue"]
                    if "year" in filters:
                        where["year"] = str(filters["year"])
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k * 2 if exclude_ids else top_k,  # 取更多结果以便过滤
                    where=where
                )
                
                # 处理结果
                papers = []
                if results["ids"] and len(results["ids"]) > 0:
                    for i in range(len(results["ids"][0])):
                        paper_id = results["ids"][0][i]
                        if exclude_ids and paper_id in exclude_ids:
                            continue
                        
                        papers.append({
                            "paper_id": paper_id,
                            "score": 1.0 - results["distances"][0][i] if "distances" in results else 0.0,
                            "metadata": results["metadatas"][0][i] if "metadatas" in results else {}
                        })
                        
                        if len(papers) >= top_k:
                            break
                
                return papers
            else:
                # TODO: Implement Milvus search
                logger.warning("Milvus search not implemented yet")
                return []
        except Exception as e:
            logger.error(f"Vector similarity search error: {e}")
            return []
    
    async def get_embedding(self, paper_id: str) -> Optional[List[float]]:
        """
        获取指定论文的向量
        
        Args:
            paper_id: 论文ID
            
        Returns:
            向量或None
        """
        try:
            if self.db_type == "chroma":
                results = self.collection.get(ids=[paper_id], include=["embeddings"])
                embs = results.get("embeddings")
                if embs is not None and len(embs) > 0:
                    e = embs[0]
                    return e.tolist() if hasattr(e, "tolist") else list(e)
            return None
        except Exception as e:
            logger.error(f"Failed to get embedding for {paper_id}: {e}")
            return None
