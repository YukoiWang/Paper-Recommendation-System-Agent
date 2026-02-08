"""
Load Papers into Database and Vector DB
"""
import asyncio
import json
from pathlib import Path
from loguru import logger

from backend.services.metadata_db import MetadataDBService
from backend.services.vector_db import VectorDBService
from backend.services.embedding import EmbeddingService


async def load_papers_from_file(file_path: str):
    """
    从JSON文件加载论文到数据库和向量库
    
    Args:
        file_path: JSON文件路径
    """
    metadata_db = MetadataDBService()
    vector_db = VectorDBService()
    embedding_service = EmbeddingService()
    
    # 读取论文数据
    with open(file_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    logger.info(f"Loading {len(papers)} papers...")
    
    # 批量处理
    batch_size = 100
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]
        
        # 1. 生成向量
        texts = []
        for paper in batch:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            texts.append(f"{title} {abstract}".strip())
        
        embeddings = await embedding_service.embed_texts(texts, batch_size=batch_size)
        
        # 2. 存储到元数据库
        await metadata_db.upsert_papers(batch)
        
        # 3. 存储到向量库
        await vector_db.add_papers(batch, embeddings)
        
        logger.info(f"Processed {min(i+batch_size, len(papers))}/{len(papers)} papers")
    
    logger.info("All papers loaded successfully!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_papers.py <papers.json>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    asyncio.run(load_papers_from_file(file_path))
