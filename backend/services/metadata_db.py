"""
Metadata Database Service
PostgreSQL数据库服务，存储论文元数据
"""
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Integer, Text, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from loguru import logger

from backend.config import settings

Base = declarative_base()


class Paper(Base):
    """论文元数据模型"""
    __tablename__ = "papers"
    
    paper_id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    authors = Column(JSON)  # List of author names
    abstract = Column(Text)
    venue = Column(String, index=True)
    year = Column(Integer)
    publish_time = Column(DateTime)
    tags = Column(JSON)  # List of tags
    citations = Column(Integer, default=0)
    url = Column(String)
    arxiv_id = Column(String, index=True)
    semantic_scholar_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MetadataDBService:
    """Metadata Database Service"""
    
    def __init__(self):
        self.engine = create_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._create_tables()
    
    def _create_tables(self):
        """创建数据库表"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Metadata database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取论文
        
        Args:
            paper_id: 论文ID
            
        Returns:
            论文字典或None
        """
        session = self.get_session()
        try:
            paper = session.query(Paper).filter(Paper.paper_id == paper_id).first()
            if paper:
                return self._paper_to_dict(paper)
            return None
        except Exception as e:
            logger.error(f"Failed to get paper {paper_id}: {e}")
            return None
        finally:
            session.close()
    
    async def get_papers_by_ids(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        批量获取论文
        
        Args:
            paper_ids: 论文ID列表
            
        Returns:
            论文列表
        """
        session = self.get_session()
        try:
            papers = session.query(Paper).filter(Paper.paper_id.in_(paper_ids)).all()
            return [self._paper_to_dict(p) for p in papers]
        except Exception as e:
            logger.error(f"Failed to get papers: {e}")
            return []
        finally:
            session.close()
    
    async def upsert_paper(self, paper_data: Dict[str, Any]) -> bool:
        """
        插入或更新论文
        
        Args:
            paper_data: 论文数据字典
            
        Returns:
            是否成功
        """
        session = self.get_session()
        try:
            paper = session.query(Paper).filter(Paper.paper_id == paper_data["paper_id"]).first()
            
            if paper:
                # Update existing
                for key, value in paper_data.items():
                    if hasattr(paper, key):
                        setattr(paper, key, value)
                paper.updated_at = datetime.utcnow()
            else:
                # Create new
                paper = Paper(**paper_data)
                session.add(paper)
            
            session.commit()
            logger.info(f"Upserted paper {paper_data.get('paper_id')}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to upsert paper: {e}")
            return False
        finally:
            session.close()
    
    async def upsert_papers(self, papers_data: List[Dict[str, Any]]) -> int:
        """
        批量插入或更新论文
        
        Args:
            papers_data: 论文数据列表
            
        Returns:
            成功插入/更新的数量
        """
        count = 0
        for paper_data in papers_data:
            if await self.upsert_paper(paper_data):
                count += 1
        return count
    
    def _paper_to_dict(self, paper: Paper) -> Dict[str, Any]:
        """将ORM对象转换为字典"""
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors or [],
            "abstract": paper.abstract or "",
            "venue": paper.venue,
            "year": paper.year,
            "publish_time": paper.publish_time.isoformat() if paper.publish_time else None,
            "tags": paper.tags or [],
            "citations": paper.citations or 0,
            "url": paper.url,
            "arxiv_id": paper.arxiv_id,
            "semantic_scholar_id": paper.semantic_scholar_id,
        }
