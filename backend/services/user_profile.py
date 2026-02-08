"""
User Profile Service
用户画像管理服务
"""
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from loguru import logger

from backend.config import settings
from backend.services.llm import LLMService
from backend.services.embedding import EmbeddingService

Base = declarative_base()


class UserProfile(Base):
    """用户画像模型"""
    __tablename__ = "user_profiles"
    
    user_id = Column(String, primary_key=True, index=True)
    interests = Column(JSON)  # List of interest tags
    profile_summary = Column(Text)  # LLM生成的画像摘要
    recent_reads = Column(JSON)  # List of recent paper IDs
    exposure_history = Column(JSON)  # List of exposed paper IDs
    read_history = Column(JSON)  # List of read paper IDs
    preferences = Column(JSON)  # User preferences dict
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserProfileService:
    """User Profile Service"""
    
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.llm_service = LLMService()
        self.embedding_service = EmbeddingService()
        self._create_tables()
    
    def _create_tables(self):
        """创建数据库表"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("User profile tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create user profile tables: {e}")
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    async def get_profile(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户画像
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户画像字典
        """
        session = self.get_session()
        try:
            profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            if profile:
                return self._profile_to_dict(profile)
            else:
                # 创建默认画像
                return await self.create_default_profile(user_id)
        except Exception as e:
            logger.error(f"Failed to get profile for {user_id}: {e}")
            return await self.create_default_profile(user_id)
        finally:
            session.close()
    
    async def create_default_profile(self, user_id: str) -> Dict[str, Any]:
        """
        创建默认用户画像
        
        Args:
            user_id: 用户ID
            
        Returns:
            默认画像字典
        """
        default_profile = {
            "user_id": user_id,
            "interests": [],
            "profile_summary": "新用户，画像待完善",
            "recent_reads": [],
            "exposure_history": [],
            "read_history": [],
            "preferences": {}
        }
        
        session = self.get_session()
        try:
            profile = UserProfile(**default_profile)
            session.add(profile)
            session.commit()
            logger.info(f"Created default profile for {user_id}")
            return default_profile
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create default profile: {e}")
            return default_profile
        finally:
            session.close()
    
    async def add_exposure_history(self, user_id: str, paper_ids: List[str]):
        """
        添加曝光历史
        
        Args:
            user_id: 用户ID
            paper_ids: 论文ID列表
        """
        session = self.get_session()
        try:
            profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            if profile:
                current_exposure = profile.exposure_history or []
                # 去重并添加
                new_exposure = list(set(current_exposure + paper_ids))
                profile.exposure_history = new_exposure[-1000:]  # 保留最近1000条
                profile.updated_at = datetime.utcnow()
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add exposure history: {e}")
        finally:
            session.close()
    
    async def add_read_history(self, user_id: str, paper_ids: List[str]):
        """
        添加阅读历史
        
        Args:
            user_id: 用户ID
            paper_ids: 论文ID列表
        """
        session = self.get_session()
        try:
            profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            if profile:
                current_reads = profile.read_history or []
                new_reads = list(set(current_reads + paper_ids))
                profile.read_history = new_reads[-500:]  # 保留最近500条
                profile.recent_reads = new_reads[-10:]  # 最近10篇
                profile.updated_at = datetime.utcnow()
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add read history: {e}")
        finally:
            session.close()
    
    async def update_interests_from_papers(self, user_id: str, paper_ids: List[str]):
        """
        从论文中提取并更新用户兴趣
        
        Args:
            user_id: 用户ID
            paper_ids: 论文ID列表
        """
        # TODO: 从论文的tags、title、abstract中提取兴趣
        # 可以使用LLM或关键词提取
        logger.info(f"Updating interests for {user_id} from {len(paper_ids)} papers")
    
    async def generate_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """
        生成/更新用户画像摘要
        
        Args:
            user_id: 用户ID
            
        Returns:
            更新后的用户画像
        """
        profile = await self.get_profile(user_id)
        
        # 使用LLM生成摘要
        try:
            summary = await self.llm_service.generate_profile_summary(profile)
            
            # 更新数据库
            session = self.get_session()
            try:
                db_profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
                if db_profile:
                    db_profile.profile_summary = summary
                    db_profile.updated_at = datetime.utcnow()
                    session.commit()
                    profile["profile_summary"] = summary
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update profile summary: {e}")
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Failed to generate profile summary: {e}")
        
        return profile
    
    def _profile_to_dict(self, profile: UserProfile) -> Dict[str, Any]:
        """将ORM对象转换为字典"""
        return {
            "user_id": profile.user_id,
            "interests": profile.interests or [],
            "profile_summary": profile.profile_summary or "",
            "recent_reads": profile.recent_reads or [],
            "exposure_history": profile.exposure_history or [],
            "read_history": profile.read_history or [],
            "preferences": profile.preferences or {}
        }
