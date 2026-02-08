"""
Ranking Model Service
传统排序模型服务（LightGBM/DeepFM）
"""
from typing import List, Dict, Any
import numpy as np
import pickle
from pathlib import Path
from loguru import logger

from backend.config import settings


class RankingModelService:
    """Ranking Model Service for traditional ML ranking"""
    
    def __init__(self):
        self.model = None
        self.model_path = settings.RANKING_MODEL_PATH
        self._load_model()
    
    def _load_model(self):
        """加载排序模型"""
        try:
            if Path(self.model_path).exists():
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded ranking model from {self.model_path}")
            else:
                logger.warning(f"Ranking model not found at {self.model_path}, will use default scoring")
        except Exception as e:
            logger.error(f"Failed to load ranking model: {e}")
    
    async def rank(
        self,
        candidates: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        使用排序模型对候选进行排序
        
        Args:
            candidates: 候选论文列表
            user_profile: 用户画像
            
        Returns:
            排序后的论文列表（包含model_score）
        """
        if not candidates:
            return []
        
        # 提取特征
        features = self._extract_features(candidates, user_profile)
        
        if self.model:
            try:
                # 使用模型预测分数
                scores = self.model.predict(features)
                for i, candidate in enumerate(candidates):
                    candidate["model_score"] = float(scores[i])
            except Exception as e:
                logger.error(f"Model prediction error: {e}, using default scoring")
                self._default_scoring(candidates, user_profile)
        else:
            # 使用默认评分
            self._default_scoring(candidates, user_profile)
        
        # 按分数排序
        ranked = sorted(candidates, key=lambda x: x.get("model_score", 0.0), reverse=True)
        
        return ranked
    
    def _extract_features(self, candidates: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> np.ndarray:
        """
        提取排序特征
        
        Args:
            candidates: 候选论文列表
            user_profile: 用户画像
            
        Returns:
            特征矩阵
        """
        features = []
        
        for candidate in candidates:
            feature_vector = [
                candidate.get("recall_score", 0.0),  # 召回分数
                candidate.get("vector_score", 0.0),  # 向量相似度
                candidate.get("citation_count", 0) / 100.0,  # 归一化引用数
                candidate.get("year", 2020) / 2024.0,  # 归一化年份
                1.0 if candidate.get("venue") in ["NeurIPS", "ICML", "ICLR"] else 0.0,  # 顶级会议
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _default_scoring(self, candidates: List[Dict[str, Any]], user_profile: Dict[str, Any]):
        """
        默认评分函数（当模型不可用时）
        
        Args:
            candidates: 候选论文列表
            user_profile: 用户画像
        """
        for candidate in candidates:
            score = 0.0
            
            # 召回分数
            score += candidate.get("recall_score", 0.0) * 0.4
            
            # 向量相似度
            score += candidate.get("vector_score", 0.0) * 0.3
            
            # 引用数（归一化）
            citations = candidate.get("citation_count", 0)
            score += min(citations / 100.0, 1.0) * 0.2
            
            # 新颖性（年份越新越好）
            year = candidate.get("year", 2020)
            recency = max(0, (year - 2020) / 4.0)
            score += recency * 0.1
            
            candidate["model_score"] = score
