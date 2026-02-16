import arxiv
import logging
from datetime import datetime, timedelta
from typing import List, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 数据模型 (建议后续移到 backend/models.py) ---
@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    categories: List[str]
    url: str
    embedding: Any = None

class ArxivTool:
    def __init__(self):
        # 初始化 arXiv 客户端
        self.client = arxiv.Client()
        
        # 初始化向量模型 (为了避免每次调用都加载，建议在单例或服务中加载)
        # 这里为了演示方便，保留在 init 中，但加了 print 提示
        print("📥 [ArxivTool] 正在加载 SentenceTransformer 模型 (首次运行会慢)...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def fetch_papers_by_profile(self, keywords: List[str], categories: List[str], days: int = 30) -> List[Paper]:
        """
        核心功能：根据画像关键词 + 分类 + 时间窗口，抓取论文并向量化
        """
        # 1. 构造时间窗口
        now = datetime.now()
        past_date = now - timedelta(days=days)
        start_str = past_date.strftime("%Y%m%d%H%M")
        end_str = now.strftime("%Y%m%d%H%M")
        
        # 2. 构造复杂 Query
        # 逻辑：(关键词 OR ...) AND (分类 OR ...) AND 时间
        if not keywords:
            logger.warning("未提供关键词，跳过搜索")
            return []
            
        keyword_part = " OR ".join([f'"{k}"' for k in keywords])
        category_part = " OR ".join([f"cat:{c}" for c in categories]) if categories else "cat:cs.CV OR cat:cs.AI"
        
        final_query = f"({keyword_part}) AND ({category_part}) AND submittedDate:[{start_str} TO {end_str}]"
        logger.info(f"🔎 执行查询: {final_query}")

        # 3. 调用 API
        search = arxiv.Search(
            query=final_query,
            max_results=10,  # 限制数量，方便测试
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        results = []
        try:
            for r in self.client.results(search):
                clean_title = r.title.replace('\n', ' ')
                clean_summary = r.summary.replace('\n', ' ')
                
                # 4. 生成向量 (Embedding)
                text_to_embed = f"{clean_title}. {clean_summary}"
                embedding_vector = self.encoder.encode(text_to_embed).tolist() # 转为 list 以便 JSON 序列化
                
                # 5. 封装对象
                paper = Paper(
                    paper_id=r.get_short_id(),
                    title=clean_title,
                    abstract=clean_summary,
                    authors=[a.name for a in r.authors],
                    published=str(r.published.date()),
                    categories=r.categories,
                    url=r.entry_id,
                    embedding=embedding_vector
                )
                results.append(paper)
        except Exception as e:
            logger.error(f"ArXiv API 调用失败: {e}")
            
        return results