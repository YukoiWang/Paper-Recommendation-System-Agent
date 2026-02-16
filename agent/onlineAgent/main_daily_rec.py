import arxiv
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Part 1: 数据模型 (Models & Mock Blackboard)
# ==========================================

@dataclass
class Paper:
    """定义论文的标准格式，与你项目中的 backend/models.py 对齐"""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    categories: List[str]
    url: str
    embedding: Any = None # 预留给向量

# --- 假装这是你的 Blackboard (State) ---
# 这是一个写死的 Dict，模拟从数据库或上游 Agent 传来的用户状态
MOCK_BLACKBOARD = {
    "user_id": "user_007",
    "user_profile": {
        # 用户感兴趣的具体技术关键词
        "interest_keywords": ["Self-Supervised Learning", "Vision Transformer"],
        # 用户关注的 arXiv 领域 (防止搜到物理/数学论文)
        "preferred_categories": ["cs.CV", "cs.LG"]
    },
    "rec_settings": {
        "time_window_days": 30,  # 只要最近一个月的
        "max_results": 10
    },
    # 预留接口：Agent 跑完后，把结果写到这里
    "daily_recommendations": [] 
}

# ==========================================
# Part 2: 工具函数 (Arxiv Tool)
# ==========================================

class ArxivTool:
    def __init__(self):
        self.client = arxiv.Client()
        # 初始化向量模型 (根据你的代码 snippet)
        logger.info("正在加载 SentenceTransformer 模型...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def fetch_papers_by_profile(self, keywords: List[str], categories: List[str], days: int = 30) -> List[Paper]:
        """
        根据用户画像构建 Query 并抓取
        """
        # 1. 构造时间窗口 (submittedDate)
        now = datetime.now()
        past_date = now - timedelta(days=days)
        
        start_str = past_date.strftime("%Y%m%d%H%M")
        end_str = now.strftime("%Y%m%d%H%M")
        
        # 2. 构造 Query
        # 逻辑：(关键词1 OR 关键词2) AND (分类1 OR 分类2) AND 时间范围
        # 例如: (Self-Supervised Learning OR Vision Transformer) AND (cat:cs.CV OR cat:cs.LG) ...
        
        keyword_part = " OR ".join([f'"{k}"' for k in keywords]) # 加引号以精确匹配词组
        category_part = " OR ".join([f"cat:{c}" for c in categories])
        
        final_query = f"({keyword_part}) AND ({category_part}) AND submittedDate:[{start_str} TO {end_str}]"
        
        logger.info(f"生成的 arXiv 查询语句: {final_query}")

        # 3. 执行搜索
        search = arxiv.Search(
            query=final_query,
            max_results=20, # 稍微多抓点，防止有些没有摘要
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        results = []
        for r in self.client.results(search):
            # 这里的 .replace 是为了防止换行符破坏打印格式或后续处理
            clean_title = r.title.replace('\n', ' ')
            clean_summary = r.summary.replace('\n', ' ')
            
            # 实例化 Paper 对象
            paper = Paper(
                paper_id=r.get_short_id(),
                title=clean_title,
                abstract=clean_summary,
                authors=[a.name for a in r.authors],
                published=str(r.published.date()),
                categories=r.categories,
                url=r.entry_id
            )
            
            # 【核心功能】直接在这里生成向量 embedding
            # 拼接 Title + Abstract 进行向量化
            text_to_embed = f"{clean_title}. {clean_summary}"
            paper.embedding = self.encoder.encode(text_to_embed)
            
            results.append(paper)
            
        return results

# ==========================================
# Part 3: 智能体逻辑 (Agent Node)
# ==========================================

def online_search_agent_node(blackboard: Dict[str, Any]):
    """
    模拟 LangGraph 节点：读取 Blackboard -> 执行任务 -> 写入 Blackboard
    """
    print("\n>>> 🕵️‍♂️ Agent 启动: 开始读取 Blackboard...")
    
    # 1. READ: 从 Blackboard 读取用户画像
    profile = blackboard.get("user_profile", {})
    settings = blackboard.get("rec_settings", {})
    
    keywords = profile.get("interest_keywords", [])
    categories = profile.get("preferred_categories", [])
    days = settings.get("time_window_days", 30)
    
    if not keywords:
        print("Blackboard 中没有用户兴趣关键词，跳过搜索。")
        return blackboard

    # 2. PROCESS: 调用工具
    tool = ArxivTool()
    print(f">>> 正在根据画像抓取: {keywords} (最近 {days} 天)")
    
    fetched_papers = tool.fetch_papers_by_profile(keywords, categories, days)
    
    # 3. WRITE: 将结果写回 Blackboard
    # 在真实系统中，这里通常是调用 VectorDB Service 存库
    # 这里我们模拟写入 State
    blackboard["daily_recommendations"] = fetched_papers
    
    print(f">>> ✅ 任务完成! 共抓取 {len(fetched_papers)} 篇论文并已写入 Blackboard。")
    return blackboard

# ==========================================
# Part 4: 执行 (Run)
# ==========================================

if __name__ == "__main__":
    # 1. 加载我们预定义的假 Blackboard
    current_state = MOCK_BLACKBOARD.copy()
    
    # 2. 运行 Agent
    updated_state = online_search_agent_node(current_state)
    
    # 3. 验证结果 (打印 Blackboard 中的数据)
    print("\n" + "="*50)
    print("【Blackboard 最终状态检查】")
    rec_list = updated_state["daily_recommendations"]
    
    if rec_list:
        for i, p in enumerate(rec_list[:3]): # 只打印前3个
            print(f"\n[推荐 {i+1}] {p.title}")
            print(f"发布日期: {p.published}")
            print(f"ID: {p.paper_id}")
            print(f"向量维度: {len(p.embedding) if p.embedding is not None else 'None'}")
            print(f"摘要(前50字): {p.abstract[:50]}...")
    else:
        print("未获取到推荐结果。")