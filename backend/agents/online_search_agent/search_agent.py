from typing import Dict, Any, List
# 【关键对接点】引入刚才写的 Tool
from daily_arxiv_tool import ArxivTool, Paper

# ==========================================
# 1. 定义 Mock Blackboard (假装这是你的 State)
# ==========================================
MOCK_BLACKBOARD = {
    # --- 来自上游 Planner 或 User Profile 的输入 ---
    "user_profile": {
        "user_id": "test_user_001",
        "interest_keywords": ["Large Language Models", "Chain of Thought"],
        "preferred_categories": ["cs.CL", "cs.AI"]
    },
    "rec_settings": {
        "time_window_days": 7,  # 只看最近一周
        "max_results": 5
    },
    
    # --- 预留给本节点的输出接口 ---
    "daily_recommendations": [], # 存具体论文对象
    "latest_paper_ids": [],      # 存 ID 列表供下游使用
    "messages": []               # 存日志/对话
}

# ==========================================
# 2. Agent Node 逻辑
# ==========================================
async def online_search_node(state: Dict[str, Any]):
    """
    LangGraph 节点：读取 State -> 调用 ArxivTool -> 更新 State
    """
    print("\n>>> 🕵️‍♂️ [Online Search Agent] 启动...")
    
    # --- STEP 1: READ (从黑板读数据) ---
    profile = state.get("user_profile", {})
    settings = state.get("rec_settings", {})
    
    keywords = profile.get("interest_keywords", [])
    categories = profile.get("preferred_categories", [])
    days = settings.get("time_window_days", 30)
    
    print(f"    - 读取到关键词: {keywords}")
    print(f"    - 设定时间窗口: 最近 {days} 天")

    # --- STEP 2: PROCESS (调用工具干活) ---
    # 实例化工具
    tool = ArxivTool() 
    
    print("    - 正在调用 ArxivTool 进行抓取...")
    fetched_papers: List[Paper] = tool.fetch_papers_by_profile(keywords, categories, days)
    
    if not fetched_papers:
        print("    - ⚠️ 未找到相关论文。")
        return {"messages": ["未找到新论文"]}

    # --- STEP 3: SAVE (模拟存入 VectorDB) ---
    # 在真实项目中，这里会调用 vector_service.upsert(fetched_papers)
    print(f"    - 📦 [Mock DB] 正在将 {len(fetched_papers)} 篇论文存入向量库...")
    # 打印第一篇的向量长度验证一下
    if fetched_papers:
        print(f"      (验证: 第一篇论文向量维度为 {len(fetched_papers[0].embedding)})")

    # --- STEP 4: WRITE (更新黑板) ---
    # 注意：LangGraph 中通常返回增量更新，但这里我们直接修改字典模拟
    
    # 将 Paper 对象转为字典以便查看 (可选，视 State 定义而定)
    # 这里我们直接存对象
    state["daily_recommendations"] = fetched_papers 
    state["latest_paper_ids"] = [p.paper_id for p in fetched_papers]
    state["messages"].append(f"成功更新 {len(fetched_papers)} 篇论文")
    
    print(">>> ✅ Agent 任务完成，黑板已更新。")
    return state

# ==========================================
# 3. 运行测试 (对接验证)
# ==========================================
if __name__ == "__main__":
    import asyncio
    
    # 1. 准备环境
    print("--- 开始测试 Online Search Node ---")
    current_state = MOCK_BLACKBOARD.copy()
    
    # 2. 运行节点
    # 因为 node 是 async 的，需要用 asyncio 运行
    final_state = asyncio.run(online_search_node(current_state))
    
    # 3. 验收成果
    print("\n" + "="*50)
    print("【最终黑板状态验收】")
    
    rec_list = final_state.get("daily_recommendations", [])
    ids_list = final_state.get("latest_paper_ids", [])
    
    print(f"1. 推荐列表长度: {len(rec_list)}")
    print(f"2. ID 列表: {ids_list}")
    
    if rec_list:
        p = rec_list[0]
        print(f"\n--- 抽查第一篇论文 ---")
        print(f"Title: {p.title}")
        print(f"Date : {p.published}")
        print(f"ID   : {p.paper_id}")
        print(f"Abs  : {p.abstract[:60]}...")