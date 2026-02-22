#!/usr/bin/env python3
"""测试排序模式一：LLM 重排。支持 Qwen/DashScope、DeepSeek、OpenAI 兼容接口。"""
import os

from dotenv import load_dotenv
# 从项目根目录加载 .env（支持在 agent/ 或项目根目录运行）
_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_env)

import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.models import Paper, UserProfile
from rank_agent import RankAgent
from data_loader import load_builtin_corpus, load_from_lance_hf


def _llm_config_from_env():
    """从环境变量读取 LLM 配置：支持 Qwen/DashScope、DeepSeek、OpenAI。"""
    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        os.getenv("QWEN_API_BASE")
        or os.getenv("DASHSCOPE_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
    )
    if not base_url:
        base_url = (
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
            if (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"))
            else "https://api.deepseek.com"
        )
    # DashScope 兼容模式常用模型：qwen-plus, qwen-max, qwen-turbo
    # 如果未指定，DashScope 默认用 qwen-plus（更稳定）
    model = os.getenv("QWEN_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        if "dashscope" in base_url or "aliyuncs" in base_url:
            model = "qwen-plus"  # DashScope 兼容模式默认用 qwen-plus
        else:
            model = "deepseek-chat"
    return api_key, base_url.rstrip("/"), model


def main():
    import argparse

    ap = argparse.ArgumentParser(description="测试排序模式一：LLM 语义重排（带不相干候选）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，保证候选顺序可复现")
    ap.add_argument("--num-relevant", type=int, default=6, help="候选中相关论文数量（按类别 cs.CL/cs.LG 粗筛）")
    ap.add_argument("--num-irrelevant", type=int, default=6, help="候选中不相干论文数量（非 cs.CL/cs.LG）")
    ap.add_argument("--top-k", type=int, default=8, help="输出 top-k")
    ap.add_argument("--query", type=str, default="推荐和 LLM 高效训练、压缩相关的论文", help="用户 query")
    ap.add_argument("--data-source", type=str, default="real", choices=["builtin", "real"],
                    help="数据源：builtin=内置示例，real=真实ArXiv论文（Lance数据集）")
    ap.add_argument("--max-papers", type=int, default=100, help="从真实数据集加载的最大论文数（仅 real 模式）")
    ap.add_argument("--lance-path", type=str, default=None,
                    help="本地 Lance 数据集路径（可选，默认尝试本地或 HuggingFace）")
    args = ap.parse_args()

    random.seed(args.seed)

    # 加载论文数据
    if args.data_source == "real":
        print("正在加载真实 ArXiv 论文数据...")
        try:
            papers = load_from_lance_hf(
                lance_path=args.lance_path,
                max_papers=args.max_papers,
                filter_categories=["cs.CL", "cs.LG", "cs.CV", "cs.AI", "cs.IR"],  # 包含相关和不相干类别
                prefer_recent=True,
            )
            if len(papers) < 10:
                print(f"警告：只加载到 {len(papers)} 篇论文，可能数据源不可用，回退到内置示例")
                papers = load_builtin_corpus()
            else:
                print(f"成功加载 {len(papers)} 篇真实论文")
        except Exception as e:
            print(f"加载真实数据失败: {e}")
            print("回退到内置示例数据")
            papers = load_builtin_corpus()
    else:
        print("使用内置示例数据")
        papers = load_builtin_corpus()

    # 构造“混合候选”：用关键词+类别匹配区分相关/不相干
    # 相关：标题/摘要包含 LLM、language model、training、compression、efficient、fine-tuning 等，或类别为 cs.CL
    # 不相干：视觉（vision、image）、强化学习（RL、reinforcement）、图（graph、molecular）、联邦学习（federated）等，或类别为 cs.CV（且不含相关关键词）
    query_keywords = ["llm", "language model", "training", "compression", "efficient", "fine-tuning", "pretraining", "transformer", "attention"]
    irrelevant_keywords = ["vision", "image", "visual", "reinforcement", "rl", "graph", "molecular", "federated", "federated learning", "robot", "robotics"]
    
    relevant_pool = []
    irrelevant_pool = []
    for p in papers:
        title_lower = (p.title or "").lower()
        abstract_lower = (p.abstract or "").lower()
        text_lower = title_lower + " " + abstract_lower
        categories = [c.lower() for c in (p.categories or [])]
        
        # 检查是否包含不相干关键词
        has_irrelevant_kw = any(kw in text_lower for kw in irrelevant_keywords)
        # 检查是否包含相关关键词
        has_relevant_kw = any(kw in text_lower for kw in query_keywords)
        # 检查类别
        is_cl_category = "cs.cl" in categories
        is_cv_category = "cs.cv" in categories
        
        # 分类逻辑
        if has_irrelevant_kw and not has_relevant_kw:
            # 有明显不相干关键词且无相关关键词 -> 不相干
            irrelevant_pool.append(p)
        elif has_relevant_kw or is_cl_category:
            # 有相关关键词或 cs.CL 类别 -> 相关
            relevant_pool.append(p)
        elif is_cv_category:
            # cs.CV 且无相关关键词 -> 不相干（视觉方向）
            irrelevant_pool.append(p)
        else:
            # 其他情况：按类别判断，cs.LG/cs.AI 算相关，其他算不相干
            if "cs.lg" in categories or "cs.ai" in categories:
                relevant_pool.append(p)
            else:
                irrelevant_pool.append(p)
    
    print(f"\n论文池统计:")
    print(f"  总论文数: {len(papers)}")
    print(f"  相关候选池: {len(relevant_pool)} 篇")
    print(f"  不相干候选池: {len(irrelevant_pool)} 篇")
    
    if len(relevant_pool) < 2 or len(irrelevant_pool) < 2:
        print("\n错误：相关/不相干论文数量不足，无法构造测试候选。")
        print(f"  relevant_pool={len(relevant_pool)}, irrelevant_pool={len(irrelevant_pool)}")
        if len(relevant_pool) > 0:
            print(f"  相关论文示例: {[p.title[:50] + '...' for p in relevant_pool[:3]]}")
        if len(irrelevant_pool) > 0:
            print(f"  不相干论文示例: {[p.title[:50] + '...' for p in irrelevant_pool[:3]]}")
        return

    num_rel = min(args.num_relevant, len(relevant_pool))
    num_irrel = min(args.num_irrelevant, len(irrelevant_pool))
    candidates = random.sample(relevant_pool, k=num_rel) + random.sample(irrelevant_pool, k=num_irrel)
    random.shuffle(candidates)

    user = UserProfile(
        user_id="test_user",
        interest_text="large language model efficient training compression",
        preferred_categories=["cs.CL", "cs.LG"],
        followed_authors=["Yichen Wang", "Sophia Lee"],
        liked_items=["2401.00001", "2401.00004"],
    )
    query = args.query

    api_key, base_url, model = _llm_config_from_env()
    if not api_key:
        print(
            "请设置 API Key 环境变量之一: "
            "DASHSCOPE_API_KEY / QWEN_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY"
        )
        sys.exit(1)
    print(f"使用 base_url={base_url}, model={model}")

    agent = RankAgent(mode="llm", api_key=api_key, base_url=base_url, model=model)
    print("\n用户画像（profile）:")
    print(f"  user_id={user.user_id}")
    print(f"  interest_text={user.interest_text}")
    print(f"  preferred_categories={user.preferred_categories}")
    print(f"  followed_authors={user.followed_authors}")
    print(f"  liked_items={user.liked_items}\n")

    print(f"\n候选论文列表（共 {len(candidates)} 篇，已打乱顺序）:")
    print("排序前（候选顺序，包含不相干论文）:")
    for i, p in enumerate(candidates):
        cats = ",".join(p.categories or [])
        pid = p.paper_id[:20] if len(p.paper_id) > 20 else p.paper_id
        print(f"  {i+1:2d}. [{pid}] [{cats}] {p.title[:70]}...")
    print()

    reranked = agent.rerank(candidates, query, user, top_k=args.top_k)
    print(f"排序后（LLM 重排 top-{len(reranked)}）:")
    for i, p in enumerate(reranked):
        cats = ",".join(p.categories or [])
        pid = p.paper_id[:20] if len(p.paper_id) > 20 else p.paper_id
        print(f"  {i+1:2d}. [score={p.score:.1f}] [{pid}] [{cats}] {p.title[:70]}...")

    # 输出理由（用户可读）
    meta = getattr(agent, "last_llm_reasons", {}) or {}
    reasons_map = meta.get("reasons_map", {}) or {}
    summary = meta.get("summary", "") or ""
    if summary:
        print(f"\n重排摘要: {summary}")
    if reasons_map:
        print("\n每篇论文的理由（简要）：")
        # reasons_map 的 key 是 0-based index（对应 candidates 的索引）
        for rank_pos, paper in enumerate(reranked, start=1):
            # 找到该 paper 在原 candidates 里的位置（按 paper_id 匹配）
            orig_idx = next((i for i, c in enumerate(candidates) if c.paper_id == paper.paper_id), None)
            if orig_idx is None:
                continue
            rs = reasons_map.get(orig_idx, [])
            if not rs:
                continue
            print(f"  - Top{rank_pos}: {paper.title[:70]}...")
            for r in rs:
                print(f"      * {r}")
    print("排序模式一（LLM）测试完成。")


if __name__ == "__main__":
    main()
