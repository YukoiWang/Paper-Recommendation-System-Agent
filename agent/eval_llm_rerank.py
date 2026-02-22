#!/usr/bin/env python3
"""评估 LLM 重排效果：在测试集上对比 baseline vs LLM 重排。

指标：
- Pairwise Accuracy: positive 排在 negative 前面的比例
- MRR: Mean Reciprocal Rank（点击/正例的平均倒数排名）
- Hit Rate@K: positive 出现在 top-K 的比例
"""
import os
from dotenv import load_dotenv
# 从项目根目录加载 .env（支持在 agent/ 或项目根目录运行）
_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_env)

import argparse
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.models import Paper, UserProfile
from rank_agent import RankAgent
from test_rank_llm import _llm_config_from_env


def load_jsonl(filepath):
    """加载 JSONL 或 JSON 文件"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not data:
            f.seek(0)
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                pass
    return data


def sample_to_papers(sample):
    """将一条样本转换为 Paper 对象列表"""
    papers = []
    
    # current paper
    current = sample.get("current_paper", {})
    papers.append(Paper(
        paper_id=f"current_{sample.get('query', '')[:20]}",
        title=current.get("title", ""),
        abstract=current.get("abstract", ""),
        authors=[],
        categories=[],
    ))
    
    # positive paper
    positive = sample.get("positive_paper", {})
    papers.append(Paper(
        paper_id="positive",
        title=positive.get("title", ""),
        abstract=positive.get("abstract", ""),
        authors=[],
        categories=[],
    ))
    
    # negative paper
    negative = sample.get("negative_paper", {})
    papers.append(Paper(
        paper_id="negative",
        title=negative.get("title", ""),
        abstract=negative.get("abstract", ""),
        authors=[],
        categories=[],
    ))
    
    return papers


def baseline_rank(papers):
    """Baseline：随机排序（模拟召回后的原始顺序）"""
    import random
    shuffled = papers.copy()
    random.shuffle(shuffled)
    return shuffled


def evaluate_pairwise_accuracy(ranked_papers):
    """计算 pairwise accuracy：positive 是否排在 negative 前面"""
    pos_idx = None
    neg_idx = None
    for i, p in enumerate(ranked_papers):
        if p.paper_id == "positive":
            pos_idx = i
        elif p.paper_id == "negative":
            neg_idx = i
    
    if pos_idx is None or neg_idx is None:
        return None
    
    return 1.0 if pos_idx < neg_idx else 0.0


def evaluate_mrr(ranked_papers):
    """计算 MRR：positive 的倒数排名"""
    for i, p in enumerate(ranked_papers):
        if p.paper_id == "positive":
            return 1.0 / (i + 1)
    return 0.0


def evaluate_hit_rate_k(ranked_papers, k=5):
    """计算 Hit Rate@K：positive 是否在 top-K"""
    for i, p in enumerate(ranked_papers[:k]):
        if p.paper_id == "positive":
            return 1.0
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="评估 LLM 重排效果")
    parser.add_argument("test_file", type=str, help="测试集文件路径（JSONL 或 JSON）")
    parser.add_argument("--max-samples", type=int, default=None, help="最大评估样本数（默认全部）")
    parser.add_argument("--top-k", type=int, default=3, help="评估 top-K（默认 3，因为只有 3 个候选）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    # 加载测试集
    print(f"正在加载测试集: {args.test_file}")
    test_data = load_jsonl(args.test_file)
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    print(f"加载了 {len(test_data)} 条测试样本")
    
    if len(test_data) == 0:
        print("错误：测试集为空")
        return
    
    # 初始化 LLM agent
    api_key, base_url, model = _llm_config_from_env()
    if not api_key:
        print("错误：请设置 API Key 环境变量")
        sys.exit(1)
    
    agent = RankAgent(mode="llm", api_key=api_key, base_url=base_url, model=model)
    
    # 评估指标
    baseline_pairwise_acc = []
    llm_pairwise_acc = []
    baseline_mrr = []
    llm_mrr = []
    baseline_hr_k = []
    llm_hr_k = []
    
    print(f"\n开始评估（共 {len(test_data)} 条样本）...")
    
    for i, sample in enumerate(test_data):
        if (i + 1) % 10 == 0:
            print(f"  处理进度: {i+1}/{len(test_data)}")
        
        # 构造候选列表
        papers = sample_to_papers(sample)
        if len(papers) < 3:
            continue
        
        # 构造 user profile 和 query
        profile_data = sample.get("user_profile", {})
        user = UserProfile(
            user_id=f"eval_user_{i}",
            interest_text=", ".join(profile_data.get("interests", [])),
            preferred_categories=[],
        )
        query = sample.get("query", "")
        
        # Baseline 排序
        baseline_ranked = baseline_rank(papers)
        baseline_pairwise_acc.append(evaluate_pairwise_accuracy(baseline_ranked))
        baseline_mrr.append(evaluate_mrr(baseline_ranked))
        baseline_hr_k.append(evaluate_hit_rate_k(baseline_ranked, k=args.top_k))
        
        # LLM 重排
        try:
            llm_ranked = agent.rerank(papers, query, user, top_k=args.top_k)
            llm_pairwise_acc.append(evaluate_pairwise_accuracy(llm_ranked))
            llm_mrr.append(evaluate_mrr(llm_ranked))
            llm_hr_k.append(evaluate_hit_rate_k(llm_ranked, k=args.top_k))
        except Exception as e:
            print(f"  样本 {i+1} LLM 重排失败: {e}")
            # LLM 失败时，用 baseline 结果填充
            llm_pairwise_acc.append(baseline_pairwise_acc[-1])
            llm_mrr.append(baseline_mrr[-1])
            llm_hr_k.append(baseline_hr_k[-1])
    
    # 计算平均指标
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    baseline_pairwise_avg = avg(baseline_pairwise_acc)
    llm_pairwise_avg = avg(llm_pairwise_acc)
    baseline_mrr_avg = avg(baseline_mrr)
    llm_mrr_avg = avg(llm_mrr)
    baseline_hr_avg = avg(baseline_hr_k)
    llm_hr_avg = avg(llm_hr_k)
    
    # 输出结果
    print(f"\n{'='*60}")
    print("评估结果对比")
    print(f"{'='*60}")
    print(f"\nPairwise Accuracy (positive 排在 negative 前面的比例):")
    print(f"  Baseline: {baseline_pairwise_avg:.4f}")
    print(f"  LLM 重排: {llm_pairwise_avg:.4f}")
    print(f"  提升: {llm_pairwise_avg - baseline_pairwise_avg:+.4f} ({((llm_pairwise_avg / baseline_pairwise_avg - 1) * 100):+.2f}%)")
    
    print(f"\nMRR (Mean Reciprocal Rank):")
    print(f"  Baseline: {baseline_mrr_avg:.4f}")
    print(f"  LLM 重排: {llm_mrr_avg:.4f}")
    print(f"  提升: {llm_mrr_avg - baseline_mrr_avg:+.4f} ({((llm_mrr_avg / baseline_mrr_avg - 1) * 100):+.2f}%)")
    
    print(f"\nHit Rate@{args.top_k}:")
    print(f"  Baseline: {baseline_hr_avg:.4f}")
    print(f"  LLM 重排: {llm_hr_avg:.4f}")
    print(f"  提升: {llm_hr_avg - baseline_hr_avg:+.4f} ({((llm_hr_avg / baseline_hr_avg - 1) * 100):+.2f}%)")
    
    print(f"\n有效样本数: {len(baseline_pairwise_acc)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
