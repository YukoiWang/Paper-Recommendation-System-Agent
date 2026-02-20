# LangGraph Agents

基于 LangGraph 的论文推荐多智能体工作流。

## 职责划分

| Agent | 文件 | 职责 |
|-------|------|------|
| **PlannerAgent** | `planner_agent.py` | 唯一决策者：意图分类、profile 检查、路由决定。输出结构化 plan |
| **RetrievalAgent** | `retrieval_agent.py` | 检索：query → 向量检索，返回候选 |
| **RecallAgent** | `recall_agent.py` | 召回：多路（向量 + 规则 + ItemCF）+ 在线/离线融合 |
| **OnlineSearchAgent** | `online_search_agent.py` | 在线搜索：ArXiv + Semantic Scholar |
| **RankAgent** | `rank_agent.py` | 排序：LLM 或 BGE 重排 |
| **PaperQAAgent** | `qa_agent.py` | 交互：纯 LLM 文本生成。三个入口：respond / ask_profile / handle_feedback |

## 关键设计

- **Planner 是唯一的决策点**：classify_intent 只在 Planner 中调用一次
- **QA 不做路由判断**：只负责"给我 papers + context，生成回答"
- **所有推荐路径都经过 Rank**：retrieval/recall → rank → respond
- **Planner 预留 LLM 升级接口**：plan() 返回结构化 plan，后续可替换为 LLM 规划

## LangGraph 流程

```
planner ──→ ask_profile ──→ END
        ├─→ handle_feedback ──→ END
        ├─→ online_search ──→ recall ──→ rank ──→ respond ──→ END
        ├─→ retrieval ──→ fuse ──→ rank ──→ respond ──→ END
        └─→ respond ──→ END  (general with existing context)
```

## 运行

```bash
# 需要 Python 3.10+ 和 langgraph
export PYTHONPATH=.:./agent
python -m langgraph_agents.run_workflow --api-key YOUR_KEY [--query "..."] [--daily]
```

## 依赖

- agent/ 下的模块（models, embedder, vector_store, recall_strategies, cold_start, rerank_prompt）
- `pip install langgraph>=0.2` (Python 3.10+)
- `pip install scikit-learn numpy openai`
