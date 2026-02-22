# Agent Architecture: Roles and Boundaries

本文档描述论文推荐系统中各 Agent 的分工与职责边界，以及 Blackboard 的设计。

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PlannerAgent (编排层)                               │
│  - 读取 Blackboard                                                           │
│  - 根据场景/意图调度各 Agent                                                   │
│  - 将结果写回 Blackboard                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
          │                    │                    │                    │
          ▼                    ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  QA Agent    │    │ Retrieval    │    │  Rank Agent  │    │ Online Search│
│  (对话/回答)  │    │ Agent (召回)  │    │  (精排)      │    │ Agent (线上)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
          │                    │                    │                    │
          └────────────────────┴────────────────────┴────────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │        Blackboard (黑板)       │
                        │  共享读写、单源真相              │
                        └───────────────────────────────┘
```

---

## 2. Blackboard（黑板）

Blackboard 是所有 Agent 共享的状态存储，用于输入输出和协同。

### 2.1 核心字段

| 字段 | 类型 | 职责 |
|------|------|------|
| `user_id` | str | 用户 ID |
| `user_profile` | UserProfile | 用户画像（兴趣、偏好、特殊要求） |
| `user_query` | str | 当前用户提问 |
| `qa_intent` | str | QA 判断的意图（recommend/qa/compare/summarize/explain/general） |
| `is_daily_rec` | bool | 是否为日常推荐场景 |
| `history` | List[BlackboardMessage] | 历史对话记录 |
| `cited_papers` | Dict[str, Paper] | 当前会话引用论文 [1]->Paper |
| `recall_result` | List[Paper] | 离线召回结果 |
| `online_search_result` | List[Paper] | 线上 ArXiv 搜索结果 |
| `online_offline_fusion_ratio` | float | 线上/线下融合比例 (0~1) |
| `fused_candidates` | List[Paper] | 融合后候选列表 |
| `ranked_papers` | List[Paper] | 精排结果 |
| `final_papers` | List[Paper] | 最终 Top-K 论文 |
| `top_k` | int | 保留论文数量 |
| `user_feedback` | str | 用户对推荐/回答的反馈 |
| `needs_profile_clarification` | bool | 是否需要 QA 主动询问兴趣 |
| `history_contains_papers` | bool | 历史对话是否已含论文上下文 |
| `prefer_latest_papers` | bool | 是否偏好最新论文 |
| `profile_updated_from_feedback` | bool | 是否已根据反馈更新画像 |

---

## 3. 各 Agent 职责

### 3.1 PlannerAgent（编排 Agent）

**职责**：读取 Blackboard，根据场景和意图调度其他 Agent，并将结果写回 Blackboard。

**输入**：Blackboard  
**输出**：修改 Blackboard，返回执行状态

**场景与流程**：

1. **日常推荐 (is_daily=True)**
   - 画像完整 → 调用 Online Agent + Retrieval Agent → 融合 → Rank Agent → QA 推送给用户
   - 画像不全 → 设置 `needs_profile_clarification`，调用 QA 主动询问

2. **用户主动提问 (is_daily=False)**
   - 根据 QA 的意图分类调度：
     - **recommend/qa/explain/compare/summarize**：调用 Retrieval + Rank，QA 基于 Blackboard 中的论文生成回答
     - **recommend + 画像不全**：QA 先主动询问，补全后再检索
     - **recommend + 要最新论文**：调用 Online Agent，提高融合比例，Retrieval 开启 `prefer_recent`
     - **general**：若历史含论文上下文则 QA 直接答；否则先 Retrieval 再 QA

3. **用户反馈 (is_feedback=True)**
   - 将 `user_query` 视为反馈，写入 `user_feedback`，由 QA 处理（更新画像或追问澄清）

**边界**：
- 不直接做检索、排序、生成回答
- 只做调度和 Blackboard 读写

---

### 3.2 QA Agent（对话 Agent）

**职责**：意图分类、主动询问、基于论文生成回答、处理用户反馈并更新画像。

**输入**：Blackboard  
**输出**：修改 Blackboard 的 `history`、`cited_papers`、`user_profile`，返回状态

**核心能力**：

1. **意图分类** `classify_intent(query)`
   - recommend / qa / compare / summarize / explain / general

2. **主动询问**
   - 当 `needs_profile_clarification` 或画像不足时，生成追问兴趣的问题

3. **基于论文回答**
   - 从 Blackboard 的 `final_papers` 或 `ranked_papers` 读取
   - 结合历史、按意图选择 prompt 生成回答

4. **历史是否含论文**
   - `_check_history_contains_papers()`：判断历史是否已有论文上下文
   - 若无，返回 `need_retrieval` 让 Planner 调用 Retrieval

5. **反馈处理**
   - 反馈明确 → 更新 `user_profile`（`special_requirements`、`interest_text`）
   - 反馈模糊 → 主动追问澄清

**边界**：
- 不做检索、排序
- 论文来源仅为 Blackboard，不直接调用 Retrieval

---

### 3.3 Retrieval Agent（召回 Agent）

**职责**：基于用户画像和查询做多路召回、合并，支持 Top-K 和 recency 偏好。

**输入**：UserProfile、query（可选）、`top_k`、`prefer_recent`  
**输出**：List[Paper]

**能力**：
- `retrieve_for_user(user, new_papers, top_k_override, prefer_recent)`：画像召回
- `retrieve_by_query(query, user, top_k, prefer_recent)`：查询召回
- 从 Blackboard 读取 `top_k` 控制返回数量
- `prefer_recent=True` 时增加时间衰减，偏重新论文

**边界**：
- 不做精排
- 不处理对话和反馈

---

### 3.4 Rank Agent（精排 Agent）

**职责**：对候选论文做精排（LLM 或 BGE Reranker）。

**输入**：papers、query、user、`top_k`  
**输出**：重排后的 List[Paper]

**能力**：
- 模式：`llm` 或 `bge_reranker`
- 从 Blackboard 读取 `top_k` 决定保留数量

**边界**：
- 不做召回
- 不处理对话

---

### 3.5 Online Search Agent（线上搜索 Agent）

**职责**：从 ArXiv API 抓取最新论文。

**输入**：Blackboard（user_profile、user_query）  
**输出**：写入 `online_search_result`、`online_search_ids`

**能力**：
- 根据画像/查询生成 ArXiv 搜索条件
- 按提交时间倒序返回论文

**边界**：
- 不参与离线索引和召回
- 结果写入 Blackboard，由 Planner 做融合

---

## 4. 典型流程

### 4.1 日常推荐（画像完整）

```
Planner.run(is_daily=True)
  → Online Agent.run(bb)           # 抓取最新论文
  → Retrieval.retrieve_for_user()  # 召回（含 top_k, prefer_recent）
  → _merge_online_offline()        # 按 fusion_ratio 融合
  → Rank.rerank(top_k=bb.top_k)    # 精排
  → bb.final_papers = ranked[:top_k]
  → QA.process_turn(bb)            # 生成推荐文案并推送
```

### 4.2 日常推荐（画像不全）

```
Planner.run(is_daily=True)
  → bb.needs_profile_clarification = True
  → QA.process_turn(bb)            # 主动询问兴趣
  → 等待用户补充 → 再次 run_daily_rec
```

### 4.3 用户提问「推荐最新论文」

```
Planner.run(is_daily=False)
  → classify_intent("latest papers") = recommend
  → bb.prefer_latest_papers = True
  → bb.online_offline_fusion_ratio = 0.7
  → Online Agent.run(bb)
  → Retrieval.retrieve_by_query(prefer_recent=True)
  → 融合 → Rank → QA 回答
```

### 4.4 用户反馈「我想要更多 diffusion 相关」

```
Planner.run(is_feedback=True)  # user_query 作为 user_feedback
  → QA.process_turn(bb)
  → _handle_feedback: 更新 profile.special_requirements
  → 可再次推荐
```

---

## 5. 使用示例

```python
from blackboard import Blackboard
from models import UserProfile
from planner_agent import PlannerAgent
# ... 初始化 retrieval, rank, qa, online agents ...

planner = PlannerAgent(retrieval, rank, qa, online, default_top_k=10)

# 日常推荐
bb = Blackboard(
    user_id="user1",
    user_profile=UserProfile(user_id="user1", interest_text="large language models"),
    top_k=10,
)
planner.run(bb, is_daily=True)
response = bb.history[-1].content  # 最后一条助理回复

# 用户提问
bb.user_query = "Recommend papers about vision transformers"
planner.run(bb)
response = bb.history[-1].content

# 用户反馈
bb.user_query = "I prefer more recent ones"
planner.run(bb, is_feedback=True)
```

---

## 6. 扩展点

- **Blackboard**：可增加 `session_id`、`request_id`、`metadata` 等字段
- **Planner**：可接入更多 Agent（如知识图谱、引用分析）
- **QA**：可增加多轮澄清、偏好记录等逻辑
