# 学术论文问答与推荐系统：多智能体技术实现说明

> 状态：**目标架构 / 实现规格**（相对当前 `langgraph_agents/` 流水线的升级方案）  
> 编排框架：**LangGraph**  
> 适用范围：有 Query 的学术问答与推荐、无 Query 的日常推荐、多轮对话与冷启动

**分节速查：** 意图三级瀑布 → [§3.2](#32-任务类型三级瀑布规则--向量近邻--大模型兜底)；数据库与记忆 / 上下文 → [§11](#11-数据存储记忆管理与上下文管理)；Harness 与自我优化 → [§14](#14-harness-与自我优化迭代)；Researcher loop → [§8](#8-researcher-agent内部-loop)；Evidence pack → [§9](#9-evidence-pack)。

本文描述的是「Planner + Recommend + Researcher + Writer + Critic + 共享 Tools + 线下 Harness」这一套系统应如何实现。当前仓库仍是以 Planner 三路路由为主的 DAG（检索 / 在线搜索 / 召回 / 重排 / QA），本文中的 Researcher 工具循环、结构化意图、Critic 门控与 Harness 为待落地能力。实现时应在现有 `langgraph_agents` 上演进，而不是另起一套产品语义。

---

## 1. 问题与设计原则

### 1.1 要同时解决的两类需求

| 模式 | 用户状态 | 系统目标 | 控制流 |
|------|----------|----------|--------|
| No-Query / Daily | 无明确指令，浏览或每日推送 | 个性化、多样、低延迟的论文列表 | 短 DAG：多路召回 → 融合 → 重排 |
| Query-Search | 有自然语言请求 | 按意图给出列表、解释、对比或短事实，且可引用 | 研究环：选工具 → 攒证据 → 生成 → 质检 |

两类需求的优化目标互相冲突：推荐要稳、要快、可走 ItemCF；问答要探索、要换源、要 grounding。因此 **不能** 用一个大 ReAct 同时拿着搜、荐、写、评全部工具。

### 1.2 原则

1. **决策者才叫 Agent**。召回、Arxiv、BGE 重排是 Tool/Service，不升格为 Agent。
2. **推荐能力收口、策略分叉**。`daily` 与有 Query 的 `recommend` 共用同一套召回–重排实现；`explain` / `compare` 不跑 Recommend Agent，最多调用 `rerank` 工具。
3. **分类管分流，结构管施工**。任务类型用「规则 → 问法向量近邻 → 大模型兜底」三级瀑布；槽位与成功标准用规则 + 校验后的抽取填充，与类型判定分开。
4. **规则可以否决模型**。空召回换源、compare 禁止合成单 query、本地已够禁止误联网，均由程序强制，不靠模型自觉。
5. **发出前质检与线下复盘分离**。Critic 挡这一次坏回答；Harness 用轨迹改策略并做回归，不在用户请求里改 prompt。
6. **线上只记账，线下才改系统**。生产路径追加 episode log；prompt / 阈值 / 级联规则的变更走提案 → 重放 → 人审 → 发版。

### 1.3 非目标

- 不做成 STORM / AutoSurvey 式长文或自动写综述系统（`survey` 最多分节概述）。
- 不追求 PaperQA2 级全文 page-level 引用作为第一期必达（Critic 先对 pack 内 chunk 做 grounding）。
- 不在线上做强化学习或自动 merge prompt。
- 不为 chitchat 拉满全部 Agent。

---

## 2. 总体架构

```text
                         用户请求 / 每日触发
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │     Planner Agent      │
                    │ 意图 · 槽位校验 · 工单  │
                    └────────────┬───────────┘
           ┌─────────────────────┼──────────────────────┐
           ▼                     ▼                      ▼
   Recommend Agent        Researcher Agent            Writer
   (daily / recommend     (explain / compare /      (澄清、闲聊、
    / 反馈后重推)          survey / factoid)         或等上游结果)
           │                     │                      │
           └──────────┬──────────┘                      │
                      ▼                                 │
              共享 Tools 层                              │
         search_local / search_arxiv / search_s2        │
         recall_multi / rerank / gather_evidence        │
         read_fulltext                                  │
                      │                                 │
                      ▼                                 ▼
                 evidence pack / ranked_papers  →  Writer
                                                       │
                                                       ▼
                                                 Critic Agent
                                            pass / rewrite / more_evidence
                                                       │
                                                       ▼
                                              用户可见回复 + episode log
                                                       │
                                        每 N 条或每日 ──► Harness（线下）
```

### 2.1 五个决策者

| Agent | 优化目标 | 有权做什么 | 无权做什么 |
|-------|----------|------------|------------|
| Planner | 任务分流与预算 | 填工单、选下一个 Agent、stall 时重规划一次 | 搜论文、写长文、改重排模型 |
| Recommend | 列表相关性、多样、时延 | 选召回通道、是否联网、是否跳过重排 | 写叙事长文、`gather_evidence` |
| Researcher | 证据覆盖与质量 | 在 playbook 允许的工具中循环选择 | 对用户说话、跑 ItemCF 日常推荐 |
| Writer | 可读且不超出证据 | 按意图模板生成；缺槽时只问一句 | 擅自新检索、发明未入 pack 的论文 |
| Critic | 可发出性 | `pass` / `rewrite` / `more_evidence` | 自己 search；改用户画像 |

### 2.2 框架选型（不考虑迁移成本时的结论）

**运行时编排使用 LangGraph**（`StateGraph` + 子图 + 条件边 + checkpointer）。

| 需求 | LangGraph 如何覆盖 |
|------|-------------------|
| Recommend 必须是短 DAG | 子图：固定边 `recall → fuse → rerank` |
| Researcher 必须是有预算的环 | 节点 `researcher` 内 Tool 循环，或 `ToolNode` + 回边，用 `max_steps` 封顶 |
| Critic 打回 Writer / Researcher | 条件边，而不是群聊 |
| 澄清打断 | `interrupt` / 本回合结束把问题交给 Writer |
| 轨迹可重放 | checkpointer + 自写 episode JSONL |
| 多模型（DeepSeek + 本地 BGE） | 与厂商 SDK 解耦 |

不采用 CrewAI / AutoGen 群聊作为主循环：推荐路径会被「开会」化，强制级联难以写成可测规则。Magentic-One 的 Task Ledger / Progress Ledger **只借鉴字段设计**，落在 LangGraph state 上。

配套：

- 检索与 chunk：继续用现有向量库，或把 `gather_evidence` 的切分实现成独立 tool（实现上可参考 LlamaIndex，但编排权仍在 LangGraph）。
- Tool / Critic 输出：Pydantic 模型约束。
- 观测：LangSmith 或 Langfuse 可选；Harness 最低可用是 JSONL。

---

## 3. 意图体系

意图表示 **用户要的成品形态**，不是「检不检索」。检索与否由意图 + 槽位 + 是否已有上下文共同决定。

### 3.1 类别（10 类）

| `intent` | 典型输入 | `next_agent` | 成品 |
|----------|----------|--------------|------|
| `chitchat` | 你好、谢谢 | Writer | 短回复 |
| `meta` | 你能做什么 | Writer | 能力说明 |
| `feedback` | 太老了、不要综述 | Planner 写画像，必要时 Recommend | 确认 + 可选新列表 |
| `daily` | 无 query / 每日任务 | Recommend | 个性化列表 |
| `recommend` | 推荐几篇 RAG | Recommend | 带理由的列表 |
| `explain` | 注意力怎么演进的 | Researcher | 叙事 + `[N]` |
| `compare` | Transformer vs Mamba | Researcher | 对照结构，两边都有引用 |
| `survey` | 做 LLM Agent 综述 | Researcher | 分节概述（非穷尽） |
| `factoid` | 这篇用了什么数据集 | Researcher 或仅 Writer | 短答 + 必须落地引用 |
| `followup` | 第二篇展开讲 | 默认 Writer；说「再找类似」则改判 | 绑定已引用论文 |

**不是意图、而是槽位的：** `need_sota`、`need_fulltext`、`expertise`、年份区间。  
**不是意图的：** `NEED_CLARIFY`。澄清 = 意图已知但 `missing_slots` 非空。

### 3.2 任务类型：三级瀑布（规则 → 向量近邻 → 大模型兜底）

只判定 §3.1 的 `intent` 枚举，**不在这一步抽实体、不写 playbook**。能前级解决的不调用后级，控制延迟与费用。现有 `_check_trivial` 迁入第 ① 级并扩展。

```text
当前问句 + 是否已有 cited_papers
  → ① 规则命中 → 直接输出 intent（confidence=1.0）
  → ② 与意图原型库做 embedding 相似度；最高分 ≥ τ 且与第二名分差 ≥ δ → 采纳
  → ③ 否则大模型结构化输出 intent（仅难例）
```

**① 规则：只吃「特别明显」的关键词 / 句式，高精确、低召回。**  
没有明显指向时 **不要硬判**，交给 ②/③。禁止用弱特征（仅出现「论文」「RAG」）在 ① 拍板。现有 `_check_trivial` 迁入并扩展。命中即停，`confidence=1.0`。

明显指向示例（可配置词表，中英并列；需同时满足「有 cited」的另注）：

| 意图 | 明显信号（命中才走 ①） |
|------|-------------------------|
| `chitchat` | 你好/谢谢/hi/thanks；去掉标点后极短且无主题词 |
| `meta` | 你是谁/你能做什么/what can you do |
| `feedback` | 太老了/不要综述/换一批/不喜欢这些 |
| `daily` | 无 query，或定时任务入口（不是用户句子里的词） |
| `followup` | **已有 cited** 且 第二篇/展开讲/那篇/more about [the first] |
| `compare` | vs / versus / 对比 / 比较 / 区别 / 和…有什么不同（两边像方法名时更稳） |
| `recommend` | 推荐几篇/找几篇/recommend papers on（有主题或有画像才定死；光「推荐论文」只标类型，主题进 `missing_slots`） |
| `explain` | 讲讲/介绍一下/怎么演进/what is / how does（且无 vs/对比） |
| `survey` | 综述/文献综述/survey of / 全面回顾 |
| `factoid` | 谁一作/哪年发的/用了什么数据集/这篇的方法（且常能指到 cited 或明确篇名） |

「再找几篇类似的」虽含「找几篇」，但是否新检索取决于 cited，**不算单一关键词意图**，不在 ① 定死，交给 ②/③。  
「RAG 论文」这种只有主题、没有推荐/讲解/对比词的，① 不应输出 `recommend` 或 `explain`。

**① 与 ② 的分工：** ① 认 *成品形态词*（推荐、讲讲、对比、综述）；② 认 *整句像哪类问法*（主题不同时近邻才可靠）。形态词一明显，不必等向量。

**② 问法向量近邻。** 与论文向量库分离，使用独立集合 `intent_prototypes`：

- 每类 10～30 条已标注问句（种子可用 `planner_routing_testcases.json`，按 10 类重标后入库）。
- 编码器与论文检索一致（如 `BAAI/bge-base-en-v1.5`），避免两套空间。
- 新 query 编码后与各类原型算余弦：类分 = 该类 TopK 均值或最大值；取 argmax。
- 采纳条件：`score ≥ τ` 且 `score_1 - score_2 ≥ δ`（τ、δ 在路由金标上网格搜索）。不满足则进入 ③。

**为何文档里常举 recommend/explain、compare/explain：** 不是只有这两对会错，而是它们满足「主题词几乎相同、下游图完全不同」。embedding 编码的是 *关于什么*，几乎不编码 *用户要列表还是要讲解还是要对照表*。其余类别要么 ① 就能切开，要么错了下游差异没那么大。完整易混关系如下，金标必须覆盖，不满足分差就进 ③，不要加码「只禁止这两对」。

| 易混对 | 为何向量容易撞 | 优先怎么切开 | 切错的代价 |
|--------|----------------|--------------|------------|
| `recommend` vs `explain` | 都含 RAG/Transformer 等主题词，差在「推荐几篇」vs「讲讲/演进」 | ① 列表诉求词 vs 讲解词；否则 ③ | 列表编排 vs 证据循环，体验完全反了 |
| `compare` vs `explain` | 「A 和 B」既像对比也像一起介绍 | ① `vs`/对比/区别；实体数 ≥2 可作软特征；否则 ③ | 一次混搜 vs 分边检索 |
| `recommend` vs `survey` | 都像「多给几篇」 | ① 综述/survey/全面回顾 vs 推荐几篇 | 并行子查询 vs 短列表 |
| `explain` vs `survey` | 都是「讲这个方向」 | ① 综述/全面/梳理文献 vs 讲讲某机制 | 覆盖面与长度不同 |
| `explain` vs `factoid` | 都像提问 | ① 短问具体槽（谁、哪年、什么数据集、这篇）且可指向 cited | 长叙事 vs 一句+必须引用 |
| `followup` vs `recommend` / `explain` | 「再找类似」语义像新检索 | **不要靠 ②**：① 看 cited + 指代词 vs 「再找几篇」 | 误检索或该搜不搜 |
| `feedback` vs `recommend` | 「不要综述、换一批」像新推荐 | ① 否定/约束词先改画像 | 没记下偏好 |
| `chitchat` / `meta` vs 任意研究类 | 「你好」与短 query | **不要靠 ②**：① 问候表、过短 | 浪费检索 |
| `daily` vs `recommend` | 都是列表 | **不要靠 ②**：有无 query / 是否定时任务 | 通道选错（ItemCF vs query 检索） |

② 真正擅长的是：**主题不同、任务类型也不同** 的句子（「ViT 论文」vs「你好」在规则之后很少进 ②；进 ② 的应是中等自信的「找扩散模型 2024」→ `recommend` 或 `explain` 由分差决定，分差不够就 ③）。

**③ 大模型兜底。** 仅当 ① 未命中且 ② 不够自信。输出合法枚举 JSON；非法则默认 `explain` 并记日志。此步仍不抽槽；对比/解释的实体留给 §3.3。

目标覆盖：线上约 80%～90% 在 ①+② 结束，③ 只吃难例。Harness 将误判句追加进正确类的原型（或调 τ/δ），不在对话中改规则。

输出：`intent`、`intent_confidence`、`intent_source`（`rule` | `embedding` | `llm`）。

### 3.3 结构化工单（类型已定之后）

在 §3.2 给出 `intent` 之后，仅对研究类 / 推荐类填槽与规划。封闭槽用规则；开放 `topic` / 实体用 LLM 提案 + §4 校验。指代（「第二篇」→ `referent_paper_ids`）优先在 ① 的 followup 规则里解析，其余在本步结合近轮标题补全。

```json
{
  "intent": "compare",
  "confidence": 0.91,
  "entities": {"methods": ["Transformer", "Mamba"], "datasets": [], "authors": []},
  "slots": {
    "topic": "sequence modeling efficiency",
    "need_sota": true,
    "need_fulltext": false,
    "expertise": "unknown",
    "year_from": null,
    "year_to": null,
    "top_k": 5
  },
  "missing_slots": [],
  "success_criteria": {
    "type": "compare",
    "min_papers_per_entity": 2,
    "min_chunks_rel_ge_3_per_entity": 2
  },
  "next_agent": "researcher",
  "playbook_id": "compare_v1",
  "budget": {"max_steps": 8, "max_search": 4, "max_fulltext": 2, "max_replan": 1}
}
```

工单写入黑板（LangGraph state），下游只读这份结构，不再重新「猜用户想干什么」。

---

## 4. 槽位校验与幻觉约束

### 4.1 谁填什么

| 槽 | 填写者 | 校验 |
|----|--------|------|
| `intent` | §3.2 三级瀑布 | 枚举；非法则默认 `explain` |
| `need_sota` | 规则（最新/SOTA/recent）或小模型 | 布尔 |
| `expertise` | 画像或小模型 | 枚举：`beginner` / `expert` / `unknown` |
| 年份 | 正则 | 超出配置窗口则改写为「recent」或记入 corrections |
| 方法名 / 实体 | 词典模糊匹配 + 可选 LLM | 必须能在 query 或近 2 轮用户话里对齐，或落在同义表 |
| `topic` | LLM 可提案 | 见 4.2；失败则 `topic = 用户原句` 或列入 `missing_slots` |

### 4.2 开放 topic 校验（必须程序化）

对 LLM 抽出的 `topic` 依次：

1. **覆盖：** 去掉「帮我找/相关/论文」后，topic 中实词须在原 query 命中或命中同义表（例：检索增强 ↔ RAG）。topic 中出现原句与同义表都没有的实体（如凭空 `GPT-4`）→ 拒绝。
2. **语义：** `cosine(embed(query), embed(topic)) ≥ τ`（建议 0.75，可配）。
3. **压缩：** topic 有效长度不得显著长于 query 的有效部分。
4. **可选 NLI：** premise = 用户句，hypothesis = `用户想找关于「{topic}」的论文`，须为 entailment。

拒绝后：`recommend`/`explain` 用原句做检索 query，不把 LLM 润色当事实。

### 4.3 `success_criteria`：模板实例化，禁止模型写作文

由开发者维护意图 → 检查清单模板。Planner 只填数字和实体。Researcher / Critic 只打勾。

| 意图 | 模板（示例，阈值可配） |
|------|------------------------|
| `recommend` | `len(ranked) ≥ top_k` 且 top 分数 ≥ θ；若 `need_sota` 则至少 1 篇年份满足窗口 |
| `daily` | 列表非空；与近 N 天已推 paper_id 去重 |
| `explain` | 相关度 ≥ 3 的 chunk 数 ≥ N（如 4） |
| `compare` | 每个 entity 论文数 ≥ 2 且高相关 chunk ≥ 2 |
| `survey` | 子主题覆盖数 ≥ 配置的子查询数的 2/3 |
| `factoid` | 至少 1 条断言能 grounding，否则必须不确定表述 |

模型输出的 `finish` 若未通过程序检查，视为非法，继续级联或打回。

---

## 5. 共享 Tools 层

路径建议：`langgraph_agents/tools/`（或 `agent/tools/`），纯函数 + 少量启发式，**无对话职责**。

| Tool | 签名（逻辑） | 调用方 |
|------|----------------|--------|
| `search_local` | `(query, filters, k) → List[Paper]` | Recommend、Researcher |
| `search_arxiv` | `(query, categories, date_range, k) → List[Paper]` | 同上，受级联与 `need_sota` 约束 |
| `search_s2` | 可选 Semantic Scholar | 同上 |
| `recall_multi` | `(profile, query=None, channels, k) → List[Paper]` | **仅 Recommend**（向量 + 规则 + ItemCF） |
| `rerank` | `(candidates, query, user, top_k) → List[Paper]` | Recommend 必用；Researcher 可选 |
| `gather_evidence` | `(papers, question, max_chunks) → List[Chunk]` | **仅 Researcher** |
| `read_fulltext` | `(paper_id) → text` | Researcher，受 `max_fulltext` 限制 |
| `ask_user` | `(slot, question)` | Planner 缺槽时由 Writer 执行；Researcher 级联末档也可请求 |

现有 `RetrievalAgent` / `OnlineSearchAgent` / `RecallAgent` / `RankAgent` 应 **降级为这些 tool 的实现包装**，不再作为图上的对等 Agent。

`run_recommend(...)` 可作为 Recommend 子图对外的粗粒度接口，供 `feedback` 后重推调用；**不要**作为 explain/compare 的默认第一步。

---

## 6. Planner Agent

### 6.1 职责

1. 任务类型三级瀑布（规则 → 问法向量近邻 → 大模型兜底），写入 `intent` / `intent_source`。
2. 抽槽并校验；写 `missing_slots` 与指代 `referent_paper_ids`。
3. 套用 `success_criteria` 模板；选择 `playbook_id` 与 `budget`。
4. 设置 `next_agent`。
5. 若本回合是 Researcher stall 后的重入：根据 Progress 摘要改 playbook（放宽年份、允许 arxiv），`max_replan` 用尽则转为澄清或降质交付。

### 6.2 路由

```text
if missing_slots:           next = writer          # 只问槽
elif intent in {chitchat, meta}: next = writer
elif intent == feedback:    更新 profile; 若需重推 then recommend else writer
elif intent in {daily, recommend}: next = recommend
elif intent == followup and cited_papers and not need_more_papers:
                            next = writer
else:                       next = researcher
```

Planner **不是**每一步选 tool 的人。选 tool 是 Researcher 或 Recommend 内部的事。

### 6.3 与现有代码的关系

现有 `plan()` 中的过滤、上下文融合、术语扩展、父查询、子查询、HyDE 迁到：

- `_check_trivial` / `_fallback_route` 并入意图 ①；`decide_route` 的三路标签改为 10 类 + `intent_source`。
- 子查询生成：主要服务 `survey` / 复杂 `explain`，结果写入工单 `sub_queries`，由 Researcher 或并行检索消费。
- HyDE：仅短 query 且 `next=recommend` 或 `search_local` 时作为 **检索向量** 的辅助文本，不改用户可见 query。
- 三路 `NO_RETRIEVAL | RETRIEVE_LOCAL | NEED_CLARIFY` 废弃为对外路由，仅可作内部兼容别名。

---

## 7. Recommend Agent（子图）

### 7.1 何时出现在图上

仅当 `next_agent == recommend`：`daily`、有 Query 的 `recommend`、feedback 后需要新列表。

**explain / compare / survey / factoid 的图路由不得进入本子图。**

### 7.2 内部 DAG

```text
选通道 → 并行召回 → fuse → 去重（含会话已推）→ rerank → 级联检查 → 输出 ranked_papers
```

通道策略（规则，可少量 LLM 辅助但默认规则）：

| 条件 | 通道 |
|------|------|
| `daily` 且画像充足 | `recall_multi`：向量（兴趣）+ 规则 + ItemCF；可选弱开 arxiv |
| `daily` 冷启动 | 关 ItemCF；规则 + 热门/趋势 + 问槽后的兴趣向量 |
| `recommend` | `search_local(topic)` 为主；可用画像做轻度 `recall_multi` 打散；`need_sota` 或本地不足则 `search_arxiv` |
| 时延紧且候选已够 | 跳过 LLM 重排，只用 BGE LoRA（现有 `RankAgent` 的 bge 模式） |

### 7.3 级联（推荐路径）

1. 本地 / 召回为空或低于 `min_candidates` → `search_arxiv`（同一 topic 或兴趣词）。
2. 仍空 → 降质：放宽年份 / 类别，或 BM25-only。
3. 仍空 → 交给 Writer 说明覆盖不足（不编列表）。

ItemCF 不用于 explain/compare，避免「协同过滤」污染证据任务。

---

## 8. Researcher Agent：内部 Loop

同一套 while；用 **Playbook** 切换合法工具、建议顺序与禁令。

### 8.1 工作状态

```text
S.playbook          # allowed / forbidden / suggest
S.success           # 来自工单
S.evidence          # 见第 9 节
S.trace[]           # 本回合 tool 名、参数摘要、返回条数
S.steps, S.stall, S.searches, S.fulltext_reads
```

### 8.2 每步顺序（不可颠倒）

**A. 停止检查（纯规则）**

| 条件 | 动作 |
|------|------|
| `success` 程序检查通过 | 退出，把 pack 给 Writer |
| `steps ≥ max_steps` | 有部分证据则降质 `finish`；否则 `ask_user` 或失败 |
| `searches ≥ max_search` | 禁止新的 `search_*`，只允许 `gather_evidence` / `rerank` / `finish` |
| 连续 2 步 coverage / 篇数 / 最高分无增益 | `stall += 1`；`≥ 2` 则回 Planner 重规划或 `ask_user` |
| 工单仍缺实体（compare 不足 2 个） | `ask_user`，结束 loop |

**B. 规则级联抢权（模型不能否决）**

| 触发 | 强制动作 |
|------|----------|
| 刚完成的 `search_local` 为空或 INSUFFICIENT，且本 query/entity 未用过 arxiv | `search_arxiv` 同主题 |
| `search_arxiv` 仍空 | 允许 1 次换词 `search_local`；再空则 `ask_user` |
| compare 某 entity 论文数为 0 | 禁止 `gather_evidence`/`finish`，先补该边检索 |
| `gather_evidence` 后高分 chunk 仍低于成功线 | 换词 search（计入 `searches`） |
| 本地已够且 `need_sota=false` | 禁止再 arxiv |
| `read_fulltext` 次数将超过 `max_fulltext` | 拒绝该 tool |

**C. 否则：策略 LLM 在 allowed 集合中选 tool**

上下文只给压缩状态：playbook 摘要、各 entity 的 coverage 数字、上一 tool 的 `n` 与 max score、剩余预算。**不把全部 chunk 原文给选工具模型。**

输出必须是 JSON：`tool` + `args` + `reason`。程序二次过滤：

- tool ∈ `allowed` 且 ∉ `forbidden`
- compare：单次 search 的 query 不得同时包含两个 entity 的表面形式
- `read_fulltext` 仅当已锁定少量论文且摘要证据不足

非法则回退到 playbook 的默认下一步（有论文则 `gather_evidence`，否则 `search_local`）。

**D. 执行 tool，更新 evidence 与 trace，`steps += 1`，回到 A。**

### 8.3 Playbook

#### `explain_v1`

- allowed：`search_local`, `search_arxiv`, `gather_evidence`, `rerank`, `read_fulltext`, `ask_user`, `finish`
- forbidden：`recall_multi`（ItemCF）
- suggest：`search_local` → `gather_evidence` → 不够则 arxiv 或换词
- 默认不把 Recommend 子图当第一步

#### `compare_v1`

- 对 `entities.methods`（或 papers）逐个 `search_local(entity)`，禁止 `"A vs B"` 一次搜完
- 缺边强制 arxiv
- 然后 `gather_evidence(question=原对比问句)`
- success 看 `coverage[entity]`

#### `survey_v1`

- 使用 Planner 产出的 `sub_queries`（须逐条做 topic 同类覆盖校验）
- 可并行 `search_local`（LangGraph `Send` / 并行节点）
- merge 后 `gather_evidence`；Writer 声明非穷尽

#### `factoid_v1`

- 优先对 `cited_papers` 做 `gather_evidence`
- 不足再精确 `search_local`（标题 / 作者）
- 禁止 ItemCF；级联终点是「无法确定」，不是推 10 篇
- `read_fulltext` 可在锁定 1 篇后使用

### 8.4 `rerank` 在 Researcher 中的含义

这是 **tool**，与 Recommend 内部调用同一 `rerank()` 实现。用于对当前候选按 question 精排后再 `gather_evidence`。  
**不是** 把控制流切到 Recommend Agent，也 **不会** 打开 ItemCF 通道。

---

## 9. Evidence Pack

Researcher 交给 Writer / Critic 的唯一知识来源。Writer 不得使用未入 pack 的论文。

```json
{
  "query": "讲讲检索增强智能体规划",
  "intent": "explain",
  "papers": [
    {
      "paper_id": "2403.xxxxx",
      "title": "...",
      "year": 2024,
      "authors": ["..."],
      "cite_n": 1,
      "source": "local",
      "abstract": "...",
      "rerank_score": 0.82
    }
  ],
  "chunks": [
    {
      "chunk_id": "2403.xxxxx#p3",
      "paper_id": "2403.xxxxx",
      "cite_n": 1,
      "text": "...",
      "section": "Method",
      "relevance": 4,
      "relevance_reason": "描述 retrieval 用于 planning"
    }
  ],
  "coverage": {
    "Transformer": {"n_papers": 3, "n_chunks_rel_ge_3": 5},
    "Mamba": {"n_papers": 2, "n_chunks_rel_ge_3": 4}
  },
  "open_gaps": [],
  "tool_trace_summary": ["search_local n=8", "gather_evidence chunks=12"]
}
```

`gather_evidence`：切 chunk → 按当前 question 打 1～5 分（交叉编码器或小 LLM map-reduce）→ 只保留 ≥ 3 的片段，每篇最多 2～3 条，pack 内 chunk 总数建议 8～12。

Recommend 路径不生产 chunks，只生产 `ranked_papers`；Writer 用列表模板，Critic 对 recommend 主要查「是否空列表 / 是否编造未出现的标题」。

---

## 10. Writer 与 Critic

### 10.1 Writer

输入：intent、history 截断、user_profile 语气、**本轮** `ranked_papers` 或 evidence pack。

模板：

| 意图 | 风格 |
|------|------|
| `recommend` / `daily` | 逐篇列表 + 理由，禁止假装做过全文精读 |
| `explain` / `survey` | 叙事，引用 `[N]`，文末不手写参考文献列表（系统附加） |
| `compare` | 方法 / 数据 / 结论对照，两边都必须出现引用 |
| `factoid` | 一两句；无 grounding 则明确不确定 |
| 澄清 | 只问 `missing_slots` 中的一项，不问开放闲聊 |

语言与用户 query 一致。禁止输出未在 papers/pack 中的论文名。

现有 `PaperQAAgent.respond` / `ask_profile` / `handle_feedback` 拆到 Writer + Planner 画像更新，避免 QA 再做路由。

### 10.2 Critic（在线，发出前）

动作空间仅三：

- `pass`：发给用户
- `rewrite`：证据还在，表述或引用标记有问题 → 回 Writer，不新搜
- `more_evidence`：覆盖不够 → 回 Researcher（若 `budget` 仍允许）；否则降质发出并在文中声明缺口

检查顺序（规则优先）：

1. 每个 `[k]` 能映射到 pack/列表中的 paper。
2. 抽样断言是否被对应 chunk 支持（语义对齐可用小 LLM；无 chunk 的 recommend 列表则检查标题是否均来自 `ranked_papers`）。
3. 程序检查 `success_criteria`。
4. 意图风控：explain 不得纯货架列表；recommend 不得只有散文没有条目。

Critic **不拥有** `search_*`。与线下 Harness Judge 分离：Critic 挡当次；Judge 打分改策略。

`rewrite` 次数建议上限 1；`more_evidence` 计入 Researcher `max_steps`，防止 Critic↔Researcher 死循环。

---

## 11. 数据存储、记忆管理与上下文管理

本节分三块：**系统里有哪些库、记忆分几层怎么读写淘汰、每次 LLM 调用的上下文怎么拼。** 三者必须对齐：记忆字段落在哪张表、进 prompt 时截成多长，不能各写各的。

### 11.1 数据库与存储清单

当前仓库已经配置、以及目标架构需要补齐的存储如下。Chroma 底层会用到 SQLite，但业务上不要把「论文元数据」只放在向量库里。

| 存储 | 现状 | 存什么 | 谁读写 |
|------|------|--------|--------|
| **PostgreSQL** | `backend` 已有 `papers`、`user_profiles` | 论文元数据、用户画像、曝光/阅读历史 | Recommend、Planner、反馈更新 |
| **ChromaDB**（可选 **Milvus**） | `VECTOR_DB_TYPE`，默认 `~/chroma_db` | 论文 embedding，供 `search_local` / 向量召回 | 检索 tools |
| **LanceDB / Numpy 向量** | `langgraph_agents` 路径常用 | 本地/实验用向量索引，接口与 Chroma 对齐 | 同上 |
| **Redis** | 配置了 host/DB，Celery broker | 会话热缓存、限流、可选 LangGraph checkpointer、日推去重集合 | Web/API、Recommend 去重 |
| **对象 / 本地 JSONL** | 基本没有结构化 episode | 回合轨迹、tool 打桩回放、Harness 报告 | 线上只追加；Harness 只读 |
| **Git 仓库内文件** | `data/`、`playbooks/`、`baselines/` | 金标、同义表、playbook、prompt 版本、baseline 快照 | 发版与 CI，不在请求中改 |
| **模型文件** | `~/bge-finetuned-v2` 等 | BGE-LoRA 重排器、embedding 模型 | `rerank` / 向量化 |

PostgreSQL 目标表（在现有 `papers` / `user_profiles` 上扩展，而不是再引入 Mongo）：

| 表 | 用途 |
|----|------|
| `users` / `user_profiles` | `interest_text`、`preferred_categories`、`role`、`purpose`、`special_requirements`、兴趣向量 id |
| `papers` | 标题、摘要、作者、年份、arxiv_id、url、类别；**不把 embedding 放这里** |
| `conversations` | `conversation_id`、`user_id`、创建/更新时间 |
| `messages` | 角色、截断后的文本、token 估计、关联 `episode_id` |
| `session_cited_papers` | 会话内 `[n] → paper_id`，followup 用 |
| `exposures` / `reads` | 已推/已读 paper_id，ItemCF 与 daily 去重 |
| `feedback_events` | 太老、不喜欢综述等，结构化约束 |
| `episodes` | 与 JSONL 双写或定期入库：intent、tool_calls、failure_type、critic、延迟 |
| `intent_prototypes` | 每类意图的原型 query + embedding，供 kNN 分类 |
| `harness_runs` | 每次复盘的指标、对比的 baseline 版本、提案路径 |

Redis 键设计（示例）：

- `sess:{conversation_id}`：最近 K 轮 + cited 列表（TTL 与会话一致，如 7 天）
- `daily_seen:{user_id}:{yyyy-mm-dd}`：当日已推 paper_id
- `ratelimit:{user_id}`
- `lg:checkpoint:{thread_id}`：若用 Redis checkpointer

向量库 **只存** `paper_id → embedding`（及检索用的 title/abstract 冗余字段）。元数据以 PostgreSQL 为准；id 对不上时以 PG 为源做对账脚本（现有 `filter_json_by_chroma_ids.py` 一类）。

**禁止：** 把完整 evidence pack、全文 PDF、Harness 提案写进 Redis 当长期记忆；把对话全文无限追加进 `user_profiles`。

### 11.1.1 要不要单独建「历史 Query 库」和「工作日志库」

你现在有的两套已经对：**文本/元数据库（PostgreSQL `papers`）+ 向量库（Chroma 等）**。缺的不是第三、第四套数据库引擎，而是 PostgreSQL 里面向 **交互与运维** 的表。

不要为历史 query、工作日志再上 Mongo / ES / 另一套 Postgres，除非日后量到必须拆读。原则是 **按数据形态分引擎，按用途分表**：

| 数据形态 | 引擎 | 为什么 |
|----------|------|--------|
| 论文标题/摘要/作者等结构化字段 | PostgreSQL | 过滤、去重、和曝光/引用做 join |
| 论文语义向量 | Chroma / Lance / Milvus | 近邻检索；id 必须等于 `papers.paper_id` |
| 用户 query、对话、画像、轨迹 | **还是 PostgreSQL 加表** | 都是结构化事件，要按 user/时间/intent 查询 |
| 热会话、日推去重 | Redis | TTL、高 QPS，可丢 |
| 超长 tool 原始返回、打桩快照 | JSONL / 对象存储 | 体积大、只给 Harness 重放 |

因此：

- **历史 Query：要存，做成表，不要做成独立数据库。**
- **工作日志（Agent 轨迹）：要存，就是 `episodes`（+ 可选 JSONL 双写），也不要独立引擎。**
- **应用/系统日志（报错、access log）：** 继续用现有 `logs/` 或标准 logging；不要和工作日志混成一张表。

#### 历史 Query 表 `queries`（建议新增）

一行 = 用户的一次自然语言输入（含 daily 的空 query 可跳过或记 `source=daily`）。

```text
query_id
user_id, conversation_id, episode_id
raw_query                 -- 用户原文
corrected_query           -- Planner 纠正后
topic                     -- 校验后的槽，可空
intent, intent_confidence
created_at
```

用途（和 `messages` 不完全重复）：

| 用途 | 为什么需要独立表而不只靠 messages |
|------|----------------------------------|
| 意图原型库 / 路由金标 | 只要 user 句 + 标好的 intent，不要 assistant 长回复 |
| 看用户兴趣随时间变化 | 对 `user_id` 做 query 序列，比扫全部聊天快 |
| Harness 统计「哪类问法空召回多」 | 按 intent、topic 聚合 |
| 个性化（可选） | 近期 query embedding 辅助画像，**不要**把整句塞进下一轮 prompt |

`messages` 仍要：完整多轮（user+assistant）、cited、followup。`queries` 是 messages 里 **user 侧的窄表 + 标注字段**。实现上可以每轮写 messages 时 **同步 insert queries**，用同一 `episode_id` 关联，避免两套事实。

保留策略：按用户最近 200 条或 90 天；脱敏后可进 Harness。不要把 query 再 embed 进论文向量库，以免检索出「用户问过的话」当论文。

#### 工作日志表 `episodes`（建议新增，即 Agent 轨迹）

一行 = 系统处理这一次请求的过程，不是用户原文本身。

```text
episode_id, query_id
policy_versions           -- playbook / prompt 版本
next_agent, playbook_id
tool_calls                -- JSON：tool、参数摘要、n、latency
failure_type, recovery
critic_decision
latency_ms, token_usage
```

这就是「工作日志」。Harness、换源救回率、回归重放都读它。  
**线上 Planner/Writer 默认不读这张表**（体积和隐私），只 append。

体积大的原始返回（arxiv XML、chunk 全文）不要进 PG：写入 `episodes_blobs/{episode_id}.jsonl` 或对象存储，表里只留摘要（n、max_score）。

#### 三张表怎么分工（避免重复建设）

```text
用户说了一句
    ├─ messages      对话展示与 followup（人读的历史）
    ├─ queries       这句话作为「问法样本」（机读的历史 query）
    └─ episodes      系统怎么跑的（工作日志）
论文本身
    ├─ papers        文本元数据
    └─ vector_db     embedding
```

`query_id` / `episode_id` / `message_id` 互相关联即可，不必三套库。

#### 数据处理流水线（论文 vs 交互）

**论文侧（你已有，保持双写）：**

```text
抓取/导入 → 清洗（去重、字段校验）
         → papers（PG，文本库）
         → embedding → 向量库（同一 paper_id）
         → 定期对账：PG 有、向量无则补 embed；向量有、PG 无则删或回填
```

**交互侧（你缺的，加表即可）：**

```text
每次请求结束（含失败）
  → insert queries（原文 + 工单里的 intent/topic）
  → insert messages（截断后的 user/assistant）
  → insert episodes（tool 摘要 + failure_type）
  → 异步：曝光写入 exposures；可选把 raw_query embed 进「用户 query 向量」小索引（与论文库分离）
Harness
  → 读 queries + episodes 做聚类和回归
  → 不写回论文向量库
```

**结论：** 继续用「一个 PostgreSQL + 一个向量引擎」两套库；把历史 query 和工作日志做成 PG 的 `queries`、`episodes`（外加 JSONL 存大对象）。只有当 episode 到千万级、或要做全文检索工作日志时，再考虑把轨迹迁到独立分析库 / OLAP，那是以后的事。

### 11.2 记忆分层（写什么、谁写、何时淘汰）

记忆 ≠ 把历史塞进 prompt。每一层有唯一写入者和淘汰策略。

```text
用户记忆（PG profile）     ── 跨周、跨会话
会话记忆（PG messages + Redis 热缓存 + cited）
回合工作记忆（LangGraph state，请求结束即丢弃大字段）
策略记忆（Git：prompt/playbook/阈值版本；Harness 只升级这一层）
轨迹记忆（JSONL / episodes 表，线上不读回对话）
```

#### （1）用户记忆（长期）

- **内容：** 兴趣短句、领域类别、role、purpose、禁忌（不要综述、只要近年）、近 N 篇已读/已曝光 id、可选兴趣 embedding。
- **写入：** Planner 在冷启动问槽成功后；Writer/Planner 处理 `feedback` 时；Recommend 曝光后异步写 `exposures`。
- **读取：** Recommend 的 `recall_multi` / `rerank`；Planner 判断是否还要问 topic。
- **不写入：** 每句聊天原文、chunk、tool 轨迹。
- **淘汰：** `exposures` 保留最近 90 天或最近 500 条；兴趣字段以最新反馈覆盖，不做隐式无限追加。

#### （2）会话记忆（中期）

- **内容：** `history` 最近 6～10 轮；`cited_papers`（最多约 20 篇）；`conversation_state`（主题、未决槽、上次 intent）。
- **写入：** Writer 发送成功后追加 message；本轮引用过的 paper 升入 cited；Critic `pass` 后才把 cited 视为「用户已看见」。
- **读取：** Planner 做 followup vs 新检索；factoid 优先 `gather_evidence(cited)`。
- **淘汰：** 超过 10 轮，旧轮经摘要模型压进 `conversation_state`（最多 400～800 字），原文从 Redis 删除，PG `messages` 可归档保留。cited 超出 20 篇时 LRU：未在最近 4 轮出现的先掉。

#### （3）回合工作记忆（短期）

- **内容：** 工单、budget、evidence pack、tool_trace、ranked_papers、critic 中间态。
- **写入：** 仅本轮 LangGraph state。
- **读取：** Researcher / Writer / Critic。
- **淘汰：** 请求结束。仅把「真实引用的 paper_id」和「1 条 episode」留下。未引用 chunk **必须丢弃**，否则下一轮 Writer 会看见过期证据。

#### （4）策略记忆（系统级，自我优化的落点）

- **内容：** playbook yaml、级联阈值、同义表、意图原型、Critic/Judge prompt 的 **版本号**（如 `compare_v1` → `compare_v2`）。
- **写入：** **只有 Harness 提案被合并发版之后**。线上 Agent 不得改这些文件。
- **读取：** Planner/Researcher 启动时加载当前版本。

#### （5）轨迹记忆（只给 Harness）

- **内容：** §14 的 episode。
- **线上：** 只 append，禁止读进 Planner/Writer 上下文（体积和隐私都不合适）。
- **线下：** Harness 聚类、救回率、回归重放。

可选「反思记忆」：Harness 把高频失败打成 **短条目**（如 `compare 时禁止单 query 含两个方法名`），写入版本化 `policies/lessons.md` 或 playbook 注释。这不是把用户对话当记忆，而是 **编译后的策略**。条目要有过期：被回归证明过时则删。

### 11.3 记忆读写时序（一轮请求）

```text
1. 加载 user_profile（PG，Redis 可缓存）
2. 加载 session：history K 轮 + cited（Redis 未命中则 PG）
3. Planner 读 1+2，写工单到 state（不落库）
4. Recommend / Researcher 写 ranked 或 pack 到 state
5. Writer 生成 → Critic
6. pass 后：messages、cited、exposures 落 PG/Redis；append episode
7. 丢弃 pack 全文与 tool 原始返回
```

失败（超时、Critic 一直 more_evidence 触顶）：仍写 episode（`failure_type` 有值）；不更新 cited；可选写一条系统消息「本轮未完成」。

### 11.4 上下文管理（每次 LLM 调用单独预算）

「上下文」= 这一次 API 调用 messages 里实际出现的 token，不是数据库里有的全部记忆。原则：**谁决策，给谁最小充分集。**

建议总预算（可按模型窗口改，下表按约 8k～16k 输入预留）：

| 调用 | 放入 | 明确不放入 | 截断顺序 |
|------|------|------------|----------|
| 意图 ① 规则 / ② 向量近邻 | query 原文 + 是否有 cited 的布尔 | history 全文、pack | 只截 query 512 字 |
| 意图 ③ 大模型兜底 | 同上；可选近 2 轮各 80 字（仅难例） | pack、tool 轨迹 | 尽量不带长历史 |
| Planner（抽槽/工单） | 本句；近 **4** 轮每轮 ≤150 字；cited **标题**最多 8 个；画像 200 字内 | chunks、tool 列表、旧 episode | 先丢更早轮，再截单轮 |
| Researcher 选 tool | playbook 摘要；coverage 数字；上一 tool 的 n / max_score；剩余 budget | chunk 原文、完整 abstracts | 无原文可截则只保留数字 |
| `gather_evidence` | 当前 question + **一篇论文内** 若干 chunk | 其他论文全文、对话历史 | 单 chunk ≤ 800 字；每篇最多 3 chunk |
| Recommend 不用 LLM 选通道时 | 无 LLM 上下文 | — | — |
| `rerank`（BGE） | query 与候选 title+abstract | 对话 | abstract 截到模型上限（如 512 token） |
| Writer | 系统人设；近 4 轮；pack 内 top **8** chunk 或 ranked **top_k** 篇的 title+摘要 400 字；cite 映射表 | 未引用候选、完整 tool_trace、Harness 报告 | chunk 按 relevance 丢尾；摘要截断 |
| Critic | 回答正文；被引用 `[k]` 对应 chunk | 整段会话、未引用论文 | 每条断言只带 1～2 个 chunk |
| 会话摘要任务 | 将被挤出窗口的旧轮 | 当前 pack | 输出固定 ≤800 字 JSON |

拼装顺序（Writer 为例，从必须保留到可丢）：

```text
1. 系统角色 + 意图模板 + 反幻觉条款
2. cite 映射 [n] → title（短）
3. evidence chunks / ranked 列表（相关性从高到低）
4. 近 4 轮对话
5. 画像一句（expertise / 禁忌）
超出窗口：先砍 4 中最早一轮，再砍 3 的低分 chunk，最后压缩 2。永不砍 1。
```

**上下文融合（多轮指代）** 优先在意图 ① 的 followup 规则中完成：把「那第二篇」解析成 `cited_papers[2]`，写入 `referent_paper_ids`，而不是把 20 轮原文丢给 Researcher。规则未命中时在抽槽步用近轮标题补全。

**隔离：** 选 tool 的模型看不到 Writer 的长证据；Writer 看不到「为什么选了 arxiv」的长推理。需要时只传 `tool_trace_summary` 一行。

### 11.5 实现要点（与现有代码）

- `WorkflowState.history` / `cited_papers` / `conversation_state` / `user_profile` 已存在，应对齐到 PG+Redis，避免 `web_app` 仅内存会话、进程一挂全丢。
- LangGraph checkpointer 存的是 **图状态快照**（可恢复 interrupt），不等于用户记忆；ttl 应短于 profile。
- 现有 QA 每轮 summarize conversation 可保留，但输出必须是固定 schema（topic、entities、open_slots），写入 `conversation_state`，不要自由散文当记忆。

---

## 12. 按意图的端到端路径

### 12.1 `chitchat` / `meta`

Planner → Writer → Critic（可跳过严格 grounding）→ 结束。

### 12.2 `feedback`

Planner 抽约束（太老、不要综述：规则优先）写入 profile。若上一轮是列表且约束影响排序 → Recommend 子图重跑 → Writer。否则 Writer 确认即可。

### 12.3 `daily`

Planner `next=recommend`，query 空。Recommend：`recall_multi(profile)` → rerank(兴趣文本) → Writer 列表模板 → Critic 查空列表与编造标题。

### 12.4 `recommend`（有 Query，走 Recommend 不是 Researcher）

Planner 校验 topic。Recommend：`search_local(topic)` + 可选画像打散 + 条件 arxiv → `rerank(topic)` → Writer 列表 → Critic。  
与 `daily` 共用子图，差别只在 query 来源与通道。

### 12.5 `explain`

Planner `next=researcher`，playbook=`explain_v1`。Researcher loop（见 §8）→ Writer 叙事 → Critic（断言 grounding）。  
图上 **没有** Recommend Agent 节点。

### 12.6 `compare`

Planner 要求 ≥2 个通过覆盖校验的 entity。Researcher 分 entity 检索并强制补边 → pack.coverage → Writer 对照表 → Critic 检查两边引用。  
`rerank` 仅作为对某一边候选的 tool。

### 12.7 `survey`

Planner 产出校验过的 `sub_queries`。Researcher 并行检索 + gather → Writer 分节 + 非穷尽声明 → Critic。

### 12.8 `factoid`

有 cited 则先 evidence；否则精确检索。Writer 短答。失败则不确定，不转推荐列表。

### 12.9 `followup`

默认 Writer 绑定 `cited_papers`。用户要求「再找类似」由 Planner 改判 `recommend` 或 `explain`。路由金标必须覆盖「像 followup 实际是新检索」的混淆。

---

## 13. 失败级联（全局梯子）

从「同工具重试」到「认输」，预算写死在工单里。

```text
1. 改写 query / 换词再 search_local
2. 换源 search_arxiv（或 S2）
3. 降质（跳过 LLM 重排、放宽过滤、摘要-only）
4. ask_user（只问缺失槽）
5. 诚实失败：汇报已尝试的工具，不编论文
```

Recommend 与 Researcher 共用梯子语义，但 Researcher 多一档：`gather_evidence` 低分视为「证据不足」再进入 1～2。

评测口径（线下从 log 计算）：

\[
\text{换源救回率} = \frac{\text{首次空召回后又换源且最终有可用论文的回合}}{\text{首次检索为空或 INSUFFICIENT 的回合}}
\]

另报：换源触发率、误换源率（本地已够仍联网）、无故澄清率。

---

## 14. Harness 与自我优化迭代

Harness 是套在服务外面的 **评测–复盘–回归车间**，不是第 6 个在线 Agent，也不是请求路径上的「自我反思模块」。  
**自我优化 = 用轨迹改进策略记忆（提示 / 阈值 / 意图原型 / playbook），并用金标防止改坏。** 线上 Agent 在对话里写「我错了」然后当场改 prompt，不算本系统的自我优化。

### 14.1 和在线路径的关系

| | 在线（用户在等） | 线下 Harness |
|--|------------------|--------------|
| 反思 | Critic 只决定 pass/rewrite/more_evidence | Judge 给整段轨迹打标签 |
| 改策略 | **禁止** | 出提案 → 回归 → 发版后才生效 |
| 读轨迹 | 不读历史 episode 进 prompt | 读 JSONL / `episodes` 表 |
| 写记忆 | profile / session / 本条 episode | 策略版本、baseline、可选 lessons |

### 14.2 线上 Episode（优化的原材料）

每回合追加一条，禁止只打非结构化 logger。可双写：本地 JSONL + PostgreSQL `episodes`。

```json
{
  "trace_id": "u123-t89",
  "ts": "2026-04-01T12:00:00Z",
  "user_id": "u123",
  "conversation_id": "c1",
  "policy_versions": {
    "playbook": "compare_v1",
    "planner_prompt": "planner_slot_v3",
    "critic": "critic_rule_v1"
  },
  "intent": "compare",
  "intent_source": "embedding",
  "intent_pred_confidence": 0.91,
  "next_agent": "researcher",
  "slots": {"entities": ["Transformer", "Mamba"]},
  "tool_calls": [
    {"tool": "search_local", "query": "Transformer", "n": 4, "latency_ms": 80},
    {"tool": "search_local", "query": "Mamba", "n": 0, "latency_ms": 70},
    {"tool": "search_arxiv", "query": "Mamba", "n": 6, "latency_ms": 900}
  ],
  "failure_type": "empty_retrieval",
  "recovery": "switch_arxiv",
  "critic": "pass",
  "scores": {"rerank_top": 0.71, "coverage": {"Mamba": 2}},
  "user_feedback": null
}
```

`failure_type`：`empty_retrieval` | `timeout` | `low_relevance` | `stale` | `hallucination_risk` | `cascade_miss` | `over_retrieve` | `slot_invalid` | `none`。

`policy_versions` 必须记：否则回归时无法回答「当时是哪版 playbook」。

### 14.3 Harness 流水线（每 N 条或每日）

```text
Ingest 新 episode
  → RuleGrade     规则打分（无 LLM）
  → LLMJudge      抽样语义分（judges/*.yaml，带版本）
  → Cluster       按 failure_type + intent 聚类
  → Propose       只写 harness/proposals/{date}-{id}.yaml
  → Replay        金标 + 对 tool 返回打桩的历史轨迹
  → Diff          对比 harness/baselines/{id}.json
  → Report        指标表 + 是否建议合并
  → Gate          默认人审；仅「改阈值且全绿」可配置自动合并
  → Release       合并后 bump 策略版本，写入 Git，线上下次加载
```

**RuleGrade 示例（应占自动分大头）：**

- 首次 `search_local` n=0 且全程无 `search_arxiv` → `cascade_miss`
- `followup` 且 cited 非空却调用了 `search_local` → `over_retrieve`
- compare 单次 query 同时含两个 entity → `slot_invalid`
- Critic 已 pass 但回答出现 pack 外标题 → `hallucination_risk`

**LLMJudge（独立 module，prompt 与在线 Critic 分目录）：** 意图是否对、topic 是否加料、该不该澄清、叙事是否列表化。只抽 20%～50% 以控制成本。

**Propose 输出必须可执行**，禁止只输出「建议提高检索质量」：

```yaml
id: 2026-04-01-014
target: langgraph_agents/playbooks/compare_v1.yaml   # 或 cascade 配置
type: rule | threshold | prompt_diff | classifier_data
change:
  empty_local_definition: "n < 3 or max_score < 0.4"
reason: "12 条 compare 轨迹单边只有 1 篇"
expected_metric: salvage_rate
```

提案默认 **不直接改主分支**。

### 14.4 「自我优化」具体改什么、谁改、怎么验收

优化对象只有 **策略记忆**，不是用户画像、不是当次回答。

| 改什么 | 例子 | 谁改 | 验收 |
|--------|------|------|------|
| 级联阈值 / 强制边 | 空召回定义、是否强制 arxiv | 人合并提案；阈值类可自动若 Replay 全绿 | 救回率↑、误换源率不升 |
| Playbook / Planner prompt | compare 禁止合成单 query | 人审 diff | 路由金标 + compare 双边覆盖率 |
| 同义表 / topic 校验 τ | 检索增强 ↔ RAG | 人审 | 槽位精确率、加料率↓ |
| 意图原型 / τ、δ | 把误判样本加入正确类原型，或收紧易混类阈值 | 人审后更新向量库 | 意图 macro-F1，recommend/explain 混淆↓ |
| Critic 规则 | 新加「explain 不得纯列表」 | 人审 | 拦截准确率、误杀率 |
| Writer 模板 | factoid 无引用必须说不确定 | 人审 | Faithfulness / grounding |

**不自动改：** embedding / BGE 权重（那是训练任务，单独 pipeline）；不在线上用用户反馈即时更新全量召回模型。

回归三层（改完必须全跑）：

1. **CI 单测：** `data/golden/routing.json`、`cascade.json`（空库必须换源、本地已够禁止换源）。
2. **打桩重放：** 历史 episode 的 tool 返回固化，新策略只改「下一步选什么」，避免 Arxiv 当天结果变化干扰。
3. **端到端评测集：** RAGAS、NDCG@3、换源救回率 vs `baselines/latest.json`。允许目标升、金标掉点超过红线（如 F1 -2pt）则拒绝合并。

这就是「改代码和 prompt，再回归测试」：优化闭环的 **执行者是 Harness + 工程师/CI**，不是 Researcher。

### 14.5 与记忆的衔接（优化结果如何回到线上）

```text
轨迹记忆（episodes）
    → Harness 聚类
    → 提案
    → 回归全绿 + 人审
    → 策略记忆版本 +1（playbook_v2、阈值配置）
    → 线上 Planner/Researcher 读新版本
```

用户记忆 **可以** 被反馈更新（「不要综述」），那是个性化，不是系统自我优化。不要把 Harness 的 lessons 写进每一个用户的 profile。

### 14.6 目录与触发

```text
harness/
  judges/           # LLM 打标签 prompt，带版本
  proposer/         # 出 yaml 提案
  replay/           # 打桩执行器
  baselines/        # 冻结指标
  proposals/        # 未合并草案
data/golden/
  routing.json
  cascade.json
scripts/run_harness.py   # cron：每 50 episode 或每日
```

触发：`len(new_episodes) ≥ N` 或每天 02:00，谁先到谁跑；跑的时候加锁，避免并行改同一 baseline。

### 14.7 明确不做的「自我进化」

- 不在对话中根据一句负反馈改全局 prompt。
- 不用 LLM 直接 overwrite `planner_agent.py`。
- 不以 Judge 分数为目标做在线 RL（会过拟合评测器）。
- Critic 的 rewrite 只改 **当次措辞**，不写入策略记忆。

---

## 15. LangGraph 状态与图拓扑

### 15.1 建议在现有 `WorkflowState` 上增加的字段

```text
intent, intent_confidence
slots, missing_slots, entities
success_criteria, playbook_id, budget
next_agent                    # writer | recommend | researcher
evidence_pack
plan_ledger                   # Task Ledger：目标、已知事实、假设
progress_ledger               # is_satisfied, is_progress, is_in_loop, stall
critic_decision               # pass | rewrite | more_evidence
episode_id
```

现有 `retrieval_result` / `fused_candidates` / `ranked_papers` 可继续作为 Recommend 子图内部字段；Researcher 以 `evidence_pack` 为对外合同。

### 15.2 图（逻辑）

```text
START → planner
          ├─ writer（澄清 / 闲聊 / followup 无检索）
          ├─ recommend_subgraph → writer
          └─ researcher_loop → writer
                writer → critic
                          ├─ END
                          ├─ writer（rewrite）
                          └─ researcher_loop（more_evidence，受预算限制）
```

`recommend_subgraph` 内部无环或仅级联一次 arxiv。  
`researcher_loop` 的环在节点内部实现亦可（单节点 while + tools），便于把 `max_steps` 做硬封顶；若拆成 LangGraph 回边，必须用 state 计数防无限循环。

---

## 16. 建议目录

```text
langgraph_agents/
  workflow.py                 # 主图：planner / recommend / researcher / writer / critic
  planner_agent.py            # 工单；分类器 + 校验
  recommend_agent.py          # 子图封装
  researcher_agent.py         # playbook + loop
  writer_agent.py             # 从 qa_agent 拆分生成职责
  critic_agent.py
  tools/
    search_local.py
    search_arxiv.py
    recall_multi.py
    rerank.py
    gather_evidence.py
    read_fulltext.py
  playbooks/
    explain_v1.yaml
    compare_v1.yaml
    survey_v1.yaml
    factoid_v1.yaml
  schemas/                    # Pydantic：Plan、EvidencePack、CriticDecision
harness/
  judges/
  proposer/
  replay/
  baselines/
data/golden/
  routing.json
  cascade.json
```

现有 `retrieval_agent.py`、`online_search_agent.py`、`recall_agent.py`、`rank_agent.py` 先改为 tools 的 facade，避免行为回退。

---

## 17. 评测矩阵

| 组件 | 指标 | 数据 |
|------|------|------|
| Planner | 意图 macro-F1；槽位精确率（实体覆盖校验通过率） | `planner_routing_testcases` 扩展为 10 类 + 混淆对（followup vs 新检索） |
| Recommend | NDCG@3 / Recall@3；相对 BGE-base 与 LLM 重排的延迟 | 现有 listwise / 离线集 |
| Researcher | 换源救回率、误换源率、平均 search 次数、compare 双边覆盖率 | 构造「本地无 / 网上有」与「本地已够」两套 |
| Writer+Critic | RAGAS Faithfulness 等；grounding 通过率；Critic 拦截准确率 | 现有 RAG 评测集 + 人工抽检 |
| 端到端 | 无故澄清率；冷启动问槽后推荐可接受率 | 会话日志 |

所有对外数字必须带：样本数、切分、对比系统、日期。禁止单独写「准确率 90%+」。

---

## 18. 落地顺序

1. **Tools 收口**：`search_local` / `search_arxiv` / `recall_multi` / `rerank` 统一接口，图上仍可先走旧边验证回归。
2. **Recommend 子图**：`daily` 与 `recommend` 共用，有 Query 推荐不再绕开重排。
3. **意图三级瀑布 + 工单 schema + topic 校验**；路由金标升级（含 recommend vs explain、followup vs 新检索）。
4. **Researcher loop + playbook**（先 explain/compare）；evidence pack 替换「把检索列表直接塞进 QA」。
5. **Writer 模板按意图拆分**；QA 不再路由。
6. **Critic 规则版**（cite 越界、缺边、空列表）；再加抽样 grounding。
7. **episode JSONL + 救回率脚本 + 金标重放**；Judge/Proposer 可第二阶段再加。

---

## 19. 与开源学术 Agent 的定位（实现时勿偏离）

本系统的差异化是 **个性化推荐 DAG 与受控学术问答环共用重排与画像**，不是把 PaperQA / STORM / Open Deep Research 做一遍。

| 对齐 | 不对齐 |
|------|--------|
| PaperQA：工具循环 + 证据分 + 发出前 grounding | 不做纯 PDF QA 产品；推荐是一等公民 |
| Magentic-One：Task / Progress Ledger、stall 重规划 | 不采用群聊运行时 |
| Open Deep Research：复杂题并行子检索、步数上限 | 不默认生成长报告 |
| 工业 RecSys：多路召回、LoRA 重排、漏斗指标 | 无 Query 路径禁止做成 ReAct |

实现完成的判据：有 Query 的 `recommend` 走 Recommend 子图；`explain`/`compare` 只走 Researcher 且可演示至少一次规则强制换源；Critic 能拦住无依据引用；Harness 能算出换源救回率并与 baseline 对比。

---

## 20. 与当前实现的对照

| 当前 (`langgraph_agents`) | 本文目标 |
|---------------------------|----------|
| Planner 三路路由 + 长 query 流水线 | 10 类意图（规则→向量近邻→LLM 兜底）+ 校验槽 + playbook 工单 |
| Retrieval / Online / Recall / Rank 作为并列节点 | 降为 Tools；Recommend 与 Researcher 按意图选用 |
| QA 生成 + 部分门控与 rerank 诊断环 | Writer 纯生成；诊断收进级联与 Critic |
| 无独立 Critic / Harness | 在线三动作 Critic + 线下回归 |
| 黑板字段偏检索列表 | 增加 evidence_pack、budget、progress_ledger |

升级时保持对外产品行为连续：每日推荐、联网搜新、BGE LoRA 重排、RAGAS 脚本仍可用；改变的是 **谁有权决策、失败如何强制恢复、回答能否在发出前被否决**。
