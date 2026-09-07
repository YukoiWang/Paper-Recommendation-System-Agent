# IntelliScholar 系统架构文档

目标多智能体实现规格（Planner / Recommend / Researcher / Writer / Critic、共享 Tools、Harness）见 [multi_agent_system_design.md](./multi_agent_system_design.md)。下文描述的是仓库当前组件划分。

## 系统架构图

详见项目README中的架构说明。

## 核心组件

### 1. 多智能体系统（Agents）

- **Planner Agent**: 决策路由和任务规划
- **Online Search Agent**: 在线搜索新论文（arXiv、Semantic Scholar）
- **Offline Retrieval Agent**: 离线检索（向量数据库）
- **Recall Agent**: 召回候选集合
- **Ranking Agent**: 双阶段排序
- **UI Agent**: 用户界面生成和画像更新

### 2. 服务层（Services）

- **VectorDBService**: 向量数据库服务
- **MetadataDBService**: 元数据数据库服务
- **EmbeddingService**: 文本向量化服务
- **RankingModelService**: 排序模型服务
- **LLMService**: 大语言模型服务
- **UserProfileService**: 用户画像管理服务
- **DedupFilterService**: 去重过滤服务

### 3. 工作流（Workflow）

- **RecommendationWorkflow**: 推荐工作流编排

## 数据流

1. 用户请求 → Planner规划
2. 多路召回（Online Search + RAG + ItemCF）
3. 去重过滤
4. 双阶段排序（模型排序 + LLM重排）
5. UI格式化
6. 用户画像更新
