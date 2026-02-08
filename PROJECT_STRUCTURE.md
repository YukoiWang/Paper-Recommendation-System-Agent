# IntelliScholar 项目结构

```
intellischolar/
├── backend/                    # 后端代码
│   ├── agents/                # 多智能体实现
│   │   ├── __init__.py
│   │   ├── planner_agent.py          # Planner Agent - 决策路由
│   │   ├── online_search_agent.py    # Online Search Agent - 在线搜索
│   │   ├── offline_retrieval_agent.py # Offline Retrieval Agent - 离线检索
│   │   ├── recall_agent.py           # Recall Agent - 召回候选
│   │   ├── ranking_agent.py           # Ranking Agent - 排序
│   │   └── ui_agent.py               # UI Agent - 用户界面
│   │
│   ├── services/              # 业务服务层
│   │   ├── __init__.py
│   │   ├── vector_db.py              # 向量数据库服务
│   │   ├── metadata_db.py            # 元数据数据库服务
│   │   ├── embedding.py              # 文本向量化服务
│   │   ├── ranking_model.py          # 排序模型服务
│   │   ├── llm.py                    # LLM服务
│   │   ├── user_profile.py           # 用户画像服务
│   │   └── dedup_filter.py           # 去重过滤服务
│   │
│   ├── workflow/               # 工作流编排
│   │   ├── __init__.py
│   │   └── recommendation_workflow.py # 推荐工作流
│   │
│   ├── api/                    # API路由
│   │   ├── __init__.py
│   │   ├── recommendations.py        # 推荐接口
│   │   └── user_profile.py           # 用户画像接口
│   │
│   ├── models/                 # 数据模型（ORM）
│   │   └── __init__.py
│   │
│   ├── data/                   # 数据处理
│   │   └── __init__.py
│   │
│   ├── utils/                  # 工具函数
│   │   └── __init__.py
│   │
│   ├── config.py               # 配置管理
│   ├── main.py                 # 应用入口
│   ├── requirements.txt        # Python依赖
│   └── __init__.py
│
├── frontend/                   # 前端代码
│   ├── app.py                  # Streamlit应用
│   └── requirements.txt        # 前端依赖
│
├── config/                     # 配置文件
│   └── .env.example            # 环境变量示例
│
├── scripts/                    # 脚本文件
│   ├── init_db.py              # 数据库初始化
│   └── load_papers.py          # 论文数据加载
│
├── tests/                      # 测试代码
│   └── __init__.py
│
├── docs/                       # 文档
│   ├── architecture.md         # 架构文档
│   ├── api.md                  # API文档
│   └── development.md          # 开发指南
│
├── data/                       # 数据文件目录（需创建）
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后的数据
│
├── models/                     # 模型文件目录（需创建）
│   └── ranking/                # 排序模型
│
├── .gitignore                  # Git忽略文件
├── README.md                   # 项目说明
└── PROJECT_STRUCTURE.md        # 本文件
```

## 核心模块说明

### Agents（智能体）

1. **Planner Agent**: 根据用户查询和画像，规划推荐流程
2. **Online Search Agent**: 从arXiv、Semantic Scholar等API获取最新论文
3. **Offline Retrieval Agent**: 从向量数据库检索相似论文
4. **Recall Agent**: 整合多路召回结果，生成候选集
5. **Ranking Agent**: 双阶段排序（模型排序 + LLM重排）
6. **UI Agent**: 格式化推荐结果，更新用户画像

### Services（服务层）

1. **VectorDBService**: 向量数据库操作（ChromaDB/Milvus）
2. **MetadataDBService**: 元数据数据库操作（PostgreSQL）
3. **EmbeddingService**: 文本向量化（sentence-transformers）
4. **RankingModelService**: 排序模型推理（LightGBM）
5. **LLMService**: 大语言模型调用（OpenAI/Qwen/DeepSeek）
6. **UserProfileService**: 用户画像管理
7. **DedupFilterService**: 去重过滤

### Workflow（工作流）

- **RecommendationWorkflow**: 编排所有Agent，执行完整推荐流程

## 数据流

```
User Request
    ↓
Planner Agent (规划)
    ↓
Recall Agent (召回)
    ├── Online Search Agent
    ├── Offline Retrieval Agent (RAG)
    └── ItemCF (协同过滤)
    ↓
Dedup Filter (去重)
    ↓
Ranking Agent (排序)
    ├── Model Ranking (LightGBM)
    └── LLM Re-rank
    ↓
UI Agent (格式化)
    ↓
User Response + Profile Update
```
