# IntelliScholar

基于 LLM 和 LangGraph 的论文推荐 / 论文问答系统。当前仓库同时包含两条实现路径：

- `langgraph_agents/`：当前更完整的 LangGraph 多智能体工作流，支持命令行单轮查询、交互式聊天和内置 Web UI。
- `backend/` + `frontend/`：FastAPI + Streamlit 的 API / MVP 界面实现。

如果你是第一次运行这个项目，建议优先从 `langgraph_agents` 路径开始。

## 当前能力

- 多智能体工作流：`Planner -> Retrieval / OnlineSearch / Recall -> Rank -> QA`
- 支持 query-based 检索、每日推荐、交互式论文问答
- 支持本地语料索引，也支持直接连接已有 ChromaDB
- 支持在线搜索补充候选论文
- 包含数据下载、向量库构建、RAG 评测与训练数据构建脚本

## 仓库结构

```text
Paper-Recommendation-System-Agent/
├── README.md
├── pyproject.toml
├── agent/                    # 共享基础模块：数据模型、embedding、vector store、data loader
├── langgraph_agents/         # 当前主要的 LangGraph 工作流实现
│   ├── workflow.py
│   ├── run_workflow.py
│   ├── web_app.py
│   ├── planner_agent.py
│   ├── retrieval_agent.py
│   ├── recall_agent.py
│   ├── rank_agent.py
│   ├── qa_agent.py
│   └── online_search_agent.py
├── backend/                  # FastAPI API、服务层与另一套工作流实现
│   ├── api/
│   ├── agents/
│   ├── services/
│   └── workflow/
├── frontend/                 # Streamlit 前端，调用 backend API
├── scripts/                  # 数据导入、下载、过滤、测试脚本
├── data/                     # 评测、训练数据构造与数据文件
└── logs/                     # 运行 web_app 后生成的日志
```

## 核心工作流

`langgraph_agents/workflow.py` 中的主流程大致如下：

```text
planner
  -> ask_clarify / handle_feedback / respond
  -> retrieval -> evaluate -> (re_retrieve loop) -> fuse -> rank -> respond
  -> online_search -> recall -> rank -> respond
```

几个关键模块分别负责：

- `PlannerAgent`：意图判断、查询改写、路由决策、检索质量评估
- `RetrievalAgent`：本地向量检索
- `RecallAgent`：离线召回与在线 / 离线融合
- `OnlineSearchAgent`：在线搜索新论文
- `RankAgent`：候选重排
- `PaperQAAgent`：最终生成推荐说明或论文问答回复

## 环境要求

- Python 3.10+
- 推荐使用 `uv`
- 可选服务：
  - ChromaDB 持久化目录
  - PostgreSQL
  - Redis

只跑 `langgraph_agents/run_workflow.py` 或 `langgraph_agents/web_app.py` 时，不一定需要把 PostgreSQL / Redis 全部准备好；但如果你要跑 `backend` API、数据写库或完整服务链路，建议把依赖服务都配齐。

## 安装依赖

### 方式一：`uv`（推荐）

```bash
uv sync
uv sync --dev
```

### 方式二：`pip`

```bash
pip install -e .
```

如果你只想装局部依赖，也可以按模块安装：

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
pip install -r langgraph_agents/requirements.txt
```

## 环境变量

项目默认从仓库根目录的 `.env` 读取配置。常用变量包括：

```bash
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

SEMANTIC_SCHOLAR_API_KEY=
CHROMA_PERSIST_DIR=~/chroma_db

OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4-turbo-preview
```

说明：

- `DEEPSEEK_API_KEY` 是 `langgraph_agents/run_workflow.py` 和 `langgraph_agents/web_app.py` 最常用的 LLM key。
- `SEMANTIC_SCHOLAR_API_KEY` 主要用于在线搜索增强。
- `CHROMA_PERSIST_DIR` 用于连接本地 ChromaDB。
- `backend/config.py` 还支持 PostgreSQL、Redis、Milvus 等服务配置。

## 快速开始

### 1. 命令行单轮查询

```bash
python -m langgraph_agents.run_workflow \
  --api-key "$DEEPSEEK_API_KEY" \
  --query "recommend papers on retrieval augmented generation"
```

### 2. 交互式聊天

```bash
python -m langgraph_agents.run_workflow \
  --api-key "$DEEPSEEK_API_KEY" \
  --chat
```

聊天模式内置命令：

- `/quit`：退出
- `/new`：新建会话
- `/debug`：查看会话状态
- `/papers`：查看当前已引用论文

### 3. 每日推荐模式

```bash
python -m langgraph_agents.run_workflow \
  --api-key "$DEEPSEEK_API_KEY" \
  --daily \
  --interest "machine learning systems"
```

### 4. 使用已有 ChromaDB

如果你已经有本地 ChromaDB，可以直接查询，不必在启动时重新索引：

```bash
python -m langgraph_agents.run_workflow \
  --api-key "$DEEPSEEK_API_KEY" \
  --chat \
  --chromadb \
  --chromadb-path ~/chroma_db
```

### 5. 启动内置 Web UI

`langgraph_agents/web_app.py` 会启动一个简单的 FastAPI Web 界面，默认地址为 `http://localhost:8000`：

```bash
python langgraph_agents/web_app.py \
  --api-key "$DEEPSEEK_API_KEY" \
  --port 8000
```

如果使用已有 ChromaDB：

```bash
python langgraph_agents/web_app.py \
  --api-key "$DEEPSEEK_API_KEY" \
  --chromadb \
  --chromadb-path ~/chroma_db
```

### 6. 启动 backend API

FastAPI 主入口在 `backend/main.py`：

```bash
uvicorn backend.main:app --reload
```

默认提供的接口包括：

- `GET /`
- `GET /health`
- `POST /api/v1/recommendations/`
- `GET /api/v1/recommendations/daily/{user_id}`
- `GET /api/v1/users/{user_id}/profile`
- `POST /api/v1/users/{user_id}/feedback`

### 7. 启动 Streamlit 前端

`frontend/app.py` 会请求 `http://localhost:8000/api/v1`，所以需要先启动 `backend.main`：

```bash
streamlit run frontend/app.py
```

## 数据准备

### 下载论文 JSON

从 Hugging Face 的 `davanstrien/arxiv-cs-papers-lance` 数据集下载论文并导出为 JSON：

```bash
python scripts/download_papers_json.py \
  -n 5000 \
  -o data/papers_downloaded.json
```

### 导入向量库 / 元数据库

从 JSON 导入：

```bash
python scripts/load_papers.py json data/papers_downloaded.json
```

从 Lance / Hugging Face 数据集导入：

```bash
python scripts/load_papers.py lance davanstrien/arxiv-cs-papers-lance -n 5000
```

### 按 Chroma 中已有 ID 过滤 JSON

这个脚本适合在你已经有一份完整 JSON、但只想保留当前 Chroma collection 中实际存在的论文时使用：

```bash
python scripts/filter_json_by_chroma_ids.py \
  --input-json data/papers_downloaded.json \
  --output-json data/papers_downloaded_60k_for_bm25.json \
  --chromadb-path ~/chroma_db
```

## 评测与数据脚本

`data/` 目录下保留了若干评测和训练数据构建脚本，例如：

- `data/create_rag_eval_dataset.py`
- `data/eval_rag_ragas.py`
- `data/eval_rag_with_system_ragas.py`
- `data/create_listwise.py`
- `data/create_listwise2.py`
- `data/bge_reranking_train.py`

这些脚本主要用于：

- 构建 RAG 评测集
- 使用 Ragas 做自动评测
- 构建排序训练样本
- 进行重排模型相关实验

## License

MIT
