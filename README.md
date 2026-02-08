# IntelliScholar - 多智能体论文推荐系统

## 项目简介

IntelliScholar 是一个基于 LLM + 多智能体架构（LangGraph）的智能学术论文推荐系统，面向机器学习领域研究者，提供**每日个性化论文推荐**，并通过用户交互实现持续自适应优化。

## 核心特性

- 多智能体协作（Planner + 5 Agents）
- 混合召回（Online Search + RAG + ItemCF）
- 双阶段排序（传统模型排序 + LLM 动态语义重排）
- 动态用户画像（文本摘要式 Profile Memory）
- 去重系统（已读/已曝光过滤）

## 技术栈

### 后端
- FastAPI
- **LangGraph**（多智能体编排，使用StateGraph实现条件路由）
- Celery + Redis（异步任务）

### 数据层
- ChromaDB / Milvus（向量数据库）
- PostgreSQL（元数据数据库）
- Redis（缓存）

### 模型层
- Embedding: bge-large-zh / bge-m3
- LLM: GPT-4 / Qwen2.5-72B / DeepSeek-V3
- Ranking: LightGBM / DeepFM

### 前端
- Streamlit（MVP阶段）
- Next.js + React（生产阶段）

## 项目结构

```
intellischolar/
├── backend/              # 后端代码
│   ├── agents/          # 多智能体实现
│   ├── services/        # 业务服务层
│   ├── models/          # 数据模型
│   ├── data/            # 数据处理
│   ├── utils/           # 工具函数
│   └── api/             # API路由
├── frontend/            # 前端代码
├── data/                # 数据文件
├── config/              # 配置文件
├── scripts/             # 脚本文件
├── tests/               # 测试代码
└── docs/                # 文档
```

## 开发路线图

### Phase 1: MVP（核心闭环）
- 单用户画像
- RAG 检索推荐
- LangGraph 多 Agent 流程
- 基础对话

### Phase 2: 功能完善
- Online Search Agent
- 混合召回（RAG + 在线搜索 + ItemCF）
- LightGBM Ranker
- 用户行为特征注入

### Phase 3: 智能化优化
- 主动式推荐
- 动态特征加权
- 自适应用户建模
- 认知级推荐

## 快速开始

### 环境要求
- Python 3.10+
- Node.js 18+（前端）
- PostgreSQL 14+
- Redis 6+

### 安装依赖

#### 使用 uv (推荐) ⚡

```bash
# 安装 uv (如果还没有)
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac: curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装所有依赖（自动创建虚拟环境）
uv sync

# 安装开发依赖
uv sync --dev
```

#### 使用 pip (传统方式)

```bash
# 从 pyproject.toml 安装
pip install -e .

# 或使用 requirements.txt
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 配置环境变量

复制 `config/.env.example` 到 `config/.env` 并填写配置。

### 运行服务

#### 使用 uv

```bash
# 启动后端
uv run uvicorn backend.main:app --reload

# 启动前端（Streamlit）
uv run streamlit run frontend/app.py
```

#### 使用传统方式

```bash
# 启动后端
cd backend
uvicorn main:app --reload

# 启动前端（Streamlit）
cd frontend
streamlit run app.py
```

## 许可证

MIT License
