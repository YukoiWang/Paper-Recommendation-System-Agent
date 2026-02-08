# IntelliScholar 快速开始指南

## 前置要求

- Python 3.10+
- PostgreSQL 14+
- Redis 6+
- Git
- **uv** (推荐) 或 pip

## 安装 uv (推荐)

`uv` 是一个极快的 Python 包管理器，比 pip 快 10-100 倍。

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip 安装
pip install uv
```

## 快速安装

### 方法一：使用 uv (推荐) ⚡

#### 1. 克隆项目（如果适用）

```bash
git clone <repository-url>
cd intellischolar
```

#### 2. 使用 uv 创建虚拟环境并安装依赖

```bash
# uv 会自动创建虚拟环境并安装所有依赖
uv sync

# 激活虚拟环境（uv 会自动管理）
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

#### 3. 安装开发依赖（可选）

```bash
uv sync --dev
```

### 方法二：使用传统 pip

#### 1. 克隆项目

```bash
git clone <repository-url>
cd intellischolar
```

#### 2. 创建虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

#### 3. 安装依赖

```bash
# 从 pyproject.toml 安装（推荐）
pip install -e .

# 或使用 requirements.txt（备用）
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 4. 配置环境变量

```bash
# 复制示例配置文件
cp config/.env.example config/.env

# 编辑 config/.env，填写以下关键配置：
# - POSTGRES_* (数据库配置)
# - OPENAI_API_KEY (LLM API密钥)
# - REDIS_HOST, REDIS_PORT (Redis配置)
```

### 5. 初始化数据库

```bash
# 确保PostgreSQL正在运行
# 创建数据库
createdb intellischolar

# 初始化表结构
python scripts/init_db.py
```

### 6. 启动Redis

```bash
# Windows (如果已安装)
redis-server

# 或使用Docker
docker run -d -p 6379:6379 redis:latest
```

### 7. 启动服务

#### 启动后端API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

#### 启动前端（新终端）

```bash
cd frontend
streamlit run app.py
```

### 8. 访问应用

- 前端界面: http://localhost:8501
- API文档: http://localhost:8000/docs
- API健康检查: http://localhost:8000/health

## 加载示例数据（可选）

如果有论文数据JSON文件：

```bash
python scripts/load_papers.py data/papers.json
```

## 测试API

```bash
# 健康检查
curl http://localhost:8000/health

# 获取推荐
curl -X POST http://localhost:8000/api/v1/recommendations/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "query": "transformer attention",
    "top_n": 10
  }'
```

## 常见问题

### 1. 数据库连接失败

- 检查PostgreSQL是否运行
- 验证 `config/.env` 中的数据库配置
- 确认数据库已创建

### 2. Redis连接失败

- 检查Redis是否运行
- 验证Redis端口（默认6379）

### 3. LLM API调用失败

- 检查API密钥是否正确
- 验证网络连接
- 查看日志错误信息

### 4. 向量数据库错误

- ChromaDB会自动创建，检查目录权限
- 如果使用Milvus，确保Milvus服务运行

## 下一步

- 阅读 [开发指南](docs/development.md)
- 查看 [API文档](docs/api.md)
- 了解 [系统架构](docs/architecture.md)
