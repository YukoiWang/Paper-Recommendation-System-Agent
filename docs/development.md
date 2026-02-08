# 开发指南

## 环境设置

### 1. Python环境

#### 使用 uv (推荐) ⚡

```bash
# 安装 uv (如果还没有)
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac: curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync

# 安装开发依赖
uv sync --dev

# 激活虚拟环境（如果需要）
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
```

#### 使用传统 pip

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .  # 从 pyproject.toml 安装
# 或
pip install -r backend/requirements.txt
```

详细说明请参考 [uv 设置指南](uv_setup.md)

### 2. 数据库设置

#### PostgreSQL

```bash
# 创建数据库
createdb intellischolar

# 运行初始化脚本
python scripts/init_db.py
```

#### Redis

```bash
# 启动Redis
redis-server
```

#### ChromaDB

ChromaDB会自动在指定目录创建，无需额外配置。

### 3. 环境变量

复制 `config/.env.example` 到 `config/.env` 并填写配置。

## 运行服务

### 使用 uv

```bash
# 后端
uv run uvicorn backend.main:app --reload

# 前端（Streamlit）
uv run streamlit run frontend/app.py
```

### 使用传统方式

```bash
# 后端
cd backend
uvicorn main:app --reload

# 前端（Streamlit）
cd frontend
streamlit run app.py
```

## 数据加载

```bash
# 使用 uv
uv run python scripts/load_papers.py data/papers.json

# 或传统方式
python scripts/load_papers.py data/papers.json
```

## 开发路线图

### Phase 1: MVP
- [x] 项目框架搭建
- [ ] 基础Agent实现
- [ ] RAG检索
- [ ] 基础UI

### Phase 2: 功能完善
- [ ] Online Search集成
- [ ] 混合召回
- [ ] LightGBM排序模型
- [ ] 用户行为反馈

### Phase 3: 智能化优化
- [ ] 主动式推荐
- [ ] 动态特征加权
- [ ] 认知级推荐
