# 使用 uv 管理项目环境

## 什么是 uv？

`uv` 是由 Astral 开发的极速 Python 包管理器，比 pip 快 10-100 倍。它提供了：

- ⚡ 极快的包安装速度
- 🔒 可靠的依赖解析
- 📦 项目管理（类似 npm/yarn）
- 🐍 自动虚拟环境管理
- 🔄 锁文件支持

## 安装 uv

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Linux/Mac

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 使用 pip 安装

```bash
pip install uv
```

## 项目配置

项目使用 `pyproject.toml` 管理依赖，这是 Python 的现代标准。

### 项目结构

```
intellischolar/
├── pyproject.toml      # 项目配置和依赖
├── uv.lock            # 锁文件（自动生成）
└── .venv/             # 虚拟环境（自动创建）
```

## 常用命令

### 初始化项目

```bash
# 创建虚拟环境并安装依赖
uv sync

# 安装开发依赖
uv sync --dev
```

### 安装依赖

```bash
# 安装所有依赖
uv sync

# 添加新依赖
uv add package-name

# 添加开发依赖
uv add --dev package-name

# 移除依赖
uv remove package-name
```

### 运行命令

```bash
# 在虚拟环境中运行命令
uv run python script.py
uv run uvicorn backend.main:app --reload
uv run streamlit run frontend/app.py

# 运行测试
uv run pytest
```

### 虚拟环境管理

```bash
# uv 自动管理虚拟环境，通常不需要手动操作
# 虚拟环境位于 .venv/ 目录

# 激活虚拟环境（如果需要）
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

## 工作流示例

### 1. 首次设置

```bash
# 克隆项目
git clone <repo-url>
cd intellischolar

# 安装依赖
uv sync

# 配置环境变量
cp config/.env.example config/.env
# 编辑 config/.env
```

### 2. 日常开发

```bash
# 启动后端
uv run uvicorn backend.main:app --reload

# 启动前端
uv run streamlit run frontend/app.py

# 运行测试
uv run pytest
```

### 3. 添加新依赖

```bash
# 添加生产依赖
uv add requests

# 添加开发依赖
uv add --dev pytest-cov

# 提交更改
git add pyproject.toml uv.lock
git commit -m "Add new dependency"
```

## uv vs pip

| 特性 | uv | pip |
|------|-----|-----|
| 安装速度 | ⚡ 极快 (10-100x) | 🐌 较慢 |
| 依赖解析 | ✅ 可靠 | ⚠️ 有时冲突 |
| 锁文件 | ✅ 支持 | ❌ 不支持 |
| 项目管理 | ✅ 内置 | ❌ 需要额外工具 |
| 虚拟环境 | ✅ 自动管理 | ⚠️ 手动管理 |

## 迁移指南

### 从 requirements.txt 迁移

项目已经配置了 `pyproject.toml`，可以直接使用：

```bash
# 使用 uv
uv sync

# 或继续使用 pip（从 pyproject.toml）
pip install -e .
```

### 保留 requirements.txt

如果需要，可以保留 `requirements.txt` 作为备用：

```bash
# 从 pyproject.toml 生成 requirements.txt
uv pip compile pyproject.toml -o requirements.txt
```

## 故障排除

### 问题：uv 命令未找到

**解决方案：**
- 确保已正确安装 uv
- 重启终端
- 检查 PATH 环境变量

### 问题：依赖冲突

**解决方案：**
```bash
# 清理并重新安装
rm -rf .venv uv.lock
uv sync
```

### 问题：锁文件不同步

**解决方案：**
```bash
# 更新锁文件
uv lock --upgrade
```

## 参考资源

- [uv 官方文档](https://github.com/astral-sh/uv)
- [pyproject.toml 规范](https://peps.python.org/pep-0621/)
- [项目 README](../README.md)
