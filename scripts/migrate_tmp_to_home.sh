#!/bin/bash
# 在 gl-login5 上运行，把 /tmp 里的数据迁到 ~/
# Usage: bash scripts/migrate_tmp_to_home.sh

set -e
SLED_IP="141.212.48.176"
HOME_BASE="$HOME"

echo "=== 迁移 /tmp 数据到 $HOME_BASE ==="

# 1. ChromaDB
if [ -d /tmp/chroma_db ]; then
    echo "[1/5] 迁移 chroma_db (~14GB)..."
    mv /tmp/chroma_db "$HOME_BASE/chroma_db"
    echo "      -> $HOME_BASE/chroma_db"
else
    echo "[1/5] /tmp/chroma_db 不存在，从 sled-whistler 拉取..."
    scp -r yukiwang@${SLED_IP}:/tmp/chroma_db "$HOME_BASE/chroma_db"
fi

# 2. paper-rec-venv
if [ -d /tmp/paper-rec-venv ]; then
    echo "[2/5] 迁移 paper-rec-venv (~6GB)..."
    mv /tmp/paper-rec-venv "$HOME_BASE/paper-rec-venv"
    echo "      -> $HOME_BASE/paper-rec-venv"
else
    echo "[2/5] /tmp/paper-rec-venv 不存在，跳过"
fi

# 3. listwise_data
if [ -d /tmp/listwise_data ]; then
    echo "[3/5] 迁移 listwise_data..."
    mv /tmp/listwise_data "$HOME_BASE/listwise_data"
    echo "      -> $HOME_BASE/listwise_data"
else
    mkdir -p "$HOME_BASE/listwise_data"
fi

# 4. 从 sled-whistler 拉取新数据
echo "[4/5] 从 sled-whistler 拉取新合成数据..."
scp yukiwang@${SLED_IP}:/tmp/listwise_data/listwise_1500_3.json "$HOME_BASE/listwise_data/" 2>/dev/null || true
scp yukiwang@${SLED_IP}:/tmp/listwise_data/listwise_combined_2500.json "$HOME_BASE/listwise_data/" 2>/dev/null || true

# 5. listwise_cache
if [ -d /tmp/listwise_cache ]; then
    echo "[5/5] 迁移 listwise_cache..."
    mv /tmp/listwise_cache "$HOME_BASE/listwise_cache"
else
    mkdir -p "$HOME_BASE/listwise_cache"
fi

# 6. BGE 微调模型（如果存在）
if [ -d /tmp/bge-finetuned-v2 ]; then
    echo "[6/6] 迁移 bge-finetuned-v2..."
    mv /tmp/bge-finetuned-v2 "$HOME_BASE/bge-finetuned-v2"
    echo "      -> $HOME_BASE/bge-finetuned-v2"
else
    echo "[6/6] 尝试从 sled-whistler 拉取 bge-finetuned-v2..."
    scp -r yukiwang@${SLED_IP}:/tmp/bge-finetuned-v2 "$HOME_BASE/" 2>/dev/null || echo "      (训练可能未完成，跳过)"
fi

echo ""
echo "=== 迁移完成 ==="
echo ""
echo "后续启动 web 服务请用："
echo "  source $HOME_BASE/paper-rec-venv/bin/activate"
echo "  cd ~/Paper-Recommendation-System-Agent"
echo "  python langgraph_agents/web_app.py --chromadb --chromadb-path $HOME_BASE/chroma_db --port 8080"
echo ""
