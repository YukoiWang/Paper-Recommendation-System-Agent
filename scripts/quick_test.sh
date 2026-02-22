#!/bin/bash
# 快速测试脚本 - 测试引用上下文API功能

echo "=========================================="
echo "引用上下文API快速测试"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 测试论文ID
PAPER_ID="10.1038/nature12373"

echo "测试论文ID: $PAPER_ID"
echo ""

# 测试1: 免费模式
echo -e "${YELLOW}=== 测试1: 免费模式（无需API key）===${NC}"
python3 scripts/get_citation_context_s2.py \
    --paper-id "$PAPER_ID" \
    --limit 3 \
    --summary-only

echo ""
echo "----------------------------------------"
echo ""

# 检查是否有API key
if [ -z "$S2_API_KEY" ]; then
    echo -e "${YELLOW}提示: 未设置S2_API_KEY环境变量${NC}"
    echo ""
    echo "要测试完整功能（引用上下文），请："
    echo "1. 获取API key: https://www.semanticscholar.org/product/api"
    echo "2. 设置环境变量: export S2_API_KEY=your-api-key"
    echo "3. 重新运行此脚本"
    echo ""
    exit 0
fi

# 测试2: API key有效性
echo -e "${YELLOW}=== 测试2: API key有效性检查 ===${NC}"
python3 scripts/test_s2_api_key.py

if [ $? -ne 0 ]; then
    echo -e "${RED}API key测试失败，请检查你的API key${NC}"
    exit 1
fi

echo ""
echo "----------------------------------------"
echo ""

# 测试3: 完整功能测试
echo -e "${YELLOW}=== 测试3: 完整引用上下文功能 ===${NC}"
python3 scripts/get_citation_context_s2.py \
    --paper-id "$PAPER_ID" \
    --limit 5 \
    --output /tmp/test_citations.json

if [ -f /tmp/test_citations.json ]; then
    echo ""
    echo -e "${GREEN}✅ 数据已保存到 /tmp/test_citations.json${NC}"
    echo ""
    echo "查看前50行:"
    head -50 /tmp/test_citations.json | python3 -m json.tool 2>/dev/null || cat /tmp/test_citations.json | head -50
fi

echo ""
echo -e "${GREEN}=========================================="
echo "测试完成！"
echo "==========================================${NC}"
