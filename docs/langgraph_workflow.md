# LangGraph 工作流实现说明

## 概述

IntelliScholar 使用 **LangGraph** 实现多智能体协作工作流。LangGraph 提供了状态图（StateGraph）机制，支持条件路由、状态管理和节点编排。

## 工作流架构

### 状态定义（State）

使用 `TypedDict` 定义工作流状态：

```python
class RecommendationState(TypedDict):
    user_id: str
    query: Optional[str]
    user_profile: Dict[str, Any]
    plan: Optional[Dict[str, Any]]
    online_search_results: List[Dict[str, Any]]
    offline_retrieval_results: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    filtered_candidates: List[Dict[str, Any]]
    ranked_papers: List[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    # ... 其他字段
```

### 节点（Nodes）

每个节点是一个异步函数，接收状态并返回更新后的状态：

1. **load_profile** - 加载用户画像
2. **planner** - Planner Agent 规划流程
3. **online_search** - 在线搜索节点（条件执行）
4. **offline_retrieval** - 离线检索节点
5. **recall** - 召回节点（整合多路召回）
6. **dedup_filter** - 去重过滤节点
7. **ranking** - 排序节点
8. **ui_format** - UI格式化节点
9. **update_profile** - 更新用户画像节点

### 边（Edges）和条件路由

工作流图结构：

```
START
  ↓
load_profile
  ↓
planner
  ↓
[条件路由: 是否需要在线搜索?]
  ├─ yes → online_search → offline_retrieval
  └─ no  → offline_retrieval
  ↓
recall
  ↓
[条件路由: 是否有候选?]
  ├─ yes → dedup_filter → ranking → ui_format → update_profile → END
  └─ no  → END
```

### 条件路由函数

```python
def _should_do_online_search(self, state: RecommendationState) -> str:
    """根据plan决定是否需要在线搜索"""
    plan = state.get("plan", {})
    if plan.get("online_search", False) and state.get("query"):
        return "yes"
    return "no"

def _has_candidates(self, state: RecommendationState) -> str:
    """判断是否有候选论文"""
    if state.get("candidates") and len(state["candidates"]) > 0:
        return "yes"
    return "no"
```

## 工作流执行

### 初始化

```python
workflow = RecommendationWorkflow()
# 在__init__中构建并编译图
self.graph = self._build_graph()
self.app = self.graph.compile()
```

### 执行工作流

```python
# 初始化状态
initial_state: RecommendationState = {
    "user_id": user_id,
    "query": query,
    # ... 其他字段
}

# 运行工作流
final_state = await self.app.ainvoke(initial_state)

# 获取结果
result = final_state["result"]
```

## LangGraph 优势

1. **状态管理**: 自动管理状态传递和更新
2. **条件路由**: 根据状态动态决定执行路径
3. **可视化**: 可以可视化工作流图
4. **可扩展**: 易于添加新节点和边
5. **错误处理**: 支持节点级别的错误处理

## 工作流可视化

可以使用 LangGraph Studio 或手动绘制工作流图：

```
                    START
                      ↓
              [load_profile]
                      ↓
                [planner]
                      ↓
         ┌────────────┴────────────┐
         │  should_do_online_search │
         └────────────┬────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
    [online_search]      [offline_retrieval]
          │                       │
          └───────────┬───────────┘
                      ↓
                  [recall]
                      ↓
         ┌────────────┴────────────┐
         │    has_candidates       │
         └────────────┬────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
    [dedup_filter]            [END]
          ↓
      [ranking]
          ↓
    [ui_format]
          ↓
  [update_profile]
          ↓
         END
```

## 扩展工作流

要添加新节点：

1. 定义节点函数：
```python
async def _new_node(self, state: RecommendationState) -> RecommendationState:
    # 节点逻辑
    return state
```

2. 添加到图：
```python
workflow.add_node("new_node", self._new_node)
```

3. 添加边：
```python
workflow.add_edge("previous_node", "new_node")
```

## 参考

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [StateGraph API](https://langchain-ai.github.io/langgraph/reference/graphs/)
