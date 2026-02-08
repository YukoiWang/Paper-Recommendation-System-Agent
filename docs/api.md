# IntelliScholar API 文档

## 基础URL

```
http://localhost:8000/api/v1
```

## 推荐接口

### 获取推荐

**POST** `/recommendations/`

请求体：
```json
{
  "user_id": "user_001",
  "query": "transformer attention",
  "top_n": 20,
  "recall_strategies": ["online_search", "rag", "itemcf"],
  "use_llm_rerank": true
}
```

响应：
```json
{
  "user_id": "user_001",
  "total": 20,
  "papers": [...],
  "user_profile": {...}
}
```

### 每日推荐

**GET** `/recommendations/daily/{user_id}?top_n=20`

## 用户画像接口

### 获取用户画像

**GET** `/users/{user_id}/profile`

### 提交用户反馈

**POST** `/users/{user_id}/feedback`

请求体：
```json
{
  "liked": ["paper_id_1", "paper_id_2"],
  "disliked": [],
  "read": ["paper_id_1"],
  "saved": []
}
```
