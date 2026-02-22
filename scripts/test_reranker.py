import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 配置 ---
MODEL_PATH = "./output/bge-finetuned"  # 训练后的模型路径
PAPER_META = "./paper_metadata.json"

class BGERerankerAgent:
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def compute_score(self, pairs):
        """
        计算 (Query, Doc) 对的相关性分数
        pairs: [['query', 'doc_text'], ['query', 'doc_text2']...]
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # Logits 即为相关性分数，维度 [batch, 1]
            scores = outputs.logits.view(-1).float()
            
            # 使用 Sigmoid 归一化到 0~1 便于人类理解 (可选，训练时用的 logits)
            scores = torch.sigmoid(scores)
            return scores.cpu().numpy()

    def rerank(self, query, candidate_ids, paper_db, top_k=5):
        """
        对外接口：输入 Query 和 候选 ID 列表，返回排序后的结果
        """
        # 1. 构造输入对
        pairs = []
        valid_cands = []
        
        for doc_id in candidate_ids:
            if doc_id not in paper_db: continue
            
            # 构造文档文本: Title + Abstract
            doc_meta = paper_db[doc_id]
            doc_text = f"{doc_meta.get('title', '')} [SEP] {doc_meta.get('abstract', '')}"
            
            pairs.append([query, doc_text])
            valid_cands.append(doc_id)

        if not pairs:
            return []

        # 2. 计算分数
        scores = self.compute_score(pairs)

        # 3. 排序
        # 将 (ID, Score) 打包并按 Score 降序排
        results = list(zip(valid_cands, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

# --- 测试代码 ---
if __name__ == "__main__":
    # 1. 加载模拟数据
    try:
        with open(PAPER_META, 'r') as f:
            paper_db = json.load(f)
    except:
        print("未找到 metadata，使用模拟数据")
        paper_db = {
            "doc1": {"title": "Deep Learning for NLP", "abstract": "Transformers are great."},
            "doc2": {"title": "Baking Bread 101", "abstract": "Flour and water mix."},
            "doc3": {"title": "Attention is All You Need", "abstract": "Sequence to sequence modeling."}
        }

    # 2. 初始化 Agent
    agent = BGERerankerAgent(MODEL_PATH)

    # 3. 模拟一个 User Persona Query
    query = "Persona: A Researcher in AI. Query: papers about transformer architecture"
    candidates = ["doc1", "doc2", "doc3"]

    print("\n--------------------------------")
    print(f"Query: {query}")
    print("Candidates:", candidates)
    print("--------------------------------")

    # 4. 执行重排序
    ranked_results = agent.rerank(query, candidates, paper_db)

    print("Ranked Results:")
    for rank, (doc_id, score) in enumerate(ranked_results, 1):
        title = paper_db[doc_id]['title']
        print(f"#{rank} | Score: {score:.4f} | ID: {doc_id} | Title: {title}")