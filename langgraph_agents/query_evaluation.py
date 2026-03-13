import json
import csv
import os
import asyncio
import pandas as pd
from dotenv import load_dotenv

try:
    from planner_agent import PlannerAgent
except ImportError:
    print("❌ 导入失败！请确保你在 intellischolar 根目录下运行，且文件名为 planner_agent.py")
    exit(1)

# 加载环境变量
load_dotenv()

async def evaluate_pipeline():
    # 1. 确保获取 API KEY
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误: 未能读取到 OPENAI_API_KEY，请检查 .env 文件！")
        return

    print("🚀 正在初始化 Planner Agent 和 GPT 模型...")
    # 2. 初始化 Planner
    planner = PlannerAgent(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model="gpt-5.1"
    )

    # 【重要修复 1】强制初始化 LLM，确保独立函数 (如 decomposition) 可以直接调用大模型
    # _ensure_llm 是队友代码里写的按需加载方法
    planner._ensure_llm({"api_key": api_key})

    results = []
    
    # 3. 读取测试用例 
    if not os.path.exists("/Users/bb/Paper-Recommendation-System-Agent/langgraph_agents/test_cases.txt"):
        print("❌ 错误: 找不到 test_cases.txt，请确保在同目录下创建了该测试文件。")
        return

    with open("/Users/bb/Paper-Recommendation-System-Agent/langgraph_agents/test_cases.txt", "r", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f if line.strip()]

    print(f"✅ 环境就绪！开始评估 {len(test_cases)} 条用例...\n")

    for case in test_cases:
        category = case.get("category", "unknown")
        user_query = case.get("input", "")
        context = case.get("context", {})
        expected = case.get("expected", "")
        
        row = {"query": user_query, "category": category, "is_pass": False, "output": ""}

        try:
            # ==========================================
            # 1. 错误修正与过滤 (Correction)
            # ==========================================
            if category == "correction":
                res = planner.filter_and_correct_query(user_query)
                is_valid = res.get("is_valid", True)
                corrected = res.get("corrected_query", "")
                
                if not is_valid:
                    row["output"] = "INVALID"
                    row["is_pass"] = (expected == "invalid")
                else:
                    row["output"] = corrected
                    # 检查 expected（如 "latest" 或 "BERT"）是否包含在输出中
                    if expected != "invalid" and any(e in corrected.lower() for e in str(expected).lower().split("/")):
                         row["is_pass"] = True
                    else:
                         row["is_pass"] = False

            # ==========================================
            # 2. 多轮对话融合 (Context Fusion)
            # ==========================================
            elif category == "fusion":
                history = context.get("history", [])
                res = planner.fuse_context(user_query, history=history, cited_papers={})
                row["output"] = res.get("enhanced_query", "")
                row["is_pass"] = str(expected).lower() in row["output"].lower()

            # ==========================================
            # 3. 学术词扩充 (Query Expansion)
            # ==========================================
            elif category == "expansion":
                res = planner.expand_terms(user_query)
                row["output"] = res.get("expanded_query", "")
                
                if expected == "no_fruit_expansion":
                    # 反例测试：Apple 不应该扩展出 fruit
                    row["is_pass"] = "fruit" not in row["output"].lower()
                else:
                    row["is_pass"] = str(expected).lower() in row["output"].lower()

            # ==========================================
            # 4. 父查询生成 (Parent Query Generation)
            # ==========================================
            elif category == "parent":
                res = planner.generate_parent_query(user_query)
                row["output"] = res.get("parent_query", "")
                # 判断标准：变长了，且包含了我们期望的核心实体
                row["is_pass"] = len(row["output"]) > len(user_query) and str(expected).lower() in row["output"].lower()

            # ==========================================
            # 5. 复杂问题拆解 (Sub-query Decomposition)
            # ==========================================
            elif category == "decomposition":
                res = planner.decompose_sub_queries(user_query)
                subs = res.get("sub_queries", [])
                is_complex = res.get("is_complex", False)
                
                # 【重要修复 2】如果为空，标记为 EMPTY_RESULT 而不是 NaN
                row["output"] = "|".join(subs) if subs else "EMPTY_RESULT"
                
                if expected == "multiple_queries":
                    row["is_pass"] = is_complex and len(subs) > 1
                else:
                    row["is_pass"] = not is_complex

            # ==========================================
            # 6. 检索决策评估 (Decision Routing)
            # ==========================================
            elif category == "decision":
                state = {
                    "user_query": user_query, 
                    "history": context.get("history", []), 
                    "api_key": api_key
                }
                updated_state = planner.plan(state)
                decision = updated_state.get("planner_decision", {})
                
                # 【重要修复 3】容错处理：使用 .get 防止 KeyError
                route = decision.get("route", "NO_RETRIEVAL")
                actual_retrieval = (route == "RETRIEVE_LOCAL")
                actual_online = decision.get("do_online_search", False) 
                
                row["output"] = f"R:{actual_retrieval}, O:{actual_online}"
                
                expected_retrieval = case.get("expected_retrieval")
                expected_online = case.get("expected_online")
                row["is_pass"] = (actual_retrieval == expected_retrieval) and (actual_online == expected_online)

            else:
                row["output"] = "Unrecognized Category"
                row["is_pass"] = False

        except Exception as e:
            # 捕获任何报错，写入 CSV，保证程序不崩
            row["output"] = f"Error: {str(e)}"
            row["is_pass"] = False

        results.append(row)
        print(f"[{category.upper()}] 处理完毕: {user_query[:30]}")

    # 4. 生成 CSV 报表
    keys = ["query", "category", "is_pass", "output"]
    csv_filename = "/Users/bb/Paper-Recommendation-System-Agent/langgraph_agents/evaluation_results_v2.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8-sig") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    # 5. 打印统计数据
    df = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print("🏆 IntelliScholar 核心大脑评估报告")
    print("="*40)
    
    # 统计通过率
    if not df.empty:
        summary = df.groupby("category")["is_pass"].mean() * 100
        print("\n各维度得分 (准确率 %):")
        for cat, score in summary.items():
            print(f" - {cat.ljust(15)}: {score:.1f}%")
            
        print(f"\n整体准确率: {df['is_pass'].mean() * 100:.1f}%")
    
    print(f"\n👉 详细 Bad Case 请查看生成的: {csv_filename}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(evaluate_pipeline())