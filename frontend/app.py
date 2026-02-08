"""
IntelliScholar Frontend - Streamlit App
"""
import streamlit as st
import requests
from typing import List, Dict, Any
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="IntelliScholar - 智能论文推荐",
    page_icon="📚",
    layout="wide"
)

st.title("📚 IntelliScholar - 智能论文推荐系统")

# Sidebar
with st.sidebar:
    st.header("用户设置")
    user_id = st.text_input("用户ID", value="user_001")
    
    st.header("推荐设置")
    query = st.text_input("搜索查询（可选）", placeholder="例如: transformer attention mechanism")
    top_n = st.slider("推荐数量", min_value=5, max_value=50, value=20)
    
    recall_strategies = st.multiselect(
        "召回策略",
        options=["online_search", "rag", "itemcf"],
        default=["rag", "itemcf"]
    )
    
    use_llm_rerank = st.checkbox("使用LLM语义重排", value=True)
    
    if st.button("获取推荐", type="primary"):
        st.session_state.get_recommendations = True


# Main content
if st.session_state.get("get_recommendations", False):
    with st.spinner("正在生成推荐..."):
        try:
            # Call API
            response = requests.post(
                f"{API_BASE_URL}/recommendations/",
                json={
                    "user_id": user_id,
                    "query": query if query else None,
                    "top_n": top_n,
                    "recall_strategies": recall_strategies,
                    "use_llm_rerank": use_llm_rerank
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            st.session_state.recommendations = result
            st.session_state.get_recommendations = False
            
        except Exception as e:
            st.error(f"获取推荐失败: {str(e)}")
            st.session_state.get_recommendations = False

# Display recommendations
if "recommendations" in st.session_state:
    result = st.session_state.recommendations
    
    # User profile
    with st.expander("用户画像", expanded=False):
        profile = result.get("user_profile", {})
        st.write(f"**兴趣**: {', '.join(profile.get('interests', []))}")
        st.write(f"**画像摘要**: {profile.get('profile_summary', 'N/A')}")
        st.write(f"**最近阅读**: {len(profile.get('recent_reads', []))} 篇")
    
    # Papers
    st.header(f"推荐论文 ({result.get('total', 0)} 篇)")
    
    papers = result.get("papers", [])
    for i, paper in enumerate(papers, 1):
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.subheader(f"{i}. {paper.get('title', 'N/A')}")
                
                authors = paper.get("authors", [])
                if authors:
                    st.write(f"**作者**: {', '.join(authors[:5])}")
                
                venue = paper.get("venue", "")
                year = paper.get("year", "")
                if venue or year:
                    st.write(f"**发表**: {venue} {year}")
                
                abstract = paper.get("abstract", "")
                if abstract:
                    st.write(f"**摘要**: {abstract[:300]}...")
                
                # Scores
                scores = paper.get("scores", {})
                if scores:
                    col_score1, col_score2, col_score3 = st.columns(3)
                    with col_score1:
                        st.metric("最终分数", f"{scores.get('final_score', 0):.3f}")
                    with col_score2:
                        st.metric("模型分数", f"{scores.get('model_score', 0):.3f}")
                    with col_score3:
                        st.metric("召回来源", paper.get("recall_source", "unknown"))
            
            with col2:
                paper_id = paper.get("paper_id", "")
                url = paper.get("url", "")
                
                if url:
                    st.link_button("查看论文", url)
                else:
                    st.write(f"ID: {paper_id}")
            
            st.divider()

else:
    st.info("👈 请在左侧设置用户ID和推荐参数，然后点击「获取推荐」按钮")
