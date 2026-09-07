from agent.models import Paper, UserProfile
from langgraph_agents.critic_agent import run_critic
from langgraph_agents.harness.eval import rule_grade, salvage_rate
from langgraph_agents.harness.replay import StubTools, replay_cascade, replay_routing
from langgraph_agents.intent.classifier import classify_intent, extract_compare_entities
from langgraph_agents.planner_runtime import build_work_order
from langgraph_agents.recommend_runtime import run_recommend
from langgraph_agents.researcher_agent import run_researcher
from langgraph_agents.writer_agent import run_writer


def test_rule_chitchat():
    intent, src, _ = classify_intent("Hello", embedder=None, llm_call=None)
    assert intent == "chitchat" and src == "rule"


def test_rule_recommend():
    intent, src, _ = classify_intent("recommend papers on RAG", embedder=None, llm_call=None)
    assert intent == "recommend" and src == "rule"


def test_rule_explain():
    intent, src, _ = classify_intent("讲讲注意力机制是怎么演进的", embedder=None, llm_call=None)
    assert intent == "explain" and src == "rule"


def test_rule_compare():
    intent, src, _ = classify_intent("Transformer vs Mamba 有什么区别", embedder=None, llm_call=None)
    assert intent == "compare" and src == "rule"


def test_topic_only_not_rule_recommend():
    intent, src, _ = classify_intent("RAG 论文", embedder=None, llm_call=None)
    assert not (intent == "recommend" and src == "rule")


def test_followup_needs_cited():
    intent, src, _ = classify_intent("第二篇展开讲一下", has_cited=True, embedder=None, llm_call=None)
    assert intent == "followup"


def test_more_like_not_followup():
    intent, src, _ = classify_intent("再找几篇类似的", has_cited=True, embedder=None, llm_call=None)
    assert intent != "followup"


def test_compare_entities():
    ents = extract_compare_entities("Compare ViT and CNN architectures")
    assert len(ents) >= 2


def test_work_order_next_recommend():
    wo = build_work_order("recommend papers on LoRA", embedder=None, llm_call=None)
    assert wo.intent == "recommend"
    assert wo.next_agent == "recommend"


def test_feedback_repush_goes_recommend():
    wo = build_work_order("换一批", last_was_list=True, embedder=None, llm_call=None)
    assert wo.intent == "feedback"
    assert wo.next_agent == "recommend"


def test_compare_missing_slot():
    wo = build_work_order("对比一下 Transformer", embedder=None, llm_call=None)
    assert wo.intent == "compare"
    assert wo.next_agent == "writer"
    assert "compare_entities" in wo.missing_slots


def test_survey_sub_queries():
    wo = build_work_order("帮我做 LLM agent 综述", embedder=None, llm_call=None)
    assert wo.intent == "survey"
    assert wo.next_agent == "researcher"
    assert wo.slots.get("sub_queries")


def test_researcher_empty_local_switches_arxiv():
    tools = StubTools(local_n=0, arxiv_n=4)
    state = {
        "user_query": "讲讲 transformers",
        "work_order": {
            "intent": "explain",
            "topic": "transformers",
            "playbook_id": "explain_v1",
            "need_sota": False,
            "budget": {"max_steps": 6, "max_search": 4},
            "success_criteria": {"type": "explain", "min_papers": 1},
            "entities": {"methods": []},
        },
        "top_k": 5,
    }
    out = run_researcher(state, tools)
    assert out.get("recovery") == "switch_arxiv"
    assert out.get("final_papers")
    assert any(c.startswith("search_arxiv") for c in tools.calls)


def test_researcher_local_sufficient_no_arxiv():
    tools = StubTools(local_n=8, arxiv_n=4)
    state = {
        "user_query": "讲讲 transformers",
        "work_order": {
            "intent": "explain",
            "topic": "transformers",
            "playbook_id": "explain_v1",
            "need_sota": False,
            "budget": {"max_steps": 6, "max_search": 4},
            "success_criteria": {"type": "explain", "min_papers": 1},
            "entities": {"methods": []},
        },
        "top_k": 5,
    }
    out = run_researcher(state, tools)
    assert out.get("recovery") != "switch_arxiv"
    assert not any(c.startswith("search_arxiv") for c in tools.calls)


def test_compare_splits_entities():
    tools = StubTools(local_n=2, arxiv_n=0)
    state = {
        "user_query": "Transformer vs Mamba",
        "work_order": {
            "intent": "compare",
            "topic": "Transformer vs Mamba",
            "playbook_id": "compare_v1",
            "entities": {"methods": ["Transformer", "Mamba"]},
            "budget": {"max_steps": 8, "max_search": 4},
            "success_criteria": {"type": "compare", "min_papers_per_entity": 1},
        },
        "top_k": 5,
    }
    out = run_researcher(state, tools)
    joined = " ".join(tools.calls)
    assert "search_local" in joined
    assert "Transformer vs Mamba" not in "".join(c for c in tools.calls)


def test_recommend_cascade():
    tools = StubTools(local_n=0, arxiv_n=3)
    state = {
        "user_query": "recommend papers on LoRA",
        "work_order": {"intent": "recommend", "topic": "LoRA", "need_sota": False},
        "top_k": 5,
        "user_profile": UserProfile(user_id="u1", interest_text="ML"),
    }
    out = run_recommend(state, tools)
    assert out.get("recovery") == "switch_arxiv"
    assert out.get("final_papers")


def test_critic_citation_oob_rewrite():
    papers = [Paper(paper_id="a", title="A", abstract="a")]
    st = run_critic({
        "response": "See [9] for details.",
        "final_papers": papers,
        "work_order": {"intent": "explain", "missing_slots": []},
        "evidence_pack": {"papers": [{"title": "A"}]},
    })
    assert st["critic_decision"] == "rewrite"
    assert st["after_critic"] == "writer"


def test_writer_clarify():
    st = run_writer({
        "user_query": "对比一下",
        "work_order": {"intent": "compare", "missing_slots": ["compare_entities"]},
        "history": [],
    }, qa=None)
    assert "两边" in st["response"]


def test_rule_grade_cascade_miss():
    flags = rule_grade({
        "intent": "explain",
        "failure_type": "empty_retrieval",
        "tool_calls": ["search_local n=0"],
    })
    assert "cascade_miss" in flags


def test_salvage_rate():
    stats = salvage_rate([
        {"failure_type": "empty_retrieval", "recovery": "switch_arxiv", "tool_calls": ["search_local n=0", "search_arxiv n=4"]},
        {"failure_type": "empty_retrieval", "recovery": "", "tool_calls": ["search_local n=0"]},
    ])
    assert stats["empty_first"] == 2
    assert stats["saved"] == 1
    assert stats["salvage_rate"] == 0.5


def test_golden_routing():
    report = replay_routing()
    failed = [r for r in report["results"] if not r["ok"]]
    assert not failed, failed


def test_golden_cascade():
    report = replay_cascade()
    failed = [r for r in report["results"] if not r["ok"]]
    assert not failed, failed
