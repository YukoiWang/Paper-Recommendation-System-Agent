"""Planner Agent: central decision-maker for the multi-agent paper QA system.

Three core capabilities:
  1. Query understanding & optimization — refine user query for better retrieval
  2. Retrieval decision & routing — decide NO_RETRIEVAL / RETRIEVE_LOCAL / NEED_CLARIFY
  3. Retrieval result evaluation — assess whether retrieved chunks are sufficient

Uses LLM-based structured reasoning via prompt templates.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route labels (standardized, used by workflow routing)
# ---------------------------------------------------------------------------
ROUTE_NO_RETRIEVAL = "NO_RETRIEVAL"
ROUTE_RETRIEVE_LOCAL = "RETRIEVE_LOCAL"
ROUTE_NEED_CLARIFY = "NEED_CLARIFY"

ALL_ROUTES = {ROUTE_NO_RETRIEVAL, ROUTE_RETRIEVE_LOCAL, ROUTE_NEED_CLARIFY}

# Keep legacy aliases so existing workflow code still compiles during migration
ROUTE_ASK_PROFILE = ROUTE_NEED_CLARIFY
ROUTE_HANDLE_FEEDBACK = "HANDLE_FEEDBACK"
ROUTE_RETRIEVAL = ROUTE_RETRIEVE_LOCAL
ROUTE_RECALL = ROUTE_RETRIEVE_LOCAL
ROUTE_RESPOND = ROUTE_NO_RETRIEVAL

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

QUERY_OPTIMIZATION_PROMPT = """\
You are a query optimizer for an academic paper retrieval system.

Given the user's raw query and the conversation state, produce an optimized \
retrieval query that is concise, keyword-rich, and likely to retrieve relevant \
ML/AI papers from a vector database.

## Rules
- Extract core technical terms, method names, dataset names, task names.
- Remove filler words, greetings, and conversational noise.
- If the conversation state reveals the user's true intent or narrows the scope, \
incorporate that context.
- Output a SINGLE optimized query string (1-3 sentences max).
- If the original query is already precise, return it as-is.
- Respond in the SAME LANGUAGE as the user's query.

## Input
User query: {user_query}
Conversation state: {conversation_state}

## Output
Return ONLY the optimized query string, nothing else."""


ROUTE_DECISION_PROMPT = """\
You are the routing planner for a multi-agent paper Q&A system. \
Based on the user's query and conversation state, decide the next action.

## Available Routes
- NO_RETRIEVAL: The query can be answered from existing conversation context \
(e.g. follow-up about already-discussed papers, greetings, meta-questions).
- RETRIEVE_LOCAL: The query requires searching the local paper knowledge base \
(e.g. paper recommendations, technical questions, comparisons).
- NEED_CLARIFY: The query is too vague or ambiguous to proceed; \
the system should ask the user for clarification \
(e.g. "recommend papers" with no topic, unintelligible input).

## Decision Criteria
1. If the user is greeting, chatting, or asking about something already in \
conversation context → NO_RETRIEVAL
2. If the user asks about papers, methods, comparisons, recommendations, \
or any topic that needs paper evidence → RETRIEVE_LOCAL
3. If the query is too vague, missing critical information, or ambiguous \
→ NEED_CLARIFY
4. When in doubt, prefer RETRIEVE_LOCAL over NO_RETRIEVAL.

## Response Style
In addition to routing, decide HOW the response should be structured:
- "recommend": The user explicitly wants paper recommendations or a list of papers \
(e.g. "推荐几篇RAG论文", "find me papers on attention", "有什么好的NLP论文"). \
Papers are the main output, listed one by one.
- "narrative": The user wants knowledge, explanation, analysis, comparison, or \
historical overview of a topic, with papers serving as supporting evidence \
(e.g. "讲讲VLM的发展", "compare ViT and CNN", "总结attention机制的演进", \
"what are the key challenges in RLHF"). Content is the main output, papers are \
cited inline as [N].

## Input
User query: {user_query}
Optimized query: {optimized_query}
Conversation state: {conversation_state}
Papers already in context: {has_paper_context}

## Output
Return valid JSON only — no markdown fences, no extra text:
{{
  "route": "NO_RETRIEVAL | RETRIEVE_LOCAL | NEED_CLARIFY",
  "response_style": "recommend | narrative",
  "reasoning": "brief explanation of your decision (1-2 sentences)"
}}"""


RETRIEVAL_EVALUATION_PROMPT = """\
You are a retrieval quality evaluator for an academic paper Q&A system. \
Assess whether the retrieved paper snippets are sufficient to answer the user's query.

## Evaluation Criteria
- SUFFICIENT: The retrieved papers contain enough relevant information to \
produce a meaningful answer. At least some papers are directly relevant.
- INSUFFICIENT: The retrieved papers are off-topic, too few, or lack the \
specific information needed. Consider suggesting a refined query.
- PARTIAL: Some relevant papers exist but more context might improve the answer. \
Proceed with what's available but note the gap.

## Input
User query: {user_query}
Optimized query: {optimized_query}
Number of retrieved papers: {num_papers}
Retrieved paper titles and abstracts (truncated):
{retrieval_summary}

## Output
Return valid JSON only — no markdown fences, no extra text:
{{
  "quality": "SUFFICIENT | INSUFFICIENT | PARTIAL",
  "reasoning": "brief explanation (1-2 sentences)",
  "suggested_refined_query": "a better query if INSUFFICIENT, else null"
}}"""

# ---------------------------------------------------------------------------
# Query Pipeline Prompt Templates
# ---------------------------------------------------------------------------

QUERY_FILTER_PROMPT = """\
You are a query validator for an academic paper retrieval system.
Analyze the user's query and perform filtering & correction.

## Rules
1. Mark as INVALID if the query is:
   - Meaningless gibberish / random characters / keyboard smashes
   - Fewer than 2 meaningful tokens after removing punctuation
   - Pure emoji or symbols with no semantic content
2. Correct factual/temporal errors:
   - Year references outside 2018–2026 should be replaced with "recent/latest"
     (e.g. "transformers invented in 2030" → "latest transformer papers")
   - Contradictory claims (e.g. "CNN is a recurrent model") → fix the contradiction
   - Obvious typos in well-known terms (e.g. "BERT" mistyped as "BRET") → correct them
3. Preserve the user's original language (Chinese / English / mixed).
4. Keep the semantic intent intact; only fix what is clearly wrong.

## Input
User query: {user_query}

## Output
Return valid JSON only — no markdown fences, no extra text:
{{
  "is_valid": true/false,
  "corrected_query": "the corrected query string (same as original if no correction needed)",
  "corrections": ["list of corrections applied, empty if none"]
}}"""


CONTEXT_FUSION_PROMPT = """\
You are a context-aware query enhancer for an academic paper Q&A system.
Fuse the current user query with relevant context from the conversation.

## Available Context
- Conversation history (recent turns): {history_summary}
- Previously cited/discussed papers: {cited_papers_summary}
- Historical keywords from past queries: {historical_keywords}

## Rules
1. Identify the core intent of the current query.
2. From the context above, extract ONLY information that is directly relevant \
to the current query (topic continuity, referenced entities, narrowing scope).
3. Produce a single enhanced query that naturally incorporates the relevant context.
4. Do NOT simply concatenate everything — be selective and concise.
5. If the current query is self-contained (new topic), return it mostly unchanged.
6. Respond in the SAME LANGUAGE as the user's query.

## Input
Current query: {current_query}

## Output
Return valid JSON only — no markdown fences, no extra text:
{{
  "enhanced_query": "the context-enhanced query string",
  "fused_context": "brief note on what context was incorporated (1 sentence)"
}}"""


TERM_EXPANSION_PROMPT = """\
You are an academic terminology expert for ML/AI/NLP/CV research.
Given a query, expand it with closely related academic synonyms and abbreviations.

## Rules
1. Add 3–5 highly relevant alternative terms (synonyms, abbreviations, \
related sub-field terms).
2. Prioritize terms that would appear in paper titles/abstracts.
3. Do NOT add loosely related or off-topic terms.
4. Keep the expanded query concise — append terms in parentheses or as a \
short comma-separated suffix.
5. Respond in the SAME LANGUAGE as the original query; for Chinese queries, \
also add the English equivalents of key terms.

## Examples
- "routing in multi-agent systems" → \
"routing in multi-agent systems (multi-agent routing, intent routing, \
symbolic routing, agent orchestration)"
- "知识蒸馏" → "知识蒸馏 (knowledge distillation, model compression, \
teacher-student learning)"

## Input
Query: {query}

## Output
Return valid JSON only — no markdown fences, no extra text:
{{
  "expanded_query": "the query with appended expansion terms",
  "added_terms": ["term1", "term2", "term3"]
}}"""


PARENT_QUERY_PROMPT = """\
You are a query intent resolver for an academic paper retrieval system.
The user's query is vague, short, or ambiguous. Infer the most likely core intent \
and generate a precise "parent query" suitable for direct retrieval.

## Rules
1. Analyze what the user most likely wants based on the query and any context.
2. Generate ONE clear, specific parent query that captures the core research intent.
3. The parent query should be directly usable for vector search — keyword-rich, \
specific, 1–2 sentences.
4. Do NOT ask the user for clarification — your job is to infer and resolve.
5. Respond in the SAME LANGUAGE as the user's query.

## Input
User query: {user_query}
Conversation context (if any): {context_summary}

## Output
Return valid JSON only — no markdown fences, no extra text:
{{
  "parent_query": "the inferred precise parent query",
  "inferred_intent": "brief explanation of what you think the user wants (1 sentence)"
}}"""


SUB_QUERY_DECOMPOSE_PROMPT = """\
You are a query decomposition expert for an academic paper retrieval system.
The user's query is complex (contains comparisons, multiple aspects, or \
multi-step reasoning). Break it into atomic sub-questions.

## Complexity Signals (any of these → decompose)
- Comparison words: "compare", "vs", "difference", "对比", "区别", "差异"
- Enumeration: "which ones", "what are", "哪些", "列举"
- Multi-aspect: "and", "以及", "和", "同时"
- Process: "how to", "steps", "怎么", "步骤", "流程"

## Rules
1. Decompose into 2–4 atomic sub-questions, each answerable by a single retrieval pass.
2. Each sub-question should be self-contained and specific.
3. Maintain the logical order (e.g. for "compare A and B": first retrieve A, then B).
4. If the query is NOT actually complex, return an empty sub_queries list.
5. Respond in the SAME LANGUAGE as the user's query.

## Input
User query: {user_query}

## Output
Return valid JSON only — no markdown fences, no extra text:
{{
  "is_complex": true/false,
  "sub_queries": ["sub-question 1", "sub-question 2", ...],
  "decomposition_rationale": "brief explanation (1 sentence)"
}}"""


# ---------------------------------------------------------------------------
# LLM client (lightweight, same interface as qa_agent.LLMClient)
# ---------------------------------------------------------------------------

class _PlannerLLM:
    """Thin LLM wrapper for planner-specific calls (low temperature, concise)."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", timeout: float = 30.0):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self._total_tokens = 0

    def call(self, prompt: str, temperature: float = 0.1,
             max_tokens: int = 512) -> str:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        if resp.usage:
            self._total_tokens += resp.usage.total_tokens
        logger.debug("PlannerLLM %.1fs tokens=%d", time.time() - t0, self._total_tokens)
        return text


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class PlannerAgent:
    """Central decision & routing module for the multi-agent paper QA system.

    Three core methods:
      1. optimize_query  — refine raw user query for retrieval
      2. decide_route    — decide NO_RETRIEVAL / RETRIEVE_LOCAL / NEED_CLARIFY
      3. evaluate_retrieval — assess if retrieval results suffice

    Main entry point: plan(state) → updated state with planner_decision.
    """

    def __init__(self, api_key: str = "", base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", max_retries: int = 1):
        self._llm: Optional[_PlannerLLM] = None
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._max_retries = max_retries
        if api_key:
            self._llm = _PlannerLLM(api_key=api_key, base_url=base_url, model=model)

    def _ensure_llm(self, state: Dict[str, Any]) -> bool:
        """Lazy-init LLM from state if not already initialized."""
        if self._llm is not None:
            return True
        key = state.get("api_key") or self._api_key
        if not key:
            logger.warning("PlannerAgent: no API key available, using fallback rules")
            return False
        self._llm = _PlannerLLM(api_key=key, base_url=self._base_url, model=self._model)
        return True

    # -----------------------------------------------------------------
    # Core capability 1: Query understanding & optimization
    # -----------------------------------------------------------------

    def optimize_query(self, user_query: str,
                       conversation_state: Optional[Dict] = None) -> str:
        """Refine raw user query into an optimized retrieval query."""
        if not self._llm or not user_query.strip():
            return user_query

        conv_summary = json.dumps(conversation_state or {}, ensure_ascii=False, default=str)
        prompt = QUERY_OPTIMIZATION_PROMPT.format(
            user_query=user_query,
            conversation_state=conv_summary,
        )
        try:
            optimized = self._llm.call(prompt, temperature=0.1, max_tokens=256)
            if optimized and len(optimized) > 3:
                logger.info("Query optimized: '%s' → '%s'", user_query[:60], optimized[:60])
                return optimized
        except Exception as e:
            logger.warning("Query optimization failed: %s", e)
        return user_query

    # -----------------------------------------------------------------
    # Core capability 2: Retrieval decision & routing
    # -----------------------------------------------------------------

    def decide_route(self, user_query: str, optimized_query: str,
                     conversation_state: Optional[Dict] = None,
                     has_paper_context: bool = False) -> Dict[str, str]:
        """Decide routing: NO_RETRIEVAL / RETRIEVE_LOCAL / NEED_CLARIFY."""
        if not self._llm:
            return self._fallback_route(user_query, has_paper_context)

        conv_summary = json.dumps(conversation_state or {}, ensure_ascii=False, default=str)
        prompt = ROUTE_DECISION_PROMPT.format(
            user_query=user_query,
            optimized_query=optimized_query,
            conversation_state=conv_summary,
            has_paper_context=has_paper_context,
        )
        try:
            raw = self._llm.call(prompt, temperature=0.1, max_tokens=256)
            decision = self._parse_json(raw)
            route = decision.get("route", "").upper().strip()
            if route not in ALL_ROUTES:
                logger.warning("LLM returned invalid route '%s', defaulting to RETRIEVE_LOCAL", route)
                route = ROUTE_RETRIEVE_LOCAL
            style = decision.get("response_style", "recommend").lower().strip()
            if style not in ("recommend", "narrative"):
                style = "recommend"
            return {
                "route": route,
                "response_style": style,
                "reasoning": decision.get("reasoning", ""),
            }
        except Exception as e:
            logger.warning("Route decision failed: %s", e)
            return self._fallback_route(user_query, has_paper_context)

    # -----------------------------------------------------------------
    # Core capability 3: Retrieval result evaluation
    # -----------------------------------------------------------------

    def evaluate_retrieval(self, user_query: str, optimized_query: str,
                           retrieved_papers: List[Any]) -> Dict[str, Any]:
        """Evaluate whether retrieval results are sufficient for answering."""
        if not self._llm or not retrieved_papers:
            quality = "INSUFFICIENT" if not retrieved_papers else "SUFFICIENT"
            return {
                "quality": quality,
                "reasoning": "No LLM available" if not self._llm else "Fallback evaluation",
                "suggested_refined_query": None,
            }

        summary_parts = []
        for i, p in enumerate(retrieved_papers[:10]):
            title = getattr(p, "title", str(p)) if not isinstance(p, dict) else p.get("title", "")
            abstract = getattr(p, "abstract", "") if not isinstance(p, dict) else p.get("abstract", "")
            abstract_trunc = abstract[:200] + "..." if len(abstract) > 200 else abstract
            summary_parts.append(f"[{i+1}] {title}\n     {abstract_trunc}")
        retrieval_summary = "\n".join(summary_parts) if summary_parts else "(empty)"

        prompt = RETRIEVAL_EVALUATION_PROMPT.format(
            user_query=user_query,
            optimized_query=optimized_query,
            num_papers=len(retrieved_papers),
            retrieval_summary=retrieval_summary,
        )
        try:
            raw = self._llm.call(prompt, temperature=0.1, max_tokens=256)
            result = self._parse_json(raw)
            quality = result.get("quality", "SUFFICIENT").upper().strip()
            if quality not in {"SUFFICIENT", "INSUFFICIENT", "PARTIAL"}:
                quality = "SUFFICIENT"
            return {
                "quality": quality,
                "reasoning": result.get("reasoning", ""),
                "suggested_refined_query": result.get("suggested_refined_query"),
            }
        except Exception as e:
            logger.warning("Retrieval evaluation failed: %s", e)
            return {
                "quality": "SUFFICIENT" if retrieved_papers else "INSUFFICIENT",
                "reasoning": f"Evaluation error: {e}",
                "suggested_refined_query": None,
            }

    # -----------------------------------------------------------------
    # Query Pipeline Step 1: Filter & Correct
    # -----------------------------------------------------------------

    def filter_and_correct_query(self, user_query: str) -> Dict[str, Any]:
        """Filter invalid queries and correct factual/temporal errors.

        Returns dict with keys: is_valid, corrected_query, corrections.
        """
        q = user_query.strip()
        if not q:
            return {"is_valid": False, "corrected_query": "", "corrections": ["empty query"]}

        meaningful_tokens = re.findall(r'[\w\u4e00-\u9fff]+', q)
        if len(meaningful_tokens) < 2 and len(q) < 4:
            return {"is_valid": False, "corrected_query": q, "corrections": ["too short / meaningless"]}

        alpha_ratio = sum(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in q) / max(len(q), 1)
        if alpha_ratio < 0.3:
            return {"is_valid": False, "corrected_query": q, "corrections": ["mostly non-alphanumeric / gibberish"]}

        if self._has_clear_intent(q):
            corrected = self._rule_based_year_fix(q)
            changed = corrected != q
            return {
                "is_valid": True,
                "corrected_query": corrected,
                "corrections": ["year corrected by rule"] if changed else [],
            }

        if not self._llm:
            corrected = self._rule_based_year_fix(q)
            changed = corrected != q
            return {
                "is_valid": True,
                "corrected_query": corrected,
                "corrections": ["year corrected by rule"] if changed else [],
            }

        prompt = QUERY_FILTER_PROMPT.format(user_query=q)
        try:
            raw = self._llm.call(prompt, temperature=0.05, max_tokens=256)
            result = self._parse_json(raw)
            is_valid = result.get("is_valid", True)
            if not is_valid and len(meaningful_tokens) >= 3:
                logger.info("LLM flagged query as invalid but has %d tokens; overriding to valid",
                            len(meaningful_tokens))
                is_valid = True
            return {
                "is_valid": is_valid,
                "corrected_query": result.get("corrected_query", q),
                "corrections": result.get("corrections", []),
            }
        except Exception as e:
            logger.warning("filter_and_correct_query LLM failed: %s", e)
            corrected = self._rule_based_year_fix(q)
            return {"is_valid": True, "corrected_query": corrected, "corrections": []}

    @staticmethod
    def _has_clear_intent(query: str) -> bool:
        """Check if query contains clear intent signals (search, recommend, etc.)
        that should never be filtered as invalid."""
        q = query.lower()
        intent_signals = [
            "recommend", "search", "find", "compare", "explain", "summarize",
            "latest", "newest", "recent", "sota",
            "推荐", "搜", "找", "对比", "比较", "解释", "总结", "介绍",
            "最新", "最近", "前沿", "上网", "联网", "在线",
            "paper", "论文", "method", "model", "方法", "模型",
        ]
        return any(sig in q for sig in intent_signals)

    @staticmethod
    def _rule_based_year_fix(query: str) -> str:
        """Replace out-of-range year references (outside 2018-2026) with '最新/latest'."""
        def _replace(m: re.Match) -> str:
            year = int(m.group())
            if 2018 <= year <= 2026:
                return m.group()
            return "latest"
        return re.sub(r'\b(19|20)\d{2}\b', _replace, query)

    # -----------------------------------------------------------------
    # Query Pipeline Step 2: Multi-turn Context Fusion
    # -----------------------------------------------------------------

    def fuse_context(self, current_query: str, history: List[Dict],
                     cited_papers: Dict[str, Any],
                     conversation_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Fuse conversation context into the current query.

        Returns dict with keys: enhanced_query, fused_context.
        """
        if not self._llm:
            return {"enhanced_query": current_query, "fused_context": "no LLM available"}

        history_summary = self._summarize_history(history, max_turns=6)
        cited_summary = self._summarize_cited_papers(cited_papers, max_papers=5)
        historical_kw = self._extract_historical_keywords(history, conversation_state)

        if not history_summary and not cited_summary and not historical_kw:
            return {"enhanced_query": current_query, "fused_context": "no prior context"}

        prompt = CONTEXT_FUSION_PROMPT.format(
            history_summary=history_summary or "(none)",
            cited_papers_summary=cited_summary or "(none)",
            historical_keywords=historical_kw or "(none)",
            current_query=current_query,
        )
        try:
            raw = self._llm.call(prompt, temperature=0.15, max_tokens=300)
            result = self._parse_json(raw)
            enhanced = result.get("enhanced_query", current_query)
            if enhanced and len(enhanced) > 3:
                return {
                    "enhanced_query": enhanced,
                    "fused_context": result.get("fused_context", ""),
                }
        except Exception as e:
            logger.warning("fuse_context LLM failed: %s", e)
        return {"enhanced_query": current_query, "fused_context": "fusion failed, using original"}

    # -----------------------------------------------------------------
    # Query Pipeline Step 3: Academic Term Expansion
    # -----------------------------------------------------------------

    def expand_terms(self, query: str) -> Dict[str, Any]:
        """Expand query with academic synonyms and abbreviations.

        Returns dict with keys: expanded_query, added_terms.
        """
        if not self._llm:
            return {"expanded_query": query, "added_terms": []}

        prompt = TERM_EXPANSION_PROMPT.format(query=query)
        try:
            raw = self._llm.call(prompt, temperature=0.2, max_tokens=256)
            result = self._parse_json(raw)
            expanded = result.get("expanded_query", query)
            if expanded and len(expanded) > len(query):
                return {
                    "expanded_query": expanded,
                    "added_terms": result.get("added_terms", []),
                }
        except Exception as e:
            logger.warning("expand_terms LLM failed: %s", e)
        return {"expanded_query": query, "added_terms": []}

    # -----------------------------------------------------------------
    # Query Pipeline Step 4: Parent Query Generation (for vague queries)
    # -----------------------------------------------------------------

    def generate_parent_query(self, user_query: str,
                              context_summary: str = "") -> Dict[str, Any]:
        """Infer a precise parent query when the user's query is vague/short.

        Returns dict with keys: parent_query, inferred_intent.
        """
        if not self._llm:
            return {"parent_query": user_query, "inferred_intent": "no LLM available"}

        prompt = PARENT_QUERY_PROMPT.format(
            user_query=user_query,
            context_summary=context_summary or "(no additional context)",
        )
        try:
            raw = self._llm.call(prompt, temperature=0.2, max_tokens=256)
            result = self._parse_json(raw)
            parent = result.get("parent_query", user_query)
            if parent and len(parent) > 3:
                return {
                    "parent_query": parent,
                    "inferred_intent": result.get("inferred_intent", ""),
                }
        except Exception as e:
            logger.warning("generate_parent_query LLM failed: %s", e)
        return {"parent_query": user_query, "inferred_intent": "generation failed"}

    # -----------------------------------------------------------------
    # Query Pipeline Step 5: Hierarchical Sub-query Decomposition
    # -----------------------------------------------------------------

    def decompose_sub_queries(self, user_query: str) -> Dict[str, Any]:
        """Decompose complex queries into atomic sub-questions.

        Returns dict with keys: is_complex, sub_queries, decomposition_rationale.
        """
        if not self._is_potentially_complex(user_query):
            return {"is_complex": False, "sub_queries": [], "decomposition_rationale": "no complexity signals"}

        if not self._llm:
            return {"is_complex": False, "sub_queries": [], "decomposition_rationale": "no LLM available"}

        prompt = SUB_QUERY_DECOMPOSE_PROMPT.format(user_query=user_query)
        try:
            raw = self._llm.call(prompt, temperature=0.15, max_tokens=400)
            result = self._parse_json(raw)
            subs = result.get("sub_queries", [])
            is_complex = result.get("is_complex", False)
            if is_complex and subs:
                return {
                    "is_complex": True,
                    "sub_queries": subs[:4],
                    "decomposition_rationale": result.get("decomposition_rationale", ""),
                }
        except Exception as e:
            logger.warning("decompose_sub_queries LLM failed: %s", e)
        return {"is_complex": False, "sub_queries": [], "decomposition_rationale": "decomposition failed"}

    @staticmethod
    def _is_potentially_complex(query: str) -> bool:
        """Quick heuristic check for complexity signals before calling LLM."""
        q = query.lower()
        signals = [
            "compare", "vs", "versus", "difference", "differ",
            "对比", "区别", "差异", "比较",
            "which ones", "what are the", "哪些", "列举", "有哪些",
            " and ", "以及", "同时", " 和 ",
            "how to", "steps", "怎么", "步骤", "流程", "如何",
        ]
        return any(s in q for s in signals)

    @staticmethod
    def _is_vague_query(query: str) -> bool:
        """Heuristic: is the query too vague/short to retrieve directly?"""
        words = query.strip().split()
        if len(words) <= 3:
            return True
        q = query.lower()
        vague_patterns = ["tell me about", "what about", "介绍一下", "说说", "讲讲", "了解"]
        return any(p in q for p in vague_patterns)

    # -----------------------------------------------------------------
    # Context extraction helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _summarize_history(history: List[Dict], max_turns: int = 6) -> str:
        """Extract a compact summary of recent conversation turns."""
        if not history:
            return ""
        recent = history[-max_turns * 2:] if len(history) > max_turns * 2 else history
        parts = []
        for msg in recent:
            role = msg.get("role", "?")
            content = msg.get("content", "")[:150]
            if content.strip():
                parts.append(f"{role}: {content}")
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _summarize_cited_papers(cited_papers: Dict[str, Any], max_papers: int = 5) -> str:
        """Build a short summary of cited papers (title only)."""
        if not cited_papers:
            return ""
        titles = []
        for pid, p in list(cited_papers.items())[:max_papers]:
            title = getattr(p, "title", str(p)) if not isinstance(p, dict) else p.get("title", str(p))
            titles.append(title)
        return "; ".join(titles)

    @staticmethod
    def _extract_historical_keywords(history: List[Dict],
                                     conversation_state: Optional[Dict] = None) -> str:
        """Extract salient keywords from past user queries and conversation state."""
        keywords: List[str] = []
        if conversation_state:
            kw_list = conversation_state.get("keywords") or conversation_state.get("topics") or []
            if isinstance(kw_list, list):
                keywords.extend(str(k) for k in kw_list[:10])
        for msg in (history or []):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if len(content) > 5:
                    keywords.append(content[:80])
        seen: set = set()
        unique: List[str] = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return ", ".join(unique[:8]) if unique else ""

    # -----------------------------------------------------------------
    # Main entry point (called by LangGraph node)
    # -----------------------------------------------------------------

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Unified decision method with full query processing pipeline.

        Pipeline:
          User input → filter & correct → context fusion → term expansion
          → vague? parent query / complex? sub-query decomposition
          → route decision → retrieval evaluation → write state
        """
        self._ensure_llm(state)

        user_query = state.get("user_query", "")
        feedback = state.get("user_feedback", "")
        conversation_state = state.get("conversation_state") or {}
        history = state.get("history", [])
        cited_papers = state.get("cited_papers", {})
        prior_retrieval = state.get("retrieval_result") or state.get("fused_candidates") or []

        # Feedback takes priority — short-circuit
        if feedback:
            decision = {
                "route": ROUTE_HANDLE_FEEDBACK,
                "optimized_query": "",
                "reasoning": "User provided feedback on previous recommendation.",
                "retrieval_evaluation": None,
            }
            state["plan"] = {"route": ROUTE_HANDLE_FEEDBACK, "reasoning": decision["reasoning"]}
            state["planner_decision"] = decision
            logger.info("Planner: feedback route")
            return state

        # Profile extraction: on second turn after profile was asked,
        # try to extract user info if they responded with profile data
        if state.get("profile_asked") and not state.get("profile_completed"):
            self._try_extract_profile(state, user_query, history)

        if not user_query.strip():
            decision = {
                "route": ROUTE_NO_RETRIEVAL,
                "optimized_query": "",
                "reasoning": "No query provided.",
                "retrieval_evaluation": None,
            }
            state["plan"] = {"route": ROUTE_NO_RETRIEVAL, "reasoning": decision["reasoning"]}
            state["planner_decision"] = decision
            return state

        # ============================================================
        # Query Processing Pipeline
        # ============================================================
        pipeline_log: Dict[str, Any] = {}

        # --- Pipeline Step 1: Filter & Correct ---
        filter_result = self.filter_and_correct_query(user_query)
        pipeline_log["filter"] = filter_result
        if not filter_result["is_valid"]:
            logger.info("Planner: query filtered as invalid — %s", filter_result["corrections"])
            decision = {
                "route": ROUTE_NEED_CLARIFY,
                "optimized_query": "",
                "reasoning": f"Query invalid: {', '.join(filter_result['corrections'])}",
                "retrieval_evaluation": None,
            }
            state["plan"] = {"route": ROUTE_NEED_CLARIFY, "reasoning": decision["reasoning"]}
            state["planner_decision"] = decision
            state["final_query"] = ""
            state["parent_query"] = ""
            state["sub_queries"] = []
            return state

        corrected_query = filter_result["corrected_query"]
        if filter_result["corrections"]:
            logger.info("Planner: query corrected — %s", filter_result["corrections"])

        # --- Pipeline Step 2: Multi-turn Context Fusion ---
        fusion_result = self.fuse_context(
            corrected_query, history, cited_papers, conversation_state,
        )
        pipeline_log["context_fusion"] = fusion_result
        fused_query = fusion_result["enhanced_query"]

        # --- Pipeline Step 3: Academic Term Expansion ---
        expansion_result = self.expand_terms(fused_query)
        pipeline_log["term_expansion"] = expansion_result
        expanded_query = expansion_result["expanded_query"]

        # --- Pipeline Step 4 / 5: Vague → parent query, Complex → sub-queries ---
        parent_query = ""
        sub_queries: List[str] = []

        is_vague = self._is_vague_query(corrected_query)
        is_complex = self._is_potentially_complex(corrected_query)

        if is_vague and not is_complex:
            context_for_parent = self._summarize_history(history, max_turns=4)
            parent_result = self.generate_parent_query(corrected_query, context_for_parent)
            pipeline_log["parent_query"] = parent_result
            parent_query = parent_result["parent_query"]
            logger.info("Planner: vague query → parent_query='%s'", parent_query[:80])
        elif is_complex:
            decompose_result = self.decompose_sub_queries(corrected_query)
            pipeline_log["sub_queries"] = decompose_result
            if decompose_result["is_complex"]:
                sub_queries = decompose_result["sub_queries"]
                logger.info("Planner: complex query → %d sub-queries", len(sub_queries))

        # --- Build final_query (the best single query for retrieval) ---
        if parent_query:
            final_query = parent_query
        else:
            final_query = expanded_query

        # Detect recency preference
        wants_latest = self._detect_recency_preference(user_query)
        if wants_latest:
            state["prefer_latest_papers"] = True
            state["online_offline_fusion_ratio"] = 0.7

        # Legacy optimize_query pass (provides the optimized_query field
        # expected by downstream nodes; reuses the already-enriched final_query)
        optimized_query = final_query

        # ============================================================
        # Route Decision (same as before)
        # ============================================================
        has_paper_context = bool(cited_papers) or any(
            "[1]" in m.get("content", "") or "paper" in m.get("content", "").lower()
            for m in history if m.get("role") == "assistant"
        )
        route_result = self.decide_route(
            user_query, optimized_query, conversation_state, has_paper_context,
        )
        route = route_result["route"]

        # Evaluate prior retrieval results (if they exist)
        retrieval_eval = None
        if prior_retrieval and route == ROUTE_RETRIEVE_LOCAL:
            retrieval_eval = self.evaluate_retrieval(
                user_query, optimized_query, prior_retrieval,
            )
            if retrieval_eval["quality"] == "SUFFICIENT":
                route = ROUTE_NO_RETRIEVAL
                route_result["reasoning"] += " Prior retrieval is sufficient; skipping re-retrieval."
            elif retrieval_eval["quality"] == "INSUFFICIENT" and retrieval_eval.get("suggested_refined_query"):
                optimized_query = retrieval_eval["suggested_refined_query"]
                final_query = optimized_query

        # ============================================================
        # Write to state
        # ============================================================
        do_online = wants_latest or state.get("is_daily_rec", False)
        decision = {
            "route": route,
            "response_style": route_result.get("response_style", "recommend"),
            "optimized_query": optimized_query,
            "reasoning": route_result.get("reasoning", ""),
            "retrieval_evaluation": retrieval_eval,
            "do_online_search": do_online,
            "pipeline_log": pipeline_log,
        }

        state["plan"] = {
            "route": route,
            "do_online_search": do_online,
            "prefer_latest_papers": wants_latest or state.get("prefer_latest_papers", False),
            "reasoning": decision["reasoning"],
        }
        state["planner_decision"] = decision
        state["optimized_query"] = optimized_query

        state["final_query"] = final_query
        state["parent_query"] = parent_query
        state["sub_queries"] = sub_queries

        logger.info(
            "Planner: route=%s final_query='%s' parent_query='%s' sub_queries=%d reasoning='%s'",
            route, final_query[:60], parent_query[:40] if parent_query else "(none)",
            len(sub_queries), decision["reasoning"][:80],
        )
        return state

    # -----------------------------------------------------------------
    # Recency preference detection
    # -----------------------------------------------------------------

    @staticmethod
    def _detect_recency_preference(user_query: str) -> bool:
        """Detect if user wants latest/newest/recent papers → trigger online search."""
        q = user_query.lower()
        recency_kw_en = ["latest", "newest", "most recent", "new papers", "recent papers",
                         "state of the art", "sota", "cutting edge", "up to date"]
        recency_kw_zh = ["最新", "最近", "新的", "前沿", "最前沿", "近期"]
        return any(kw in q for kw in recency_kw_en + recency_kw_zh)

    # -----------------------------------------------------------------
    # Pre-LLM trivial input detection (saves LLM calls)
    # -----------------------------------------------------------------

    @staticmethod
    def _check_trivial(user_query: str, history: list, cited_papers: dict) -> Optional[Dict[str, Any]]:
        """Return a decision dict if the query is trivially resolvable without LLM.
        Returns None if LLM-based planning is needed."""
        q = user_query.strip().lower().rstrip("!.?")

        greetings = {
            "hi", "hello", "hey", "hiya", "howdy",
            "thanks", "thank you", "thx", "ty",
            "bye", "goodbye", "see you",
            "ok", "okay", "sure", "got it", "yes", "no", "yep", "nope",
            "你好", "谢谢", "好的", "嗯", "再见",
        }
        if q in greetings or len(q) <= 2:
            return {
                "route": ROUTE_NO_RETRIEVAL,
                "optimized_query": user_query,
                "reasoning": "Greeting or trivial input; no retrieval needed.",
                "retrieval_evaluation": None,
            }

        # Vague recommendation/search requests with no specific topic → ask user
        vague_words = {"recommend", "suggest", "find", "search", "papers",
                       "help", "show", "give", "list", "get",
                       "推荐", "找", "搜", "帮", "论文", "文章", "看看",
                       "some", "me", "i", "want", "need", "please",
                       "一些", "我", "想", "要", "能", "吗", "呢", "啊",
                       "可以", "帮我", "给我"}
        words = set(q.split())
        has_topic_keyword = words - vague_words - {"a", "the", "of", "on", "for", "about", "的", "了", "下"}
        if not has_topic_keyword and len(q.split()) <= 8:
            vague_triggers = {"recommend", "suggest", "find", "search", "papers",
                              "help", "推荐", "找", "搜", "论文", "文章"}
            if words & vague_triggers:
                return {
                    "route": ROUTE_NEED_CLARIFY,
                    "optimized_query": user_query,
                    "reasoning": "Query asks for papers but lacks specific topic; need clarification.",
                    "retrieval_evaluation": None,
                }

        # Vague pattern matching for common phrases without topic
        import re
        vague_patterns = [
            r"^(推荐|找|搜|帮我找|给我推荐|帮我推荐|推荐一下|推荐一些|帮我搜)(论文|文章|paper|papers)?[吗呢啊吧]?$",
            r"^(recommend|suggest|find|search|show)(\s+me)?(\s+some)?(\s+papers?)?[.!?]?$",
            r"^(i\s+)?(want|need|looking\s+for)(\s+some)?(\s+papers?)?[.!?]?$",
            r"^(can\s+you\s+)?(help|assist)(\s+me)?(\s+find)?(\s+papers?)?[.!?]?$",
            r"^(有什么|有没有)(好的|推荐的)?(论文|文章)?[吗呢]?$",
            r"^(我想|我要)(看|读|找)(论文|文章|paper)[吗呢]?$",
        ]
        if any(re.match(pat, q) for pat in vague_patterns):
            return {
                "route": ROUTE_NEED_CLARIFY,
                "optimized_query": user_query,
                "reasoning": "Vague recommendation request without specific topic.",
                "retrieval_evaluation": None,
            }

        # Follow-up about already-discussed papers (no retrieval needed)
        if cited_papers and len(history) >= 2:
            followup_signals = [
                "more about", "tell me more", "elaborate", "expand on",
                "what about", "and what", "also", "how about",
                "第一篇", "第二篇", "详细", "展开",
            ]
            if any(sig in q for sig in followup_signals):
                return {
                    "route": ROUTE_NO_RETRIEVAL,
                    "optimized_query": user_query,
                    "reasoning": "Follow-up on already-discussed papers; context sufficient.",
                    "retrieval_evaluation": None,
                }

        return None

    # -----------------------------------------------------------------
    # Fallback (rule-based, used when no LLM available)
    # -----------------------------------------------------------------

    @staticmethod
    def _fallback_route(user_query: str, has_paper_context: bool) -> Dict[str, str]:
        """Keyword-based fallback when LLM is unavailable."""
        q = user_query.lower().strip()

        greetings = {"hi", "hello", "hey", "thanks", "thank you", "bye", "ok", "okay",
                     "你好", "谢谢", "好的", "嗯", "再见"}
        if q in greetings or len(q) < 3:
            return {"route": ROUTE_NO_RETRIEVAL, "response_style": "recommend",
                    "reasoning": "Greeting or trivial input."}

        import re
        vague_patterns = [
            r"^(recommend|suggest|find|search|show)(\s+me)?(\s+some)?(\s+papers?)?[.!?]?$",
            r"^(推荐|找|搜|帮我找|给我推荐|帮我推荐)(论文|文章)?[吗呢啊吧]?$",
            r"^(i\s+)?(want|need)(\s+some)?(\s+papers?)?[.!?]?$",
            r"^(有什么|有没有)(论文|文章)?[吗呢]?$",
        ]
        if any(re.match(pat, q) for pat in vague_patterns):
            return {"route": ROUTE_NEED_CLARIFY, "response_style": "recommend",
                    "reasoning": "Vague request with no topic."}

        clarify_signals = [
            q == "recommend",
            q == "suggest",
            len(q.split()) < 2 and "?" not in q,
        ]
        if any(clarify_signals):
            return {"route": ROUTE_NEED_CLARIFY, "response_style": "recommend",
                    "reasoning": "Query too vague for retrieval."}

        narrative_kw = [
            "compare", "summarize", "explain", "what is", "how does", "difference",
            "history", "evolution", "overview", "development",
            "对比", "总结", "讲讲", "介绍", "发展", "演进", "综述", "概述", "区别",
        ]
        style = "narrative" if any(kw in q for kw in narrative_kw) else "recommend"

        retrieval_kw = [
            "paper", "method", "approach", "model", "technique", "compare",
            "recommend", "suggest", "find", "search", "latest", "recent",
            "summarize", "explain", "what is", "how does", "difference",
        ]
        if any(kw in q for kw in retrieval_kw) or "?" in q:
            return {"route": ROUTE_RETRIEVE_LOCAL, "response_style": style,
                    "reasoning": "Query likely needs paper evidence."}

        if has_paper_context:
            return {"route": ROUTE_NO_RETRIEVAL, "response_style": style,
                    "reasoning": "General query with existing context."}

        return {"route": ROUTE_RETRIEVE_LOCAL, "response_style": style,
                "reasoning": "Default: attempt retrieval."}

    # -----------------------------------------------------------------
    # Profile extraction
    # -----------------------------------------------------------------

    PROFILE_EXTRACT_PROMPT = """\
The assistant previously asked the user about their purpose/background for \
looking up papers. Analyze the user's latest message and INFER a user profile.

User message: "{user_message}"

Recent conversation:
{history_snippet}

INFERENCE GUIDE — the user may not state these explicitly; infer from context:
- "写论文/research paper/毕业论文" → role: researcher/grad_student, needs: depth + novelty
- "课程作业/class project/homework" → role: student, needs: introductory + survey papers
- "工作项目/work project/production" → role: engineer, needs: practical + SOTA methods
- "刚入门/new to this/exploring" → role: beginner, needs: tutorials + surveys
- "做综述/literature review" → role: researcher, needs: broad coverage + recent papers
- Mentioned specific topics (NLP, CV, RL, etc.) → preferred_categories

Return valid JSON only:
{{
  "has_profile_info": true/false,
  "interest_text": "inferred research interests as a short phrase, or empty string",
  "role": "grad_student | undergrad | researcher | professor | engineer | beginner | other | unknown",
  "purpose": "writing_paper | coursework | work_project | literature_review | exploration | other | unknown",
  "preferred_categories": ["inferred research areas, e.g. NLP, CV, RL, LLM"],
  "special_requirements": ["inferred preferences, e.g. 'needs survey papers', 'wants recent work', 'practical focus'"]
}}

If the user did NOT provide ANY useful info (they completely ignored the question \
and asked something unrelated), set has_profile_info to false and leave fields empty. \
But if they gave even a HINT (e.g. "我在写论文"), extract what you can."""

    def _try_extract_profile(self, state: Dict[str, Any],
                             user_query: str, history: List[Dict]) -> None:
        """Attempt to extract profile info from user's response. Updates state in place."""
        if not self._llm:
            state["profile_completed"] = True
            return

        history_snippet = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')[:150]}"
            for m in history[-4:]
        ) if history else "(no prior history)"

        try:
            raw = self._llm.call(
                self.PROFILE_EXTRACT_PROMPT.format(
                    user_message=user_query[:500],
                    history_snippet=history_snippet,
                ),
                temperature=0.1,
            )
            data = self._parse_json(raw)
        except Exception as e:
            logger.warning("Profile extraction failed: %s", e)
            state["profile_completed"] = True
            return

        if not data.get("has_profile_info"):
            logger.info("Planner: user did not provide profile info, moving on")
            state["profile_completed"] = True
            return

        profile = state.get("user_profile")
        if profile is None:
            from agent.models import UserProfile
            profile = UserProfile(user_id=state.get("user_id", "anonymous"))

        if data.get("interest_text"):
            profile.interest_text = data["interest_text"]
        if data.get("preferred_categories"):
            profile.preferred_categories = data["preferred_categories"]

        reqs = list(profile.special_requirements)
        if data.get("purpose") and data["purpose"] != "unknown":
            reqs.append(f"purpose:{data['purpose']}")
        if data.get("role") and data["role"] != "unknown":
            reqs.append(f"role:{data['role']}")
        if data.get("special_requirements"):
            reqs.extend(data["special_requirements"])
        profile.special_requirements = reqs

        state["user_profile"] = profile
        state["profile_completed"] = True
        logger.info("Planner: extracted profile — interests='%s', categories=%s, reqs=%s",
                     profile.interest_text, profile.preferred_categories,
                     profile.special_requirements)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, stripping markdown fences if present."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
