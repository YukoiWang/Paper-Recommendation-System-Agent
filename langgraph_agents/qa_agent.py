"""Paper QA agent — conversation, history, state summarization, response synthesis.

Four responsibilities:
  1. Chat naturally with user (proactive, anti-hallucination, forward-driving)
  2. Manage conversation history (append every turn, auto-trim)
  3. Summarize conversation state each turn → structured dict on shared state
  4. Synthesize final answer from retrieved papers

Three entry points called by workflow nodes:
  - respond(state)         → generate answer based on papers + history
  - ask_profile(state)     → proactively gather research interests
  - handle_feedback(state) → process user feedback, update profile
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from agent.models import Paper, UserProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", temperature: float = 0.7,
                 max_tokens: int = 2048, timeout: float = 60.0):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._total_tokens = 0
        logger.info("LLMClient: model=%s", model)

    def chat(self, messages: List[Dict[str, str]], **kwargs):
        """Chat completion helper.

        Extra kwargs (optional):
          - with_logprobs: bool = False  → request token-level logprobs
          - return_raw: bool = False     → return full API response instead of text
        """
        with_logprobs: bool = bool(kwargs.pop("with_logprobs", False))
        return_raw: bool = bool(kwargs.pop("return_raw", False))
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if with_logprobs:
            # Request token-level log probabilities when the backend supports it.
            # Some models may ignore this field; callers must handle missing logprobs.
            params["logprobs"] = True
        t0 = time.time()
        response = self.client.chat.completions.create(**params)
        text = (response.choices[0].message.content or "").strip()
        if response.usage:
            self._total_tokens += response.usage.total_tokens
        logger.info("LLM %.1fs, tokens=%s", time.time() - t0, self._total_tokens)
        if return_raw:
            return response
        return text

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


# ---------------------------------------------------------------------------
# Unified system prompt (replaces all intent-specific templates)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert ML/AI research assistant helping researchers discover, \
understand, and analyze machine learning papers.

## Conversation Style
- Be natural, warm, and conversational — like a knowledgeable colleague, \
not a search engine.
- Avoid robotic or formulaic responses. Vary your phrasing and structure.
- Use a professional yet approachable tone.

## Proactiveness
- If the user's request is vague or missing detail, proactively ask 1-2 \
specific questions to clarify (e.g. research topics, preferred ArXiv \
categories, time range, specific authors).
- After answering, suggest a natural next step: "Want me to dive deeper \
into any of these?" or "I can compare these methods if you'd like."
- Drive the conversation forward — don't just answer, anticipate what the \
user might need next.

## Anti-Hallucination Rules (CRITICAL)
- ONLY answer based on the retrieved papers provided in context. \
Do NOT fabricate papers, authors, results, or citations.
- If the retrieved papers are insufficient, say so honestly: \
"Based on the papers I have, I can't fully answer this — could you \
rephrase or give me more details?"
- If the conversation history lacks relevant context, admit it rather \
than guessing.
- Cite papers as [1], [2], etc., matching the numbered list below.
- Do NOT append a reference list or bibliography at the end of your response. \
The system will automatically add one. Just use [N] inline citations.

## Response Guidelines
- Recommendations: explain WHY each paper is relevant, highlight key \
contributions, note connections between papers.
- Questions: answer precisely with citations; distinguish paper claims \
from your interpretation.
- Comparisons: structure clearly — methodology, strengths, weaknesses, \
applicable scenarios.
- Summaries: group related work, highlight main findings and trends.

## System Capabilities
- This system CAN search the internet for papers via ArXiv and Semantic Scholar.
- If online search was performed for the current query, the results are included \
in the retrieved papers below. Do NOT tell the user "I cannot access the internet" \
— if papers were fetched online, mention that; if the online search returned no \
results, say "I searched but found no relevant papers online" rather than \
claiming you lack internet access.

## Language
- Always respond in the same language as the user's query."""


# ---------------------------------------------------------------------------
# Tone / style by user expertise (adjust response tone based on user profile)
# ---------------------------------------------------------------------------

TONE_BEGINNER = """\
## Tone for this user: Beginner-friendly
- The user is a beginner (e.g. student, newcomer to the field). Adjust your tone accordingly:
- Use plain, accessible language. Avoid jargon; when technical terms are necessary, briefly explain them.
- Prefer short sentences and concrete examples. Build up concepts step by step.
- Do NOT assume prior knowledge of advanced methods or notation. Keep explanations simple and accessible."""

TONE_INTERMEDIATE = """\
## Tone for this user: General / Intermediate
- The user has some background. Use a balanced tone: clear but not oversimplified.
- You may use standard terminology; briefly clarify only when it is domain-specific or ambiguous."""

TONE_EXPERT = """\
## Tone for this user: Expert / Academic
- The user has high academic or professional level. You may use a more academic, precise style:
- Use standard technical terminology and notation where appropriate.
- You can discuss methodological details, limitations, and related work in a concise, expert-to-expert manner.
- Use academic, concise phrasing; no need to simplify or avoid technical terms."""

TONE_DEFAULT = """\
## Tone for this user: Unknown level
- No strong signal about the user's level. Use a clear, neutral tone: avoid both oversimplification and unnecessary jargon. Prefer clarity."""


def _tokenize_pref(text: str) -> set:
    return {w for w in re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower()) if len(w) > 1}


def _compute_qa_preference_score(state: Dict[str, Any], papers: List[Paper]) -> float:
    """Match ranked papers to user profile + conversation_state + recent user turns (0–1)."""
    if not papers:
        return 1.0
    profile = state.get("user_profile")
    conv = state.get("conversation_state") or {}
    history = state.get("history", [])
    parts: List[str] = []
    if profile:
        if getattr(profile, "interest_text", ""):
            parts.append(str(profile.interest_text))
        parts.extend(list(getattr(profile, "preferred_categories", None) or []))
        parts.extend(list(getattr(profile, "special_requirements", None) or [])[:6])
    parts.extend(conv.get("keywords", []) or [])
    parts.extend(conv.get("research_topics", []) or [])
    for m in history[-6:]:
        if m.get("role") == "user":
            parts.append(str(m.get("content", "")[:240]))
    pref_blob = " ".join(str(p) for p in parts).lower()
    pref_tokens = _tokenize_pref(pref_blob)
    paper_chunks: List[str] = []
    for p in papers[:12]:
        paper_chunks.append((p.title or "").lower())
        paper_chunks.append(" ".join(p.categories or []).lower())
        paper_chunks.append((p.abstract or "")[:320].lower())
    paper_blob = " ".join(paper_chunks)
    paper_tokens = _tokenize_pref(paper_blob)
    if not pref_tokens:
        return 0.82
    inter = len(pref_tokens & paper_tokens)
    j = inter / max(1, len(pref_tokens))
    cat_user = {c.lower() for c in (getattr(profile, "preferred_categories", None) or [])} if profile else set()
    cat_papers: set = set()
    for p in papers[:12]:
        for c in p.categories or []:
            cat_papers.add(c.lower())
    if cat_user:
        cat_match = len(cat_user & cat_papers) / max(1, len(cat_user))
    else:
        cat_match = 0.55
    score = 0.5 * j + 0.5 * cat_match
    return max(0.0, min(1.0, float(score)))


def _build_qa_rerank_feedback(state: Dict[str, Any], papers: List[Paper], score: float) -> str:
    conv = state.get("conversation_state") or {}
    profile = state.get("user_profile")
    topics = ", ".join((conv.get("research_topics") or [])[:6])
    cats = ", ".join(list(getattr(profile, "preferred_categories", None) or [])[:6]) if profile else ""
    titles = "; ".join(((p.title or "")[:80]) for p in papers[:5])
    return (
        f"qa_preference_score={score:.2f}; user_topics={topics}; user_categories={cats}; "
        f"bias rerank toward these signals. sample_titles={titles}"
    )


def _safe_json_loads(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _get_tone_instruction(state: Dict[str, Any]) -> str:
    """Return system prompt block for tone/style based on user_profile.expertise_level or conversation_state.inferred_expertise."""
    profile = state.get("user_profile")
    conv = state.get("conversation_state") or {}
    level = None
    if profile and getattr(profile, "expertise_level", None):
        level = (profile.expertise_level or "").strip().lower()
    if not level and conv:
        level = (conv.get("inferred_expertise") or "").strip().lower()
    if level in ("beginner",):
        return TONE_BEGINNER
    if level in ("expert", "researcher"):
        return TONE_EXPERT
    if level in ("intermediate",):
        return TONE_INTERMEDIATE
    return TONE_DEFAULT


FIRST_TURN_PROFILE_PROMPT = """\
[First-turn instruction] This is the very beginning of the conversation. \
FIRST answer the user's question normally (or greet them warmly). \
THEN, at the end, casually ask about their PURPOSE for looking up papers — \
this helps you build a mental model of who they are without directly interrogating them.

The key insight: ask about their GOAL, not their identity. From their answer \
you can infer their level, field, and needs.

Example good approaches (pick ONE that fits the conversation flow):
- "对了，你找这些论文是为了写论文、做课程作业，还是工作项目需要？\
告诉我的话我可以更有针对性地帮你。"
- "By the way, are you looking into this for a research paper, a class project, \
or a work project? Knowing your goal helps me give better suggestions."
- "顺便问一下，你是在做哪个方向的研究？是刚入门想了解综述，\
还是已经在做具体的课题了？"
- "Just curious — is this for a literature review, exploring a new direction, \
or building something specific? I can adjust my recommendations accordingly."

Rules:
- ALWAYS answer/address the user's actual message first. The profile question \
is a casual follow-up at the very end.
- Keep the follow-up to 1-2 sentences. Make it sound like friendly small talk, \
not a form to fill out.
- If the user already mentioned their purpose or background, acknowledge it \
naturally ("听起来你在做XX方面的研究") instead of re-asking.
- Do NOT list multiple questions — pick the ONE most natural follow-up.
- This is a ONE-TIME ask. If they ignore it later, never bring it up again."""


PAPER_CONTEXT_BLOCK = """
## Retrieved Papers
{context}

Use ONLY these papers to support your answer. Cite as [1], [2], etc."""


RECOMMENDATION_INSTRUCTION = """\
## Recommendation Format (MUST follow when recommending papers)

You are recommending papers to the user. For EACH paper you mention, you MUST \
provide a **recommendation reason** that explains why this specific paper is \
relevant to the user's query.

### Per-paper structure:
**[N] Paper Title**
- **Recommendation reason**: 1-2 sentences explaining WHY this paper is relevant \
to the user's query — connect the paper's contribution to what the user is asking.
- **Key contribution**: the main methodological or empirical insight.
- **Relevance details**: specific aspects (method, dataset, task, findings) that \
relate to the query.

### Rules:
1. The recommendation reason MUST reference the user's query or intent — \
do NOT write generic reasons like "this is a good paper".
2. If the paper has a relevance score, incorporate it naturally \
(e.g. "highly relevant", "somewhat related") — do NOT show raw numbers.
3. After listing all papers, provide a brief synthesis: common themes, \
how the papers relate to each other, and a suggested next step.
4. Respond in the same language as the user's query.

User's query for reference: {user_query}"""


NARRATIVE_INSTRUCTION = """\
## Narrative Format (MUST follow when explaining, comparing, or summarizing topics)

You are answering a knowledge question about an academic topic. \
The user wants to UNDERSTAND something — not receive a paper list.

### Structure your response around CONTENT, not papers:
- Organize by logical structure: development stages, method categories, \
comparison dimensions, key concepts, or chronological evolution.
- Weave paper citations [N] naturally into the narrative as supporting evidence. \
For example: "The introduction of the Transformer architecture [3] marked a \
turning point, enabling models like BERT [5] to achieve..."
- When mentioning a paper, briefly note its key contribution inline — \
do NOT dedicate a separate section to each paper.

### Rules:
1. Do NOT use the per-paper recommendation format \
(i.e. do NOT write "**[N] Paper Title**" followed by bullet points for each paper).
2. The response should read like a knowledgeable explanation or essay, \
with papers cited as evidence throughout.
3. You may group papers when they share a common theme: \
"Several works [2][4][6] have explored this direction..."
4. End with a brief synthesis or forward-looking insight, and offer \
to dive deeper into any aspect.
5. Respond in the same language as the user's query.

User's query for reference: {user_query}"""


# Prompt for the state-summarization LLM call (low temperature, concise)
STATE_SUMMARY_PROMPT = """\
Based on the conversation history below, extract a structured JSON summary. \
Output valid JSON only — no markdown fences, no extra text.

Conversation:
{history_text}

Return exactly this structure:
{{
  "user_intent": "recommend | qa | compare | summarize | feedback | general",
  "research_topics": ["topic1", "topic2"],
  "keywords": ["keyword1", "keyword2"],
  "time_preference": "year or period, or null",
  "venue_preference": ["venue1"],
  "papers_discussed": ["title or short id of papers already discussed"],
  "user_satisfaction": "satisfied | neutral | unsatisfied | unknown",
  "open_questions": "what the user still wants to know (1 sentence or empty)",
  "summary": "1-2 sentence summary of the conversation state",
  "inferred_expertise": "beginner | intermediate | expert | unknown"
}}

Infer inferred_expertise from cues: e.g. "course/assignment/introductory/just starting" -> beginner; "paper/research/thesis/PhD" -> expert; otherwise unknown."""


# ---------------------------------------------------------------------------
# PaperQAAgent
# ---------------------------------------------------------------------------

class PaperQAAgent:
    """Conversation agent with four responsibilities:
    1. Natural, proactive dialogue with anti-hallucination guardrails
    2. Conversation history management
    3. Per-turn conversation state summarization (→ shared state / Blackboard)
    4. Final response synthesis from retrieved papers
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", max_abstract_chars: int = 600,
                 temperature: float = 0.7, max_tokens: int = 2048):
        self.llm = LLMClient(api_key=api_key, base_url=base_url, model=model,
                             temperature=temperature, max_tokens=max_tokens)
        self.max_abstract_chars = max_abstract_chars
        logger.info("PaperQAAgent init: model=%s", model)

    def _llm_natural_qa_rerank_bundle(
        self,
        state: Dict[str, Any],
        papers: List[Paper],
        score: float,
    ) -> tuple[str, str]:
        """Return (user_visible_message, rerank_instruction_for_rank_query)."""
        query = (state.get("user_query") or "").strip()
        conv = state.get("conversation_state") or {}
        profile = state.get("user_profile")
        prof_bits: List[str] = []
        if profile:
            if getattr(profile, "interest_text", ""):
                prof_bits.append(f"interest: {profile.interest_text}")
            if getattr(profile, "preferred_categories", None):
                prof_bits.append("categories: " + ", ".join(profile.preferred_categories[:8]))
        conv_bits = []
        if conv.get("research_topics"):
            conv_bits.append("topics: " + ", ".join(conv.get("research_topics")[:8]))
        if conv.get("keywords"):
            conv_bits.append("keywords: " + ", ".join(conv.get("keywords")[:10]))
        paper_lines = []
        for i, p in enumerate(papers[:5]):
            ab = (p.abstract or "")[:220].replace("\n", " ")
            cats = ", ".join((p.categories or [])[:6])
            paper_lines.append(
                f"[{i+1}] title={p.title!r} year={getattr(p, 'published', '')!r} cats={cats!r} abs={ab!r}"
            )
        prompt = (
            "You are a UX + recommendation alignment expert.\n"
            "The ranked papers do not match the user's preferences/context well.\n"
            "Return JSON only (no markdown fences).\n"
            "Fields:\n"
            '- "user_message": 1-2 sentences, same language as user_query, explain mismatch gently.\n'
            '- "rerank_instruction": 2-5 sentences, actionable instructions for a reranker query side. '
            "Mention which papers/aspects are off and what to emphasize instead (topics, venues, methods, recency).\n"
            f"qa_preference_score={score:.3f}\n"
            f"user_query: {query}\n"
            f"profile: {'; '.join(prof_bits) or '(none)'}\n"
            f"conversation_state: {'; '.join(conv_bits) or '(none)'}\n"
            "ranked_papers:\n" + "\n".join(paper_lines) + "\n"
            "JSON schema:\n"
            '{"user_message":"...","rerank_instruction":"..."}\n'
        )
        try:
            raw = self.llm.chat([{"role": "system", "content": prompt}], temperature=0.2, max_tokens=420)
            data = _safe_json_loads(raw) or {}
            um = (data.get("user_message") or "").strip()
            ins = (data.get("rerank_instruction") or "").strip()
            if um and ins:
                return um[:500], ins[:1200]
        except Exception as e:
            logger.warning("QA natural rerank bundle LLM failed: %s", e)
        return "", ""

    # ---------------------------------------------------------------
    # Entry point 1: respond — main conversation + answer synthesis
    # ---------------------------------------------------------------

    def respond(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on retrieved papers + conversation history.
        Uses the unified prompt — no intent-specific template switching.

        For NO_RETRIEVAL route (planner decided not to retrieve papers), this method
        will:
          - enable token-level logprobs (if supported by the backend) and compute a
            confidence score from them;
          - apply a conservative prompt that lets the model explicitly emit
            `[[NEED_RETRIEVAL]]` when it feels uncertain.
        These signals are written back to state for downstream routing."""
        query = state.get("user_query", "")
        history = state.get("history", [])
        papers = state.get("final_papers") or state.get("ranked_papers") or []
        cited = dict(state.get("cited_papers", {}))
        evaluation_mode = bool(state.get("evaluation_mode", False))

        decision = state.get("planner_decision") or {}
        route = (decision.get("route") or "").upper()
        papers_for_pref = list(papers)
        qa_pref = _compute_qa_preference_score(state, papers_for_pref) if papers_for_pref else 1.0
        state["qa_preference_score"] = qa_pref

        # One-shot QA → rerank when preferences poorly match the ranked list (retrieval path).
        if (
            papers_for_pref
            and route == "RETRIEVE_LOCAL"
            and not evaluation_mode
            and qa_pref < 0.7
            and int(state.get("qa_rerank_count", 0)) == 0
        ):
            fb = _build_qa_rerank_feedback(state, papers_for_pref, qa_pref)
            short = (
                "当前推荐列表与您的画像及对话上下文的匹配度偏低，"
                "我将根据您的偏好重新排序论文后再给出完整回答。"
            )
            um, ins = self._llm_natural_qa_rerank_bundle(state, papers_for_pref, qa_pref)
            if um:
                short = um
            if ins:
                fb = ins
            new_history = self._append_and_trim(
                history, query, short, {"qa_preference_score": qa_pref, "action": "qa_rerank"},
            )
            conv_state = self._summarize_conversation_state(new_history)
            out: Dict[str, Any] = {
                **state,
                "response": short,
                "history": new_history,
                "cited_papers": cited,
                "conversation_state": conv_state,
                "qa_preference_score": qa_pref,
                "qa_needs_rerank": True,
                "qa_feedback_for_rerank": fb,
                "_rank_from_qa": True,
            }
            is_first_turn = not history and not state.get("profile_asked", False)
            if is_first_turn and not evaluation_mode:
                out["profile_asked"] = True
            return out

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        tone_block = _get_tone_instruction(state)
        if tone_block:
            messages.append({"role": "system", "content": tone_block})

        is_first_turn = not history and not state.get("profile_asked", False)
        if is_first_turn and not evaluation_mode:
            messages.append({
                "role": "system",
                "content": FIRST_TURN_PROFILE_PROMPT,
            })

        retrieval_insufficient = state.get("retrieval_insufficient", False)
        if retrieval_insufficient:
            retry_count = state.get("retrieval_retry_count", 0)
            messages.append({
                "role": "system",
                "content": (
                    f"[System note: The retrieval system searched {retry_count + 1} time(s) "
                    f"but could not find papers that closely match the user's query. "
                    f"The papers below are the best available but may not be an ideal match. "
                    f"You MUST acknowledge this to the user honestly — for example: "
                    f"'I wasn't able to find papers that perfectly match your query, "
                    f"but here are the closest results I found. "
                    f"You could try rephrasing your query or providing more specific keywords.' "
                    f"Still cite and discuss whatever papers are available, but set expectations.]"
                ),
            })

        did_online = decision.get("do_online_search", False)
        online_count = len(state.get("online_search_result") or [])
        if did_online:
            online_note = (
                f"[System note: An online search (ArXiv) was performed for this query "
                f"and returned {online_count} paper(s). "
                + ("These are included in the papers below." if online_count > 0
                   else "No results were found online; only local papers are shown.")
                + "]"
            )
            messages.append({"role": "system", "content": online_note})

        response_style = decision.get("response_style", "recommend")

        # When planner chose NO_RETRIEVAL and there are no papers, we are in
        # "bare LLM" mode. Make the model extremely conservative and allow it
        # to verbally flag that retrieval is needed instead of guessing.
        no_retrieval_mode = (route == "NO_RETRIEVAL" and not papers)

        if papers:
            max_papers = state.get("max_context_papers")
            max_abstract_chars = state.get("max_context_abstract_chars")
            if max_papers is not None:
                papers = papers[:max_papers]
            context = self._build_paper_context(
                papers,
                max_abstract_chars=max_abstract_chars,
            )
            for i, p in enumerate(papers):
                cited[f"[{i+1}]"] = p
            messages.append({
                "role": "system",
                "content": PAPER_CONTEXT_BLOCK.format(context=context),
            })
            if response_style == "narrative":
                messages.append({
                    "role": "system",
                    "content": NARRATIVE_INSTRUCTION.format(user_query=query),
                })
            else:
                messages.append({
                    "role": "system",
                    "content": RECOMMENDATION_INSTRUCTION.format(user_query=query),
                })

        if no_retrieval_mode:
            messages.append({
                "role": "system",
                "content": (
                    "User asks: {query}\n"
                    "You are answering WITHOUT external retrieval.\n"
                    "Constraint: If the query involves specific academic papers, "
                    "obscure data, or recent events (post-training knowledge), you "
                    "MUST output a special token: [[NEED_RETRIEVAL]]. Do not attempt "
                    "to guess or fabricate details. If the question can be safely "
                    "answered from general, widely-known knowledge, you may answer "
                    "normally without using this token."
                ).format(query=query),
            })

        for m in history:
            messages.append({"role": m.get("role", "user"),
                             "content": m.get("content", "")})
        if not history or history[-1].get("content") != query:
            messages.append({"role": "user", "content": query})

        # Method 1: logprobs-based confidence when available and in NO_RETRIEVAL mode.
        use_logprobs = no_retrieval_mode
        raw_resp = None
        logprob_confidence: Optional[float] = None
        logprobs_available = False

        if use_logprobs:
            try:
                raw_resp = self.llm.chat(messages, with_logprobs=True, return_raw=True)
                response_text = (raw_resp.choices[0].message.content or "").strip()
                logprob_confidence, logprobs_available = self._compute_logprob_confidence(raw_resp)
            except Exception as e:
                logger.warning("LLM logprobs unavailable or failed, falling back to text-only: %s", e)
                raw_resp = None
                response_text = self.llm.chat(messages)
        else:
            response_text = self.llm.chat(messages)

        if papers:
            response_text = self._append_reference_list(response_text, cited)

        # Method 3 signal: explicit verbalized uncertainty token.
        verbal_flag = "[[NEED_RETRIEVAL]]" in response_text

        new_history = self._append_and_trim(
            history, query, response_text, {"num_papers": len(papers)},
        )
        conv_state = self._summarize_conversation_state(new_history)

        result = {
            **state,
            "response": response_text,
            "history": new_history,
            "cited_papers": cited,
            "conversation_state": conv_state,
            "qa_preference_score": qa_pref,
            "qa_needs_rerank": False,
        }

        # Expose NO_RETRIEVAL confidence signals back to workflow.
        if no_retrieval_mode:
            result["no_retrieval_confidence"] = logprob_confidence
            result["no_retrieval_logprobs_available"] = logprobs_available
            result["no_retrieval_verbal_flag"] = verbal_flag

        if is_first_turn and not evaluation_mode:
            result["profile_asked"] = True
        return result

    @staticmethod
    def _compute_logprob_confidence(raw_response: Any) -> tuple[Optional[float], bool]:
        """Compute confidence score from token logprobs, if present.

        Returns (confidence, available_flag). Confidence is in [0, 1] when
        available, otherwise (None, False)."""
        try:
            choice = raw_response.choices[0]
            logprobs_obj = getattr(choice, "logprobs", None)
            if not logprobs_obj:
                return None, False
            content = getattr(logprobs_obj, "content", None)
            if not content:
                return None, False

            token_logprobs = []
            for token in content:
                lp = getattr(token, "logprob", None)
                if lp is None and isinstance(token, dict):
                    lp = token.get("logprob")
                if lp is not None:
                    token_logprobs.append(float(lp))

            if not token_logprobs:
                return None, False

            avg_logprob = float(np.mean(token_logprobs))
            confidence = float(np.exp(avg_logprob))
            # Clamp to [0, 1] for safety.
            confidence = max(0.0, min(1.0, confidence))
            return confidence, True
        except Exception as e:
            logger.warning("Failed to compute logprob confidence: %s", e)
            return None, False

    # ---------------------------------------------------------------
    # Entry point 2: ask_profile — proactively gather user interests
    # ---------------------------------------------------------------

    def ask_profile(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Proactively ask user to clarify their request — research interests,
        preferred topics, time range, or other specifics."""
        query = state.get("user_query", "")
        history = state.get("history", [])
        is_first_turn = not history and not state.get("profile_asked", False)

        planner_reasoning = ""
        decision = state.get("planner_decision") or {}
        if decision.get("reasoning"):
            planner_reasoning = f"\nPlanner note: {decision['reasoning']}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        tone_block = _get_tone_instruction(state)
        if tone_block:
            messages.append({"role": "system", "content": tone_block})

        if is_first_turn:
            messages.append({
                "role": "system",
                "content": (
                    "This is the very FIRST message in the conversation. The user's request "
                    "is vague and lacks detail. Your goals:\n"
                    "1. Greet them warmly.\n"
                    "2. Ask about their PURPOSE — why they need papers. From their answer "
                    "you can naturally infer their field, level, and needs.\n"
                    "3. Keep it conversational and brief (2-3 sentences).\n\n"
                    "Example approaches:\n"
                    "- \"你好！我可以帮你找论文。方便说一下你找论文是为了什么吗？"
                    "比如写论文、做课程作业、还是工作上的项目？这样我能更有针对性地推荐。\"\n"
                    "- \"Hi! I'd love to help. Are you looking for papers for a research "
                    "project, a class assignment, or exploring a new area? "
                    "That helps me know what depth and style to aim for.\"\n\n"
                    "Do NOT make up papers. Just ask about their goal."
                    f"{planner_reasoning}"
                ),
            })
        else:
            messages.append({
                "role": "system",
                "content": (
                    "The user's request lacks enough detail to proceed with paper search. "
                    "Your job: ask 1-2 SHORT, SPECIFIC follow-up questions to understand "
                    "what they need. Be warm and conversational.\n\n"
                    "Good examples of follow-up questions:\n"
                    "- \"Sure! What research area are you interested in? "
                    "For example: NLP, computer vision, reinforcement learning, LLMs...?\"\n"
                    "- \"I'd love to help! Are you looking for survey papers, "
                    "recent methods, or something specific like a comparison?\"\n"
                    "- \"Got it — any preference on time range? "
                    "Like papers from the last year, or classic foundational work?\"\n\n"
                    "Do NOT make up papers or topics. Just ask what they want.\n"
                    "Keep your response under 3 sentences."
                    f"{planner_reasoning}"
                ),
            })

        for m in history:
            messages.append({"role": m.get("role", "user"),
                             "content": m.get("content", "")})
        if query and (not history or history[-1].get("content") != query):
            messages.append({"role": "user", "content": query})

        response_text = self.llm.chat(messages)

        new_history = self._append_and_trim(
            history, query, response_text, {"action": "ask_profile"},
        )
        conv_state = self._summarize_conversation_state(new_history)

        result = {
            **state,
            "response": response_text,
            "history": new_history,
            "needs_profile_clarification": True,
            "conversation_state": conv_state,
        }
        if is_first_turn:
            result["profile_asked"] = True
        return result

    def evaluate_no_retrieval_answer(self, query: str, answer: str) -> str:
        """Self-reflection evaluator when logprobs are unavailable.

        Returns:
          - 'LOW_CONFIDENCE'  → prefer to fall back to retrieval
          - 'HIGH_CONFIDENCE' → answer is likely safe as-is
        """
        q = (query or "").strip()
        a = (answer or "").strip()
        if not q or not a:
            return "LOW_CONFIDENCE"

        prompt = (
            "你是一个事实核查员。\n\n"
            f"用户问题：{q}\n\n"
            f"模型回答：{a}\n\n"
            "任务：请评估这个回答中是否包含具体的实体名称、论文标题、精确数字、"
            "或者非常冷门/需要外部资料支撑的知识。\n"
            "如果你认为这个回答有相当一部分内容可能是编造的、缺乏依据，"
            "或者很可能不准确，请只输出：LOW_CONFIDENCE。\n"
            "如果你认为这是常识性、入门级或高度确定的内容，且几乎不依赖外部检索，"
            "请只输出：HIGH_CONFIDENCE。\n"
            "注意：只能输出这两个单词之一，不要输出任何解释。"
        )

        try:
            resp = self.llm.chat(
                [{"role": "system", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            text = (resp or "").strip().upper()
            if "LOW_CONFIDENCE" in text:
                return "LOW_CONFIDENCE"
            if "HIGH_CONFIDENCE" in text:
                return "HIGH_CONFIDENCE"
        except Exception as e:
            logger.warning("Self-reflection evaluation failed: %s", e)
        # On any failure, be slightly optimistic to avoid over-triggering retrieval.
        return "HIGH_CONFIDENCE"

    # ---------------------------------------------------------------
    # Entry point 3: handle_feedback — process feedback on recs
    # ---------------------------------------------------------------

    def handle_feedback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback. Clarify if vague; update profile if substantive."""
        feedback = (state.get("user_feedback") or "").strip()
        profile = state.get("user_profile") or UserProfile(
            user_id=state.get("user_id", "anonymous"),
        )
        history = state.get("history", [])

        is_vague = len(feedback.split()) < 4 or feedback.lower() in (
            "no", "nope", "not really", "wrong", "bad", "不好",
        )

        new_history = list(history)
        if feedback:
            new_history.append({
                "role": "user", "content": feedback, "timestamp": time.time(),
            })

        if is_vague:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            tone_block = _get_tone_instruction(state)
            if tone_block:
                messages.append({"role": "system", "content": tone_block})
            messages.append({
                "role": "system",
                "content": (
                    "The user gave vague or brief feedback on the previous "
                    "recommendation. Follow the Proactiveness guidelines: "
                    "ask them to clarify what topics they'd prefer, what was "
                    "wrong, or which direction to explore. Be warm and concise."
                ),
            })
            for m in new_history:
                messages.append({"role": m.get("role", "user"),
                                 "content": m.get("content", "")})
            response_text = self.llm.chat(messages)
            new_history.append({
                "role": "assistant", "content": response_text,
                "timestamp": time.time(),
                "metadata": {"action": "feedback_clarify"},
            })
            _trim_history(new_history)
            conv_state = self._summarize_conversation_state(new_history)
            return {
                **state,
                "response": response_text,
                "history": new_history,
                "needs_profile_clarification": True,
                "user_feedback": "",
                "conversation_state": conv_state,
            }

        _apply_feedback_to_profile(profile, feedback)
        response_text = (
            "Got it — I've noted your preferences. "
            "Want me to recommend papers again with these updated interests?"
        )
        new_history.append({
            "role": "assistant", "content": response_text,
            "timestamp": time.time(),
            "metadata": {"action": "feedback_applied"},
        })
        _trim_history(new_history)
        conv_state = self._summarize_conversation_state(new_history)

        return {
            **state,
            "response": response_text,
            "history": new_history,
            "user_profile": profile,
            "user_feedback": "",
            "conversation_state": conv_state,
        }

    # ---------------------------------------------------------------
    # Conversation state summarization (Responsibility 3)
    # ---------------------------------------------------------------

    def _summarize_conversation_state(self, history: List[Dict]) -> Dict[str, Any]:
        """Extract structured conversation state from recent history via LLM.
        Result is written to shared state so other agents (planner, retrieval)
        can read it on the next turn."""
        if not history:
            return _empty_conversation_state()

        recent = history[-10:]
        history_text = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in recent
        )

        try:
            raw = self.llm.chat(
                [{"role": "system",
                  "content": STATE_SUMMARY_PROMPT.format(history_text=history_text)}],
                temperature=0.1,
                max_tokens=512,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            state = json.loads(raw)
            logger.info("Conversation state summary: %s", state.get("summary", ""))
            return state
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to parse conversation state: %s", e)
            return _empty_conversation_state()

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _strip_llm_reference_section(response: str) -> str:
        """Remove any reference/bibliography section the LLM may have generated,
        since we will append a programmatic one with accurate metadata."""
        patterns = [
            r'\n---\s*\n\*{0,2}(?:References|参考文献|引用文献|参考论文|Bibliography)\*{0,2}\s*\n[\s\S]*$',
            r'\n#{1,3}\s*(?:References|参考文献|引用文献|参考论文|Bibliography)\s*\n[\s\S]*$',
            r'\n\*{2}(?:References|参考文献|引用文献|参考论文|Bibliography)\*{2}\s*\n[\s\S]*$',
        ]
        for pat in patterns:
            response = re.sub(pat, '', response, flags=re.IGNORECASE)
        return response.rstrip()

    @staticmethod
    def _append_reference_list(response: str, cited: Dict[str, "Paper"]) -> str:
        """Strip any LLM-generated reference section, then append a
        programmatic one with accurate paper metadata."""
        refs_used = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", response)))
        if not refs_used:
            return response

        response = PaperQAAgent._strip_llm_reference_section(response)

        lines = ["\n\n---\n**References**"]
        for n in refs_used:
            key = f"[{n}]"
            p = cited.get(key)
            if p is None:
                continue
            parts = [f"[{n}] {p.title}"]
            authors = ", ".join(p.authors[:4]) if p.authors else ""
            if len(p.authors) > 4:
                authors += " et al."
            if authors:
                parts.append(authors)
            if p.published:
                parts.append(f"({p.published})")
            cats = " | ".join(p.categories) if p.categories else ""
            if cats:
                parts.append(cats)
            lines.append("  " + ". ".join(parts))
        if len(lines) == 1:
            return response
        return response + "\n".join(lines)

    def _build_paper_context(
        self,
        papers: List[Paper],
        max_abstract_chars: Optional[int] = None,
    ) -> str:
        if not papers:
            return "(No papers retrieved)"
        limit = max_abstract_chars if max_abstract_chars is not None else self.max_abstract_chars
        parts = []
        for i, p in enumerate(papers):
            abstract = p.abstract[:limit]
            if len(p.abstract) > limit:
                abstract += "..."
            authors = ", ".join(p.authors[:4])
            if len(p.authors) > 4:
                authors += f" et al. ({len(p.authors)} authors)"
            cats = " | ".join(p.categories)
            date = p.published or "N/A"
            score_line = f"     Relevance score: {p.score:.4f}\n" if p.score else ""
            parts.append(
                f"[{i+1}] {p.title}\n"
                f"     Authors: {authors}\n"
                f"     Categories: {cats} | Date: {date}\n"
                f"{score_line}"
                f"     Abstract: {abstract}\n"
            )
        return "\n".join(parts)

    def _append_and_trim(self, history: List[Dict], query: str,
                         response: str, metadata: Dict = None) -> List[Dict]:
        """Append user query + assistant response to history, then trim."""
        new_history = list(history)
        if query and (not new_history or new_history[-1].get("content") != query):
            new_history.append({
                "role": "user", "content": query, "timestamp": time.time(),
            })
        new_history.append({
            "role": "assistant", "content": response,
            "timestamp": time.time(), "metadata": metadata or {},
        })
        _trim_history(new_history)
        return new_history


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _empty_conversation_state() -> Dict[str, Any]:
    return {
        "user_intent": "general",
        "research_topics": [],
        "keywords": [],
        "time_preference": None,
        "venue_preference": [],
        "papers_discussed": [],
        "user_satisfaction": "unknown",
        "open_questions": "",
        "summary": "",
        "inferred_expertise": "unknown",
    }


def _apply_feedback_to_profile(profile: UserProfile, feedback: str):
    """Update profile fields from substantive feedback text."""
    profile.special_requirements = profile.special_requirements or []
    for w in feedback.split():
        if len(w) > 2 and w not in profile.special_requirements:
            profile.special_requirements.append(w)
    if profile.interest_text and feedback not in profile.interest_text:
        profile.interest_text = profile.interest_text + " " + feedback
    elif not profile.interest_text:
        profile.interest_text = feedback


def _trim_history(history: List[Dict], max_turns: int = 20,
                  max_chars: int = 60000):
    """Trim history list in-place to stay within limits."""
    if len(history) > max_turns * 2:
        del history[:len(history) - max_turns * 2]
    total = sum(len(m.get("content", "")) for m in history)
    while total > max_chars and len(history) > 2:
        removed = history.pop(0)
        total -= len(removed.get("content", ""))
