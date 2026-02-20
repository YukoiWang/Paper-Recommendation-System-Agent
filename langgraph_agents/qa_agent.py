"""Paper QA agent: pure LLM text generation. No routing, no intent classification.

Three entry points called by workflow nodes:
  - respond(state) -> generate answer based on papers + history
  - ask_profile(state) -> ask user about research interests
  - handle_feedback(state) -> process user feedback, update profile
"""
from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from agent.models import Paper, UserProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for conversation management
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str
    content: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    conversation_id: str = ""
    messages: List[Message] = field(default_factory=list)
    cited_papers: Dict[str, Paper] = field(default_factory=dict)
    max_history_turns: int = 20
    max_context_chars: int = 60000

    def __post_init__(self):
        if not self.conversation_id:
            self.conversation_id = str(uuid.uuid4())[:8]

    def add_user_message(self, content: str) -> Message:
        msg = Message(role="user", content=content, timestamp=time.time())
        self.messages.append(msg)
        self._trim()
        return msg

    def add_assistant_message(self, content: str, metadata: Dict = None) -> Message:
        msg = Message(role="assistant", content=content, timestamp=time.time(), metadata=metadata or {})
        self.messages.append(msg)
        self._trim()
        return msg

    def get_history_for_api(self) -> List[Dict[str, str]]:
        return [m.to_api_dict() for m in self.messages]

    def _trim(self):
        if len(self.messages) > self.max_history_turns * 2:
            self.messages = self.messages[-(self.max_history_turns * 2):]
        total = sum(len(m.content) for m in self.messages)
        while total > self.max_context_chars and len(self.messages) > 2:
            removed = self.messages.pop(0)
            total -= len(removed.content)

    def clear(self):
        self.messages.clear()
        self.cited_papers.clear()


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", temperature: float = 0.7, max_tokens: int = 2048,
                 timeout: float = 60.0):
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

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        t0 = time.time()
        response = self.client.chat.completions.create(**params)
        text = (response.choices[0].message.content or "").strip()
        if response.usage:
            self._total_tokens += response.usage.total_tokens
        logger.info("LLM %.1fs, tokens=%s", time.time() - t0, self._total_tokens)
        return text

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert ML research assistant. You help researchers discover, understand, and analyze machine learning papers from ArXiv.
- Recommend papers, explain methods, compare approaches, summarize findings.
- Cite papers as [1], [2] matching the provided list. Be precise and technical. Respond in English."""

RECOMMEND_PROMPT = """Based on the user's interests and the retrieved papers below, give personalized recommendations. For each: why it's relevant, key contribution, connections to others.

Retrieved papers:
{context}"""

QA_PROMPT = """Answer using the retrieved papers. Cite as [1], [2].

Retrieved papers:
{context}"""

COMPARE_PROMPT = """Compare and contrast the approaches in the retrieved papers. Methodology, strengths, weaknesses, scenarios.

Retrieved papers:
{context}"""

SUMMARIZE_PROMPT = """Concise summary of key findings and contributions from the retrieved papers. Group related work.

Retrieved papers:
{context}"""

EXPLAIN_PROMPT = """Explain the concepts and methods in the retrieved papers. Use [1], [2] citations.

Retrieved papers:
{context}"""

INTENT_PROMPTS = {
    "recommend": RECOMMEND_PROMPT,
    "qa": QA_PROMPT,
    "compare": COMPARE_PROMPT,
    "summarize": SUMMARIZE_PROMPT,
    "explain": EXPLAIN_PROMPT,
}

PROACTIVE_ASK_PROMPT = """The user has not provided enough interests for paper recommendation.
Politely ask 1-2 specific questions to understand their research interests, e.g.:
- Topics/areas (e.g. LLMs, vision, reinforcement learning)
- Preferred categories (e.g. cs.LG, cs.CL)
- Specific authors or recent papers they liked
Keep it concise and friendly. Respond in English."""

FEEDBACK_CLARIFY_PROMPT = """The user gave feedback on a recommendation, but it's vague or ambiguous.
Ask them to clarify what they're looking for: more specific topics, different time range,
or any other preferences. Be friendly and concise. Respond in English."""


# ---------------------------------------------------------------------------
# PaperQAAgent: pure text generation
# ---------------------------------------------------------------------------

class PaperQAAgent:
    """
    Pure LLM text generation agent. No routing, no intent classification.
    Called by workflow nodes with papers and context already prepared.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", max_abstract_chars: int = 600,
                 temperature: float = 0.7, max_tokens: int = 2048):
        self.llm = LLMClient(api_key=api_key, base_url=base_url, model=model,
                             temperature=temperature, max_tokens=max_tokens)
        self.max_abstract_chars = max_abstract_chars
        logger.info("PaperQAAgent: model=%s", model)

    def respond(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response based on papers already in state (from rank/retrieval).
        Planner has already decided intent; papers are in final_papers or ranked_papers.
        """
        query = state.get("user_query", "")
        intent = state.get("qa_intent", "general")
        history = state.get("history", [])
        papers = state.get("final_papers") or state.get("ranked_papers") or []
        cited = dict(state.get("cited_papers", {}))

        context = self._build_paper_context(papers)
        for i, p in enumerate(papers):
            cited[f"[{i+1}]"] = p

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        prompt = INTENT_PROMPTS.get(intent, QA_PROMPT)
        if papers and prompt:
            messages.append({"role": "system", "content": prompt.format(context=context)})
        for m in history:
            messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        if not history or history[-1].get("content") != query:
            messages.append({"role": "user", "content": query})

        response_text = self.llm.chat(messages)

        new_history = list(history)
        if not new_history or new_history[-1].get("content") != query:
            new_history.append({"role": "user", "content": query})
        new_history.append({"role": "assistant", "content": response_text,
                            "metadata": {"intent": intent, "num_papers": len(papers)}})
        _trim_history(new_history)

        return {
            **state,
            "response": response_text,
            "history": new_history,
            "cited_papers": cited,
        }

    def ask_profile(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a question asking the user about their research interests."""
        history = state.get("history", [])
        query = state.get("user_query", "")

        messages = [
            {"role": "system", "content": PROACTIVE_ASK_PROMPT},
        ]
        for m in history:
            messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        if query and (not history or history[-1].get("content") != query):
            messages.append({"role": "user", "content": query})

        response_text = self.llm.chat(messages)

        new_history = list(history)
        if query and (not new_history or new_history[-1].get("content") != query):
            new_history.append({"role": "user", "content": query})
        new_history.append({"role": "assistant", "content": response_text,
                            "metadata": {"action": "ask_profile"}})
        _trim_history(new_history)

        return {
            **state,
            "response": response_text,
            "history": new_history,
            "needs_profile_clarification": True,
        }

    def handle_feedback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user feedback. If vague, ask for clarification.
        If substantive, update profile and offer to re-recommend.
        """
        feedback = (state.get("user_feedback") or "").strip()
        profile = state.get("user_profile") or UserProfile(user_id=state.get("user_id", "anonymous"))
        history = state.get("history", [])

        is_vague = len(feedback.split()) < 4 or feedback.lower() in (
            "no", "nope", "not really", "wrong", "bad", "不好",
        )

        new_history = list(history)
        if feedback:
            new_history.append({"role": "user", "content": feedback})

        if is_vague:
            messages = [{"role": "system", "content": FEEDBACK_CLARIFY_PROMPT}]
            for m in new_history:
                messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            response_text = self.llm.chat(messages)
            new_history.append({"role": "assistant", "content": response_text,
                                "metadata": {"action": "feedback_clarify"}})
            _trim_history(new_history)
            return {
                **state,
                "response": response_text,
                "history": new_history,
                "needs_profile_clarification": True,
                "user_feedback": "",
            }

        # Substantive feedback: update profile
        profile.special_requirements = profile.special_requirements or []
        for w in feedback.split():
            if len(w) > 2 and w not in profile.special_requirements:
                profile.special_requirements.append(w)
        if profile.interest_text and feedback not in profile.interest_text:
            profile.interest_text = profile.interest_text + " " + feedback
        elif not profile.interest_text:
            profile.interest_text = feedback

        response_text = "I've updated your preferences. Would you like me to recommend papers again with these interests?"
        new_history.append({"role": "assistant", "content": response_text,
                            "metadata": {"action": "feedback_applied"}})
        _trim_history(new_history)

        return {
            **state,
            "response": response_text,
            "history": new_history,
            "user_profile": profile,
            "user_feedback": "",
        }

    def _build_paper_context(self, papers: List[Paper]) -> str:
        if not papers:
            return "(No papers retrieved)"
        parts = []
        for i, p in enumerate(papers):
            abstract = p.abstract[:self.max_abstract_chars]
            if len(p.abstract) > self.max_abstract_chars:
                abstract += "..."
            authors = ", ".join(p.authors[:4])
            if len(p.authors) > 4:
                authors += f" et al. ({len(p.authors)} authors)"
            cats = " | ".join(p.categories)
            date = p.published or "N/A"
            parts.append(
                f"[{i+1}] {p.title}\n"
                f"     Authors: {authors}\n"
                f"     Categories: {cats} | Date: {date}\n"
                f"     Abstract: {abstract}\n"
            )
        return "\n".join(parts)


def _trim_history(history: List[Dict], max_turns: int = 20, max_chars: int = 60000):
    """Trim history list in-place."""
    if len(history) > max_turns * 2:
        del history[:len(history) - max_turns * 2]
    total = sum(len(m.get("content", "")) for m in history)
    while total > max_chars and len(history) > 2:
        removed = history.pop(0)
        total -= len(removed.get("content", ""))
