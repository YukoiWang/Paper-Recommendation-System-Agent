"""Paper QA agent: retrieval + LLM. Query-aware retrieval, intent routing, multi-turn, citations."""
from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models import Paper, UserProfile

logger = logging.getLogger(__name__)


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


class Intent:
    RECOMMEND = "recommend"
    QA = "qa"
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    EXPLAIN = "explain"
    GENERAL = "general"


def classify_intent(query: str) -> str:
    q = query.lower().strip()
    compare_kw = [
        "compare ", "difference between", " vs ", " versus ",
        "how does it differ", "how do they differ", "pros and cons", "which is better", "compared to",
    ]
    summarize_kw = [
        "summarize", "summary", "overview", "key points", "main findings", "tl;dr", "tldr", "recap",
    ]
    recommend_kw = [
        "recommend", "suggest", "find me", "latest papers", "new papers", "recent papers",
        "show me papers", "papers about", "find papers",
    ]
    explain_kw = [
        "explain", "what is ", "what are ", "how does ", "how do ", "why does ", "why do ",
        "tell me about", "describe", "elaborate", "clarify",
    ]
    for kw in compare_kw:
        if kw in q:
            return Intent.COMPARE
    for kw in summarize_kw:
        if kw in q:
            return Intent.SUMMARIZE
    for kw in recommend_kw:
        if kw in q:
            return Intent.RECOMMEND
    for kw in explain_kw:
        if kw in q:
            return Intent.EXPLAIN
    if "?" in q or any(w in q for w in ["paper", "method", "approach", "model", "technique"]):
        return Intent.QA
    return Intent.GENERAL


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
    Intent.RECOMMEND: RECOMMEND_PROMPT,
    Intent.QA: QA_PROMPT,
    Intent.COMPARE: COMPARE_PROMPT,
    Intent.SUMMARIZE: SUMMARIZE_PROMPT,
    Intent.EXPLAIN: EXPLAIN_PROMPT,
}


class PaperQAAgent:
    def __init__(self, retrieval_agent, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", top_k_context: int = 10, max_abstract_chars: int = 600,
                 temperature: float = 0.7, max_tokens: int = 2048):
        self.retrieval = retrieval_agent
        self.llm = LLMClient(api_key=api_key, base_url=base_url, model=model,
                             temperature=temperature, max_tokens=max_tokens)
        self.top_k_context = top_k_context
        self.max_abstract_chars = max_abstract_chars
        self._sessions: Dict[str, Conversation] = {}
        logger.info("PaperQAAgent: model=%s top_k=%s", model, top_k_context)

    def chat(self, query: str, user: Optional[UserProfile] = None,
             conversation_id: Optional[str] = None) -> Dict[str, Any]:
        conv = self._get_or_create_session(conversation_id)
        intent = classify_intent(query)
        if user is None:
            user = UserProfile(user_id="anonymous")
        papers = self._query_retrieve(query, user, intent)
        context = self._build_paper_context(papers)
        for i, p in enumerate(papers):
            conv.cited_papers[f"[{i+1}]"] = p
        conv.add_user_message(query)
        messages = self._build_llm_messages(query, context, intent, conv)
        response_text = self.llm.chat(messages)
        conv.add_assistant_message(response_text, metadata={"intent": intent, "num_papers": len(papers)})
        return {
            "response": response_text,
            "intent": intent,
            "papers": [
                {"rank": i + 1, "paper_id": p.paper_id, "title": p.title, "authors": p.authors[:3],
                 "categories": p.categories, "score": round(p.score, 4)}
                for i, p in enumerate(papers)
            ],
            "conversation_id": conv.conversation_id,
            "tokens_used": self.llm.total_tokens,
        }

    def get_session(self, conversation_id: str) -> Optional[Conversation]:
        return self._sessions.get(conversation_id)

    def clear_session(self, conversation_id: str) -> None:
        if conversation_id in self._sessions:
            self._sessions[conversation_id].clear()

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def _query_retrieve(self, query: str, user: UserProfile, intent: str) -> List[Paper]:
        if intent == Intent.GENERAL:
            return []
        try:
            query_vec = self.retrieval.embedder.encode(query)
        except Exception as e:
            logger.warning("Encode failed: %s", e)
            query_vec = None
        candidates: Dict[str, float] = {}
        if query_vec is not None and self.retrieval.store.size > 0:
            from recall_strategies import vector_recall
            for pid, score in vector_recall(query_vec, self.retrieval.store, top_k=self.top_k_context * 3):
                candidates[pid] = candidates.get(pid, 0.0) + score
        if user.interest_text or user.interest_vector is not None:
            profile_result = self.retrieval.retrieve_for_user(user)
            for p in profile_result.recommended_papers[:self.top_k_context]:
                candidates[p.paper_id] = candidates.get(p.paper_id, 0.0) + p.score * 0.3
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        papers = []
        for pid, score in sorted_candidates[:self.top_k_context]:
            paper = self.retrieval._cache.get(pid)
            if paper:
                papers.append(Paper(
                    paper_id=paper.paper_id, title=paper.title, abstract=paper.abstract,
                    authors=paper.authors, categories=paper.categories, published=paper.published,
                    score=score,
                ))
        logger.info("QueryRetrieve: %s papers intent=%s", len(papers), intent)
        return papers

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
            parts.append(f"[{i+1}] {p.title}\n     Authors: {authors}\n     Categories: {cats} | Date: {date}\n     Abstract: {abstract}\n")
        return "\n".join(parts)

    def _build_llm_messages(self, query: str, context: str, intent: str, conv: Conversation) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        intent_prompt = INTENT_PROMPTS.get(intent, "")
        if intent_prompt and context:
            messages.append({"role": "system", "content": intent_prompt.format(context=context)})
        for msg in conv.messages[:-1]:
            messages.append(msg.to_api_dict())
        messages.append({"role": "user", "content": query})
        return messages

    def _get_or_create_session(self, conversation_id: Optional[str]) -> Conversation:
        if conversation_id and conversation_id in self._sessions:
            return self._sessions[conversation_id]
        conv = Conversation(conversation_id=conversation_id or "")
        self._sessions[conv.conversation_id] = conv
        logger.info("New conversation: %s", conv.conversation_id)
        return conv
