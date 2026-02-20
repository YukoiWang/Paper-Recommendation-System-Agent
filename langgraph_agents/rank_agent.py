"""Rank agent: LLM or BGE reranker. From agent/rank_agent."""
from __future__ import annotations
import logging
import re
from typing import Dict, List, Literal, Optional

import sys
_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from agent.models import Paper, UserProfile
from agent.rerank_prompt import build_rerank_messages, parse_rerank_response

logger = logging.getLogger(__name__)
DEFAULT_BGE_RERANKER_PATH = "./output/bge-finetuned"
FALLBACK_BGE_RERANKER = "BAAI/bge-reranker-base"
RankMode = Literal["llm", "bge_reranker"]


def _default_llm_client(api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com",
                       model: str = "deepseek-chat", **kwargs):
    try:
        from langgraph_agents.qa_agent import LLMClient
    except ImportError:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        client = OpenAI(api_key=api_key or "", base_url=base_url, timeout=kwargs.get("timeout", 60.0))
        return _OpenAIWrapper(client, model=model, **kwargs)
    return LLMClient(api_key=api_key or "", base_url=base_url, model=model, **kwargs)


class _OpenAIWrapper:
    def __init__(self, client, model: str = "deepseek-chat", temperature: float = 0.3, max_tokens: int = 1024):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[dict], **kwargs) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return (r.choices[0].message.content or "").strip()


def _resolve_model_path(model_path: str) -> str:
    """
    If model_path is a local directory that exists, use it (finetuned).
    Otherwise fall back to the base HuggingFace model (auto-downloaded).
    """
    import os
    if os.path.isdir(model_path):
        config_file = os.path.join(model_path, "config.json")
        if os.path.isfile(config_file):
            logger.info("Using finetuned reranker: %s", model_path)
            return model_path
        logger.warning("Directory %s exists but has no config.json; falling back to %s", model_path, FALLBACK_BGE_RERANKER)
    else:
        logger.info("Finetuned model not found at %s; falling back to %s", model_path, FALLBACK_BGE_RERANKER)
    return FALLBACK_BGE_RERANKER


def _score_with_bge_reranker(model_path: str, query: str, doc_texts: List[str], batch_size: int = 32) -> Optional[List[float]]:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        logger.warning("transformers/torch not installed; BGE reranker unavailable: %s", e)
        return None

    resolved_path = _resolve_model_path(model_path)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading BGE reranker from %s on %s", resolved_path, device)
        tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        model = AutoModelForSequenceClassification.from_pretrained(resolved_path).to(device)
        model.eval()
        pairs = [[query, d] for d in doc_texts]
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            with torch.no_grad():
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                outputs = model(**inputs)
                scores = outputs.logits.view(-1).float().cpu().numpy()
            all_scores.extend([float(s) for s in scores])
        return all_scores
    except Exception as e:
        logger.warning("BGE reranker load/predict failed (%s): %s", resolved_path, e)
        return None


class RankAgent:
    """Rerank papers by query + user profile. Modes: llm, bge_reranker."""

    def __init__(
        self,
        mode: RankMode = "llm",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        max_abstract_chars: int = 300,
        bge_reranker_model_path: Optional[str] = None,
    ):
        self._mode = mode
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_abstract_chars = max_abstract_chars
        self._bge_reranker_model_path = bge_reranker_model_path or DEFAULT_BGE_RERANKER_PATH
        self._llm: Optional[object] = None
        self._last_llm_reasons: Dict[str, object] = {}
        logger.info("RankAgent: mode=%s", mode)

    @property
    def last_llm_reasons(self) -> Dict[str, object]:
        return self._last_llm_reasons

    def _get_llm(self):
        if self._llm is None:
            self._llm = _default_llm_client(
                api_key=self._api_key, base_url=self._base_url, model=self._model,
                temperature=self._temperature, max_tokens=self._max_tokens,
            )
        return self._llm

    def rerank(
        self,
        papers: List[Paper],
        query: str,
        user: UserProfile,
        top_k: Optional[int] = None,
    ) -> List[Paper]:
        if not papers:
            return []
        if len(papers) == 1:
            p = papers[0]
            out = [Paper(paper_id=p.paper_id, title=p.title, abstract=p.abstract, authors=p.authors,
                        categories=p.categories, published=p.published, embedding=p.embedding, score=1.0)]
            return out[:top_k] if top_k else out
        if self._mode == "bge_reranker":
            return self._rerank_bge_reranker(papers, query, user, top_k)
        return self._rerank_llm(papers, query, user, top_k)

    def _rerank_bge_reranker(self, papers: List[Paper], query: str, user: UserProfile, top_k: Optional[int]) -> List[Paper]:
        try:
            from agent.rerank_prompt import _build_user_profile_text as _profile
            profile_text = _profile(user)
        except Exception:
            profile_text = ""
        query_side = query.strip()
        if profile_text:
            query_side = query_side + "\n" + profile_text
        doc_texts = [f"{p.title} [SEP] {(p.abstract or '')}" for p in papers]
        scores = _score_with_bge_reranker(self._bge_reranker_model_path, query_side, doc_texts)
        if scores is None or len(scores) != len(papers):
            ordered_indices = list(range(len(papers)))
            scores = [0.0] * len(papers)
        else:
            ordered_indices = sorted(range(len(papers)), key=lambda i: scores[i], reverse=True)
        reranked = []
        for r, idx in enumerate(ordered_indices):
            p = papers[idx]
            scr = scores[idx] if scores else (len(papers) - r)
            reranked.append(Paper(paper_id=p.paper_id, title=p.title, abstract=p.abstract, authors=p.authors,
                                  categories=p.categories, published=p.published, embedding=p.embedding, score=float(scr)))
        return reranked[:top_k] if top_k else reranked

    def _rerank_llm(self, papers: List[Paper], query: str, user: UserProfile, top_k: Optional[int]) -> List[Paper]:
        messages = build_rerank_messages(user=user, query=query, papers=papers, max_abstract_chars=self._max_abstract_chars)
        try:
            llm = self._get_llm()
            response = llm.chat(messages, temperature=self._temperature, max_tokens=self._max_tokens)
            ordered_indices, reasons_map, summary = parse_rerank_response(response, len(papers))
            self._last_llm_reasons = {"reasons_map": reasons_map, "summary": summary, "raw": response}
        except Exception as e:
            logger.warning("LLM rerank failed: %s, keeping original order", e)
            ordered_indices = list(range(len(papers)))
            self._last_llm_reasons = {"reasons_map": {}, "summary": "", "raw": ""}
        used = set(ordered_indices)
        reranked = []
        for idx in ordered_indices:
            p = papers[idx]
            reranked.append(Paper(paper_id=p.paper_id, title=p.title, abstract=p.abstract, authors=p.authors,
                                  categories=p.categories, published=p.published, embedding=p.embedding, score=0.0))
        for i, p in enumerate(papers):
            if i not in used:
                reranked.append(Paper(paper_id=p.paper_id, title=p.title, abstract=p.abstract, authors=p.authors,
                                      categories=p.categories, published=p.published, embedding=p.embedding, score=0.0))
        for i, p in enumerate(reranked):
            p.score = max(0.0, len(reranked) - i)
        return reranked[:top_k] if top_k else reranked
