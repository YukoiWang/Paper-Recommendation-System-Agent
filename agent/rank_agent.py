"""Dual-mode reranker: LLM rerank or finetuned BGE reranker. Combines user profile and query.

Modes:
  - llm: LLM returns comma-separated paper indices (default).
  - bge_reranker: Score (query+user_context, doc) with a finetuned BGE reranker
    (AutoModelForSequenceClassification). Pass bge_reranker_model_path.
"""
from __future__ import annotations
import logging
import re
from typing import Dict, List, Literal, Optional

from agent.models import Paper, UserProfile
from rerank_prompt import build_rerank_messages, parse_rerank_response

logger = logging.getLogger(__name__)

# Default path; same as scripts/train_reranker OUTPUT_DIR. Replace when finetuned.
DEFAULT_BGE_RERANKER_PATH = "./output/bge-finetuned"

# "llm": use LLM to output ordering; "bge_reranker": use finetuned BGE reranker (query-doc scores)
RankMode = Literal["llm", "bge_reranker"]


def _default_llm_client(api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com",
                       model: str = "deepseek-chat", **kwargs):
    """Lazy import to avoid requiring openai when not using LLM rerank."""
    try:
        from qa_agent import LLMClient
    except ImportError:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        # Minimal inline client if qa_agent not available
        client = OpenAI(api_key=api_key or "", base_url=base_url, timeout=kwargs.get("timeout", 60.0))
        return _OpenAIWrapper(client, model=model, **kwargs)
    return LLMClient(api_key=api_key or "", base_url=base_url, model=model, **kwargs)


class _OpenAIWrapper:
    """Thin wrapper when qa_agent.LLMClient is not used."""
    def __init__(self, client, model: str = "deepseek-chat", temperature: float = 0.3, max_tokens: int = 1024):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[dict], **kwargs) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return (r.choices[0].message.content or "").strip()


def _parse_llm_ordering(response: str, n: int) -> List[int]:
    """Parse LLM output to 0-based indices. Expects comma-separated numbers or '1. 3' style."""
    response = response.strip()
    # Try comma-separated numbers first
    numbers = re.findall(r"\d+", response)
    if not numbers:
        return list(range(min(n, 50)))
    indices = []
    seen = set()
    for num in numbers:
        idx = int(num) - 1  # 1-based to 0-based
        if 0 <= idx < n and idx not in seen:
            indices.append(idx)
            seen.add(idx)
    if not indices:
        return list(range(min(n, 50)))
    return indices


def _score_with_bge_reranker(
    model_path: str,
    query: str,
    doc_texts: List[str],
    batch_size: int = 32,
) -> Optional[List[float]]:
    """
    Load finetuned BGE reranker from path and score (query, doc) pairs.
    Uses AutoModelForSequenceClassification (same format as train_reranker output).
    Returns list of scores, or None if model not available (e.g. not finetuned yet).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        logger.warning("transformers/torch not installed; BGE reranker unavailable: %s", e)
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()

        pairs = [[query, d] for d in doc_texts]
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            with torch.no_grad():
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                outputs = model(**inputs)
                scores = outputs.logits.view(-1).float().cpu().numpy()
            all_scores.extend([float(s) for s in scores])
        return all_scores
    except Exception as e:
        logger.warning("BGE reranker load/predict failed (model may not be finetuned yet): %s", e)
        return None


class RankAgent:
    """
    Rerank papers by query + user profile. Two modes:
    - "llm": LLM outputs ordering (comma-separated indices).
    - "bge_reranker": Finetuned BGE reranker scores (query, doc) pairs; pass bge_reranker_model_path.
    Input: papers, query, user. Output: reranked list with .score set by rank.
    """

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
        """
        Last LLM rerank explanations (if mode == 'llm' and response parsed).
        Keys:
          - 'reasons_map': Dict[int, List[str]] (0-based index -> reasons)
          - 'summary': str
          - 'raw': str (raw LLM response)
        """
        return self._last_llm_reasons

    def _get_llm(self):
        if self._llm is None:
            self._llm = _default_llm_client(
                api_key=self._api_key,
                base_url=self._base_url,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        return self._llm

    def rerank(
        self,
        papers: List[Paper],
        query: str,
        user: UserProfile,
        top_k: Optional[int] = None,
    ) -> List[Paper]:
        """
        Rerank papers by semantic relevance to query and user profile.

        Mode "llm": uses LLM to output ordering.
        Mode "bge_reranker": uses finetuned BGE reranker (query-doc) to score and sort.

        Args:
            papers: Candidate papers (order will be changed).
            query: User's current query.
            user: User profile (interests, categories, authors, liked items).
            top_k: If set, return only the top_k after reranking; else return all in new order.

        Returns:
            Reranked list of papers with .score set by rank.
        """
        if not papers:
            return []
        if len(papers) == 1:
            p = papers[0]
            out = [Paper(paper_id=p.paper_id, title=p.title, abstract=p.abstract, authors=p.authors,
                         categories=p.categories, published=p.published, embedding=p.embedding, score=1.0)]
            if top_k is not None:
                out = out[:top_k]
            return out

        if self._mode == "bge_reranker":
            return self._rerank_bge_reranker(papers, query, user, top_k)
        return self._rerank_llm(papers, query, user, top_k)

    def _rerank_bge_reranker(
        self,
        papers: List[Paper],
        query: str,
        user: UserProfile,
        top_k: Optional[int],
    ) -> List[Paper]:
        """Rerank using finetuned BGE reranker: (query + user context, doc) -> score, then sort."""
        profile_text = ""
        try:
            from rerank_prompt import _build_user_profile_text as _profile
            profile_text = _profile(user)
        except Exception:
            profile_text = ""
        query_side = query.strip()
        if profile_text:
            query_side = query_side + "\n" + profile_text
        # Match train_reranker format: title [SEP] abstract
        doc_texts = [f"{p.title} [SEP] {(p.abstract or '')}" for p in papers]

        scores = _score_with_bge_reranker(
            self._bge_reranker_model_path, query_side, doc_texts
        )
        if scores is None or len(scores) != len(papers):
            logger.warning(
                "bge_reranker mode: model unavailable or score length mismatch; keeping original order"
            )
            ordered_indices = list(range(len(papers)))
            scores = [0.0] * len(papers)
        else:
            ordered_indices = sorted(range(len(papers)), key=lambda i: scores[i], reverse=True)

        reranked = []
        for r, idx in enumerate(ordered_indices):
            p = papers[idx]
            scr = scores[idx] if scores else (len(papers) - r)
            reranked.append(Paper(
                paper_id=p.paper_id,
                title=p.title,
                abstract=p.abstract,
                authors=p.authors,
                categories=p.categories,
                published=p.published,
                embedding=p.embedding,
                score=float(scr),
            ))
        if top_k is not None:
            reranked = reranked[:top_k]
        logger.info("Reranked %s papers with bge_reranker (top_k=%s)", len(reranked), top_k)
        return reranked

    def _rerank_llm(
        self,
        papers: List[Paper],
        query: str,
        user: UserProfile,
        top_k: Optional[int],
    ) -> List[Paper]:
        """Rerank using LLM: prompt with profile + query + papers, parse ordering from response."""
        messages = build_rerank_messages(
            user=user,
            query=query,
            papers=papers,
            max_abstract_chars=self._max_abstract_chars,
        )
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
            reranked.append(Paper(
                paper_id=p.paper_id,
                title=p.title,
                abstract=p.abstract,
                authors=p.authors,
                categories=p.categories,
                published=p.published,
                embedding=p.embedding,
                score=0.0,
            ))
        for i, p in enumerate(papers):
            if i not in used:
                reranked.append(Paper(
                    paper_id=p.paper_id, title=p.title, abstract=p.abstract,
                    authors=p.authors, categories=p.categories, published=p.published,
                    embedding=p.embedding, score=0.0,
                ))

        for i, p in enumerate(reranked):
            p.score = max(0.0, len(reranked) - i)

        if top_k is not None:
            reranked = reranked[:top_k]
        logger.info("Reranked %s papers with LLM (top_k=%s)", len(reranked), top_k)
        return reranked