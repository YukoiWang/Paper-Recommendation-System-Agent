# -*- coding: utf-8 -*-
"""
Prompt templates for listwise data generation.
Structure: (1) user profile + query generation, (2) candidate paper scoring (0/1/2).
"""

# =============================================================================
# 1. Generate user profile and query from paper info
# =============================================================================

SYSTEM_PROFILE_QUERY = """You are an assistant that generates user profiles and search queries for an academic recommendation scenario.
Given a paper's author info, field, and topics, produce a plausible "user profile" and one "user query".
The profile should match the paper's author identity and field; the query should be generalizable and not overly formal or academic in tone."""

USER_PROFILE_QUERY_TEMPLATE = """## Paper info
- Title: {title}
- Abstract (first 300 chars): {abstract}
- Author info:
{author_info}
- Field / topics:
{field_topic}
- Available tags from this paper (choose ONLY from this list):
{available_tags}

## Requirements
1. **User profile** (JSON only, no extra explanation):
   - position: role or title (e.g., PhD student, master student, researcher, engineer, faculty), consistent with author info
   - goal: purpose (e.g., writing a paper, finding related work, learning a direction, doing a project)
   - professional_level: level (e.g., beginner, intermediate, expert)
   - interest_field: a LIST of 3-5 interest tags, each item MUST be copied from the \"Available tags\" list above (no new tags)

2. **User query**: One natural-language query with variety. Examples:
   - "Recommend papers similar to this one"
   - "Recommend some papers in <field> that I can use to get started"
   - "I want to learn about <topic/method>, any recommendations?"
   - "Any papers related to <topic> that are good for beginners?"
   Vary the phrasing; avoid always starting with "Please recommend...". Tone can be casual or neutral.

Output strictly the following JSON (no markdown code block, no other text):
{{"user_profile": {{"position": "...", "goal": "...", "professional_level": "...", "interest_field": ["...", "..."]}}, "query": "..."}}"""


def build_profile_query_prompt(
    title: str,
    abstract: str,
    author_info: str,
    field_topic: str,
    available_tags: list,
) -> list:
    """Build messages (system + user) for generating user profile and query."""
    abstract_preview = (abstract or "")[:300]
    if (abstract or "") and len(abstract or "") > 300:
        abstract_preview += "..."
    tags = [str(x).strip() for x in (available_tags or []) if str(x).strip()]
    # keep prompt compact
    tags = tags[:15]
    tags_block = "\n".join([f"- {t}" for t in tags]) if tags else "- (none)"
    user_content = USER_PROFILE_QUERY_TEMPLATE.format(
        title=title or "(no title)",
        abstract=abstract_preview,
        author_info=author_info or "(no author info)",
        field_topic=field_topic or "(no field info)",
        available_tags=tags_block,
    )
    return [
        {"role": "system", "content": SYSTEM_PROFILE_QUERY},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# 2. Score candidate papers 0/1/2 (relevance to query and user profile)
# =============================================================================

SYSTEM_SCORE_CANDIDATES = """You are a relevance judge for a paper recommendation system.
Given a user profile and query, score each candidate paper in the list for relevance.
Score meaning:
- 2: Highly relevant to query and user profile, strongly recommend
- 1: Somewhat relevant, recommend
- 0: Largely irrelevant or weak relevance, do not recommend
Output only the scores, no explanation."""

USER_SCORE_CANDIDATES_TEMPLATE = """## User profile
- position: {position}
- goal: {goal}
- professional_level: {professional_level}
- interest_field: {interest_field}

## User query
{query}

## Candidate papers ({n} total)
{candidates_block}

## Output
Score each candidate above with 0, 1, or 2.
Output strictly the following JSON (no markdown, no other text):
{{"scores": [s1, s2, s3, ...]}}
where scores is an array of length {n}, in order for candidate 1 to candidate {n}, each element 0, 1, or 2."""


def _build_candidates_block(candidates: list, max_abstract_chars: int = 300) -> str:
    """Format candidate paper list into a string block."""
    lines = []
    for i, c in enumerate(candidates, 1):
        title = (c.get("title") or "").strip()
        abstract = (c.get("abstract") or "")[:max_abstract_chars]
        if (c.get("abstract") or "") and len(c.get("abstract") or "") > max_abstract_chars:
            abstract += "..."
        authors = ", ".join((c.get("authors") or [])[:4]) or "(unknown)"
        lines.append(
            f"### Candidate {i}\n"
            f"- Title: {title}\n"
            f"- Abstract: {abstract}\n"
            f"- Authors: {authors}\n"
        )
    return "\n".join(lines)


def build_score_candidates_prompt(
    user_profile: dict,
    query: str,
    candidates: list,
    max_abstract_chars: int = 300,
) -> list:
    """Build messages for scoring candidate papers 0/1/2."""
    profile = user_profile or {}
    position = profile.get("position", "")
    goal = profile.get("goal", "")
    professional_level = profile.get("professional_level", "")
    interest_field = profile.get("interest_field", "")
    if isinstance(interest_field, list):
        interest_field = ", ".join([str(x).strip() for x in interest_field if str(x).strip()])
    n = len(candidates)
    candidates_block = _build_candidates_block(candidates, max_abstract_chars)
    user_content = USER_SCORE_CANDIDATES_TEMPLATE.format(
        position=position,
        goal=goal,
        professional_level=professional_level,
        interest_field=interest_field,
        query=query or "",
        n=n,
        candidates_block=candidates_block,
    )
    return [
        {"role": "system", "content": SYSTEM_SCORE_CANDIDATES},
        {"role": "user", "content": user_content},
    ]
