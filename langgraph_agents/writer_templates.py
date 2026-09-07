"""Writer intent templates (design §10.1). Injected as extra system text."""

TEMPLATES = {
    "recommend": (
        "Style: numbered paper list. For each item give a short reason from the title/abstract. "
        "Do not pretend you read the full PDF. Do not invent papers that are not in the provided list."
    ),
    "daily": (
        "Style: personalized daily list. Numbered items with one-line reasons. "
        "Do not invent titles. Do not pretend full-text reading."
    ),
    "explain": (
        "Style: narrative explanation with citations like [1]. "
        "Do not output a bare shopping-list of papers. End without a handmade bibliography "
        "(the system appends references)."
    ),
    "compare": (
        "Style: structured comparison (method / data / findings). "
        "Both sides MUST appear with citations. Do not merge them into one uncited paragraph."
    ),
    "survey": (
        "Style: sectioned overview. State explicitly that this is not exhaustive. Cite with [N]."
    ),
    "factoid": (
        "Style: one or two sentences. If the evidence does not support the claim, say you are uncertain. "
        "Do not pivot into a recommendation list."
    ),
    "followup": (
        "Focus on the referenced paper(s) already in the session. Do not start a new retrieval narrative."
    ),
    "chitchat": "Reply briefly. Do not retrieve or invent papers.",
    "meta": "Explain what you can do: recommend papers, explain, compare methods, daily recs. Be short.",
    "feedback": "Acknowledge the preference update in one or two sentences.",
}


def template_for(intent: str) -> str:
    return TEMPLATES.get(intent) or TEMPLATES["explain"]
