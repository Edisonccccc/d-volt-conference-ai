"""Per-seller company context for AI prompt injection.

Each user belongs to a seller-company (User.company). When that company's
reps analyze a customer card or summarize a conversation, the AI prompts
need to know what THE SELLER does — what they sell, to whom, what their
positioning is — so it can generate pain points, opening questions, and
follow-up emails that actually fit the seller's go-to-market.

This module fetches that context once per seller via Claude+web_search
and caches the result in the ``company_profiles`` table. Subsequent
calls hit the cache. The fetch is automatic and lazy: the first card
analysis from a new seller-company triggers it, all subsequent reps from
that same company reuse the cached blob.

The blob is plain prose, fed verbatim into other prompts. Keeping it as
a single string (instead of structured fields) means we can grow what
goes into the context without schema churn.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from . import storage
from .costing import anthropic_cost
from .llm import make_client, stream_to_terminal


log = logging.getLogger("conference-ai.company-profile")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")


SYSTEM = (
    "You research B2B companies and produce concise, factual profiles. "
    "You do not editorialize or speculate. If web searches don't surface "
    "authoritative information about a company, you say so explicitly "
    "rather than guess."
)


def _ws_tool(max_uses: int = 3) -> dict:
    return {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }


SUBMIT_TOOL = {
    "name": "submit_seller_profile",
    "description": (
        "Deliver a concise factual profile of the seller's own company. "
        "Call exactly once after gathering enough info from web search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "context_blob": {
                "type": "string",
                "description": (
                    "An 80-150 word factual paragraph about the company. "
                    "Cover: what the company does, main products/services, "
                    "and target customers (industry/segment). This text "
                    "is fed verbatim into other AI prompts as background "
                    "context, so make every word count. If web search "
                    "found little authoritative info, say so explicitly "
                    "in this blob — do not invent details."
                ),
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URLs you actually consulted.",
            },
        },
        "required": ["context_blob"],
    },
}


def _prompt(company_name: str) -> str:
    return f"""\
Look up the company "{company_name}" using web search. Then produce a brief
profile that will be used as background context for THAT company's sales
reps when they analyze customer profiles in our sales tool.

Use up to 3 searches. Then call `submit_seller_profile` exactly once.

The `context_blob` (80-150 words) should cover:
- What the company does, in one sentence.
- Their main products or service lines.
- Their target customers (industries, segments, deployment scale).
- Anything distinctive about their positioning or differentiation.

Be factual. If you can't find authoritative info (small/private/obscure
company), state that explicitly inside the blob — e.g. "Limited public
information found about {company_name}; based on available signals…".
Never invent products or markets.
"""


def fetch_company_profile(company_name: str) -> Tuple[str, list, float]:
    """Run a Claude+web_search call. Returns ``(blob, sources, cost_usd)``.

    On error or empty company name, returns ``("", [], 0.0)`` rather than
    raising — the caller treats no-context as "fall back to prompt
    defaults" rather than "fail the pipeline".
    """
    name = (company_name or "").strip()
    if not name:
        return "", [], 0.0

    client = make_client(timeout=90.0)
    try:
        final = stream_to_terminal(
            client,
            label="seller-profile",
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM,
            tools=[_ws_tool(3), SUBMIT_TOOL],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": _prompt(name)}],
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("fetch_company_profile failed for %r: %s", name, exc)
        return "", [], 0.0

    cost = anthropic_cost(getattr(final, "usage", None), MODEL)

    for block in final.content:
        if (getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "submit_seller_profile"):
            data = block.input or {}
            blob = (data.get("context_blob") or "").strip()
            sources = list(data.get("sources") or [])
            return blob, sources, cost

    log.warning(
        "fetch_company_profile: model didn't call submit_seller_profile "
        "for %r — no profile cached.", name,
    )
    return "", [], cost


def get_or_fetch_company_profile(company_name: Optional[str]) -> Optional[str]:
    """Return a profile blob string for the seller-company.

    Cache hit → return cached blob (instant).
    Cache miss → run web-search fetch, cache it, return blob.
    Empty/None input → return None (caller falls back to env default).

    We cache even an empty result on a successful API call to avoid
    re-spending on every retry; manager-triggered refresh is the way
    to recover from a bad first lookup (when we add that endpoint).
    """
    name = (company_name or "").strip()
    if not name:
        return None

    cached = storage.get_company_profile(name)
    if cached is not None:
        return cached.get("context_blob") or ""

    log.info("seller-profile: fetching for %r (first time)", name)
    blob, sources, cost = fetch_company_profile(name)
    storage.set_company_profile(name, blob, sources, cost)
    return blob
