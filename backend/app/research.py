"""Company research agent. Claude + web_search server tool -> structured brief.

Optimization (slice 5): instead of one ~40s sequential research call with up
to 5 chained web searches, we fan out into THREE concurrent calls that each
do focused work — basics + classification, recent news, and contact LinkedIn.
The three run in a ThreadPoolExecutor and cost ~the same in dollars but
collapse wall-clock to ~max(call) ≈ 12-15s.

Falls back to a search-less brief if web_search is unavailable, so the
pipeline always produces something.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Tuple

from anthropic import APIStatusError

from .costing import anthropic_cost, count_web_search_calls
from .llm import make_client, stream_to_terminal
from .models import CompanyResearch, ExtractedCard


log = logging.getLogger("conference-ai.research")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

# Last-resort fallback when no company_context blob is available (e.g.
# the rep's seller-profile fetch failed and there's no env override).
# Kept generic enough to work for any seller; the prompt still flows.
DEFAULT_COMPANY_CONTEXT = (
    "(No background context for the seller is available. Rely on the "
    "customer card data alone and keep recommendations general.)"
)

# Cache_control caches the `tools` block + system prefix across calls.
# Repeated cards on the same model hit the cache for ~10% of normal input.
CACHE = {"type": "ephemeral"}


def _ws_tool(max_uses: int) -> dict:
    return {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }


# ---- Tool schemas --------------------------------------------------------

# Tool 1: company basics + classification + pain points + opening questions.
SUBMIT_BASICS = {
    "name": "submit_company_basics",
    "description": (
        "Deliver company facts and pre-meeting talking points. Call exactly "
        "once after gathering enough info from web search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company_website": {
                "type": "string",
                "description": "The company's official website (root URL).",
            },
            "one_liner": {
                "type": "string",
                "description": "One-sentence description of the company.",
            },
            "company_category": {
                "type": "string",
                "description": (
                    "A short label describing where this company sits "
                    "relative to the seller's go-to-market — pick whatever "
                    "fits the seller's industry and channels. Common "
                    "examples across industries: 'Customer', 'Channel "
                    "Partner', 'Reseller', 'Distributor', 'System "
                    "Integrator', 'Supplier', 'Competitor', 'Other'. "
                    "Keep it 1-3 words."
                ),
            },
            "category_rationale": {
                "type": "string",
                "description": "One short sentence on why this category was chosen.",
            },
            "industry": {"type": "string"},
            "estimated_size": {
                "type": "string",
                "description": "Headcount range or revenue band if known.",
            },
            "products": {
                "type": "array", "items": {"type": "string"},
                "description": "Their main products or service lines.",
            },
            "pain_points": {
                "type": "array", "items": {"type": "string"},
                "description": (
                    "Likely pain points framed against the seller's offering. "
                    "Be specific; tie to something observed in the research."
                ),
            },
            "opening_questions": {
                "type": "array", "items": {"type": "string"},
                "description": (
                    "Exactly three short opening questions specific to the "
                    "research findings (not generic)."
                ),
            },
            "sources": {
                "type": "array", "items": {"type": "string"},
                "description": "URLs cited.",
            },
        },
        "required": ["one_liner"],
    },
}

# Tool 2: recent news bullets (last ~90 days).
SUBMIT_NEWS = {
    "name": "submit_recent_news",
    "description": "Return bullet-style notes on company news from the last ~90 days.",
    "input_schema": {
        "type": "object",
        "properties": {
            "recent_news": {
                "type": "array", "items": {"type": "string"},
                "description": (
                    "Bullet notes on news from roughly the last 90 days. "
                    "Each bullet should be one short sentence."
                ),
            },
            "sources": {
                "type": "array", "items": {"type": "string"},
                "description": "URLs cited.",
            },
        },
        "required": [],
    },
}

# Tool 3: verified contact LinkedIn + title.
SUBMIT_CONTACT = {
    "name": "submit_contact_info",
    "description": (
        "Deliver the contact's PERSONAL LinkedIn URL and verified title. "
        "Leave fields blank if not confidently found."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "contact_linkedin": {
                "type": "string",
                "description": (
                    "The contact's PERSONAL LinkedIn profile URL "
                    "(linkedin.com/in/<handle>). Not the company's page."
                ),
            },
            "contact_title_verified": {
                "type": "string",
                "description": (
                    "The contact's current title from LinkedIn or the "
                    "company website. Leave blank if not found."
                ),
            },
            "sources": {
                "type": "array", "items": {"type": "string"},
                "description": "URLs cited.",
            },
        },
        "required": [],
    },
}


SYSTEM_BASE = (
    "You are a sharp B2B sales analyst. You research companies and produce "
    "concise, sourced facts. You prefer specifics over fluff."
)


def _system_block() -> list:
    """System block with cache_control so the static prefix gets cached."""
    return [{"type": "text", "text": SYSTEM_BASE, "cache_control": CACHE}]


def _contact_lines(card: ExtractedCard) -> str:
    rows = []
    for label, val in [
        ("Name", card.name), ("Title", card.title), ("Company", card.company),
        ("Website", card.website), ("Email", ", ".join(card.emails or [])),
        ("Address", card.address), ("LinkedIn", card.linkedin),
    ]:
        if val:
            rows.append(f"{label}: {val}")
    return "\n".join(rows) if rows else "(no fields extracted)"


# ---- Sub-call prompts ----------------------------------------------------

def _basics_prompt(
    card: ExtractedCard, company_context: str, seller_company: str,
) -> str:
    return f"""\
Your seller is at {seller_company}. Their context:
{company_context}

Contact info from a business card:
{_contact_lines(card)}

GOAL: Produce a focused company brief. Use up to 3 web searches to find:
- The company's official website (verify, don't guess).
- What the company does (one-liner, industry, size if public, products).
- Any signals that hint at problems the seller's offering would solve.

Then call `submit_company_basics` exactly once with:
- `company_category`: a short 1-3 word label fitting the seller's go-to-market
  (Customer / Channel Partner / Reseller / Distributor / Integrator /
  Supplier / Competitor / Other — or whatever's idiomatic for the seller's
  industry). State the reasoning in `category_rationale`.
- `pain_points`: likely concerns framed against the seller's offering,
  tied to something concrete you observed in the research.
- `opening_questions`: exactly THREE short questions specific to what you
  found, not generic.
"""


def _news_prompt(card: ExtractedCard) -> str:
    return f"""\
You are gathering recent news for a sales pre-meeting brief.

Contact info:
{_contact_lines(card)}

GOAL: Find news from roughly the last 90 days about the company. Run up to
2 web searches. Then call `submit_recent_news` exactly once with one-sentence
bullets in `recent_news`. Empty list is fine if you find nothing relevant.
"""


def _contact_prompt(card: ExtractedCard) -> str:
    return f"""\
You are verifying a sales contact's LinkedIn profile and current title.

Contact info:
{_contact_lines(card)}

GOAL:
- Find this PERSON's PERSONAL LinkedIn profile (linkedin.com/in/<handle>).
  Search 'site:linkedin.com/in <name> <company>'. Do NOT return the
  company's LinkedIn page. Leave `contact_linkedin` blank unless you can
  confidently match this specific person.
- Find their current title (LinkedIn or the company team/about page).
  Put it in `contact_title_verified` (may confirm or correct the card).

Run up to 2 web searches, then call `submit_contact_info` exactly once.
"""


# ---- Sub-call execution helpers -----------------------------------------

def _parse_tool_input(final: Any, tool_name: str) -> Optional[dict]:
    for block in getattr(final, "content", []) or []:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == tool_name
        ):
            try:
                return dict(block.input)
            except Exception:  # noqa: BLE001
                log.exception("failed to parse %s input", tool_name)
                return None
    return None


def _call_with_search(
    client, *, label: str, prompt: str, max_uses: int,
    submit_tool: dict, tool_name: str,
) -> Tuple[dict, float]:
    """Run one fan-out call (web_search + structured submit). Returns (data, cost)."""
    try:
        # cache_control is set on the LAST tool — Anthropic caches the entire
        # tools block + the system prefix above it.
        cached_submit = {**submit_tool, "cache_control": CACHE}
        final = stream_to_terminal(
            client,
            label=label,
            model=MODEL,
            max_tokens=2048,
            system=_system_block(),
            tools=[_ws_tool(max_uses), cached_submit],
            messages=[{"role": "user", "content": prompt}],
        )
        cost = anthropic_cost(
            getattr(final, "usage", None), MODEL,
            web_search_calls=count_web_search_calls(final),
        )
        data = _parse_tool_input(final, tool_name) or {}
        return data, cost
    except APIStatusError as exc:
        log.warning("[%s] APIStatusError %s — %s",
                    label, getattr(exc, "status_code", "?"), str(exc)[:200])
        return {}, 0.0
    except Exception:  # noqa: BLE001
        log.exception("[%s] sub-call hit unexpected error", label)
        return {}, 0.0


def _research_no_search(
    client, card: ExtractedCard, company_context: str, seller_company: str,
):
    """Fallback when web_search is unavailable (or the parallel calls all failed).
    Returns ``(CompanyResearch, cost_usd)``. No search; uses the model's general
    knowledge alone."""
    prompt = (
        _basics_prompt(card, company_context, seller_company)
        + "\n\nIMPORTANT: web_search is unavailable for this run. Produce "
        "the brief from the contact info alone. Qualify guesses with "
        "'Likely…' or 'Possibly…'. Three opening questions are still required."
    )
    try:
        final = stream_to_terminal(
            client, label="research-fallback",
            model=MODEL, max_tokens=2048,
            system=_system_block(),
            tools=[{**SUBMIT_BASICS, "cache_control": CACHE}],
            tool_choice={"type": "tool", "name": "submit_company_basics"},
            messages=[{"role": "user", "content": prompt}],
        )
        cost = anthropic_cost(getattr(final, "usage", None), MODEL)
        data = _parse_tool_input(final, "submit_company_basics") or {}
    except Exception as exc:  # noqa: BLE001
        log.exception("fallback research failed")
        return CompanyResearch(
            one_liner=f"Research failed: {exc.__class__.__name__}: {exc}"
        ), 0.0
    return _merge_into_research(basics=data, news={}, contact={}), cost


def _merge_into_research(
    *, basics: dict, news: dict, contact: dict,
) -> CompanyResearch:
    """Combine the three sub-call dicts into a single CompanyResearch."""
    sources = (
        (basics.get("sources")  or []) +
        (news.get("sources")    or []) +
        (contact.get("sources") or [])
    )
    # Dedup while preserving order.
    seen = set()
    uniq_sources = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            uniq_sources.append(s)

    return CompanyResearch(
        contact_linkedin=contact.get("contact_linkedin") or None,
        contact_title_verified=contact.get("contact_title_verified") or None,
        company_website=basics.get("company_website") or None,
        one_liner=basics.get("one_liner") or None,
        company_category=basics.get("company_category") or None,
        category_rationale=basics.get("category_rationale") or None,
        industry=basics.get("industry") or None,
        estimated_size=basics.get("estimated_size") or None,
        products=basics.get("products") or [],
        recent_news=news.get("recent_news") or [],
        pain_points=basics.get("pain_points") or [],
        opening_questions=basics.get("opening_questions") or [],
        sources=uniq_sources,
    )


# ---- Public entrypoint ---------------------------------------------------

def research_company(
    card: ExtractedCard,
    company_context: Optional[str] = None,
    seller_company: Optional[str] = None,
):
    """Run the research agent (3-way fan-out) and return ``(CompanyResearch, cost_usd)``.

    Three concurrent web_search calls (basics / news / contact) collapse the
    sequential ~40s wall time into ~max(call) ≈ 12-15s. Cost is roughly
    the same as the old single-call path.

    `seller_company` and `company_context` together describe WHO THE SELLER
    IS so the AI can frame pain points and opening questions to fit the
    seller's go-to-market. Both are looked up by the pipeline from the
    card's owning user before this function is called.
    """
    if not card.company:
        return CompanyResearch(
            one_liner="No company name was extracted from the card; research skipped."
        ), 0.0

    client = make_client(timeout=120.0)
    company_context = company_context or os.getenv(
        "COMPANY_CONTEXT", DEFAULT_COMPANY_CONTEXT,
    )
    seller_company = (seller_company or "the seller").strip() or "the seller"

    log.info(
        "research: starting 3-way fan-out for company=%r (seller=%r)",
        card.company, seller_company,
    )

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_basics  = ex.submit(
            _call_with_search, client,
            label="research-basics",
            prompt=_basics_prompt(card, company_context, seller_company),
            max_uses=3,
            submit_tool=SUBMIT_BASICS,
            tool_name="submit_company_basics",
        )
        f_news    = ex.submit(
            _call_with_search, client,
            label="research-news",
            prompt=_news_prompt(card),
            max_uses=2,
            submit_tool=SUBMIT_NEWS,
            tool_name="submit_recent_news",
        )
        f_contact = ex.submit(
            _call_with_search, client,
            label="research-contact",
            prompt=_contact_prompt(card),
            max_uses=2,
            submit_tool=SUBMIT_CONTACT,
            tool_name="submit_contact_info",
        )
        basics,  basics_cost  = f_basics.result()
        news,    news_cost    = f_news.result()
        contact, contact_cost = f_contact.result()

    total_cost = basics_cost + news_cost + contact_cost
    merged = _merge_into_research(basics=basics, news=news, contact=contact)

    # Last-resort fallback: if every parallel call came back empty (e.g., the
    # account doesn't have web_search at all), run a single search-less call.
    if not merged.one_liner:
        log.warning("all three parallel research calls returned empty; "
                    "running search-less fallback")
        fb_research, fb_cost = _research_no_search(
            client, card, company_context, seller_company,
        )
        return fb_research, total_cost + fb_cost

    log.info(
        "research: fan-out done. cost = basics $%.4f + news $%.4f + contact $%.4f = $%.4f",
        basics_cost, news_cost, contact_cost, total_cost,
    )
    return merged, total_cost
