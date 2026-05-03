"""Company research agent. Claude + web_search server tool -> structured brief.

Falls back to a search-less brief if web_search is unavailable on the
account or the call errors out, so the pipeline always produces something.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from anthropic import APIStatusError

from .llm import make_client, stream_to_terminal
from .models import CompanyResearch, ExtractedCard


log = logging.getLogger("conference-ai.research")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

DEFAULT_COMPANY_CONTEXT = (
    "d-volt sells voltage regulation and power conditioning hardware to "
    "industrial and commercial customers. Frame pain points and openings "
    "around power quality, energy savings, equipment reliability, and "
    "downtime reduction."
)

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}

SUBMIT_TOOL = {
    "name": "submit_research_brief",
    "description": (
        "Deliver the final research brief to the salesperson. Call this once, "
        "after web research is complete."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "contact_linkedin": {
                "type": "string",
                "description": (
                    "The contact's PERSONAL LinkedIn profile URL "
                    "(linkedin.com/in/<handle>). Use search like 'site:linkedin.com/in <name> "
                    "<company>'. Leave blank if you cannot find a confident "
                    "match for this specific person."
                ),
            },
            "contact_title_verified": {
                "type": "string",
                "description": (
                    "The contact's current title from LinkedIn or the "
                    "company website. May confirm or correct the title on "
                    "the card. Leave blank if not found."
                ),
            },
            "company_website": {
                "type": "string",
                "description": (
                    "The company's official website (root URL). Verify "
                    "via web search rather than relying on the card alone."
                ),
            },
            "one_liner": {
                "type": "string",
                "description": "One-sentence description of the company.",
            },
            "company_category": {
                "type": "string",
                "enum": [
                    "Utility",
                    "Vendor",
                    "EPC",
                    "Sales Representatives",
                    "Distributors",
                    "End Users",
                    "Other",
                ],
                "description": (
                    "Classify the company relative to d-volt's market. "
                    "Utility = electric/power utility. "
                    "Vendor = supplier of components or equipment. "
                    "EPC = engineering/procurement/construction firm. "
                    "Sales Representatives = independent reps selling power "
                    "products. Distributors = resellers / channel partners. "
                    "End Users = the actual users of d-volt's hardware "
                    "(industrial sites, commercial buildings, data centers). "
                    "Other = anything else, including competitors."
                ),
            },
            "category_rationale": {
                "type": "string",
                "description": (
                    "One short sentence explaining why you chose that "
                    "category. If 'Other', name the specific reason "
                    "(e.g. 'competitor of d-volt')."
                ),
            },
            "industry": {"type": "string"},
            "estimated_size": {
                "type": "string",
                "description": "Headcount range or revenue band if known.",
            },
            "products": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Their main products or service lines.",
            },
            "recent_news": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Bullet notes on news from roughly the last 90 days. "
                    "Each bullet should be one short sentence."
                ),
            },
            "pain_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Likely pain points framed against the seller's offering. "
                    "Be specific; tie each to something observed in the research."
                ),
            },
            "opening_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exactly three short questions the salesperson can open with."
                ),
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URLs cited.",
            },
        },
        "required": ["one_liner"],
    },
}

SYSTEM = (
    "You are a sharp B2B sales analyst. You research companies and produce "
    "concise, sourced briefs. You prefer specifics over fluff."
)


def _build_prompt(card: ExtractedCard, company_context: str) -> str:
    contact_lines = []
    if card.name:
        contact_lines.append(f"Name: {card.name}")
    if card.title:
        contact_lines.append(f"Title: {card.title}")
    if card.company:
        contact_lines.append(f"Company: {card.company}")
    if card.website:
        contact_lines.append(f"Website: {card.website}")
    if card.emails:
        contact_lines.append(f"Email: {', '.join(card.emails)}")
    if card.address:
        contact_lines.append(f"Address: {card.address}")
    if card.linkedin:
        contact_lines.append(f"LinkedIn: {card.linkedin}")

    contact = "\n".join(contact_lines) if contact_lines else "(no fields extracted)"

    return f"""\
You are preparing a salesperson at d-volt for a meeting with the contact
described below. d-volt's context:

{company_context}

Contact info pulled from a business card:
{contact}

Use the web_search tool to gather information. Cover BOTH the contact and
their company:

About the contact:
- Find this PERSON's LinkedIn profile (linkedin.com/in/<handle>). Search
  '<name> <company> linkedin' or 'site:linkedin.com/in <name> <company>'.
  Do not return the company's LinkedIn page here. Only return the URL if
  you are confident it is the same person — otherwise leave it blank.
- Confirm or correct their current title from LinkedIn or the company's
  team/about page. Put the verified title in `contact_title_verified`.

About the company:
- Verify the company's official website URL (root domain).
- Classify the company in `company_category` as exactly one of:
  Utility | Vendor | EPC | Sales Representatives | Distributors | End Users | Other.
  Use 'Other' only when none of the first six fit, and put the specific
  label (e.g. 'competitor of d-volt') in `category_rationale`.
- Recent news from the last ~90 days.
- Products/services, headcount or revenue if public.
- Anything that hints at power-quality, reliability, or energy challenges.

When research is sufficient (no more than 5 searches), call
`submit_research_brief` exactly once with a complete brief. Be concrete and
sourced. If you cannot find the company, return what you can with empty
arrays for unknown fields.

Three opening questions are required and must be specific to what you found,
not generic.
"""


def _parse_brief(final, fallback_one_liner: str) -> Optional[CompanyResearch]:
    """Pull the submit_research_brief tool_use input out of a Message."""
    for block in final.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "submit_research_brief"
        ):
            try:
                return CompanyResearch.model_validate(block.input)
            except Exception:  # noqa: BLE001
                log.exception("submit_research_brief input failed schema validation")
                return CompanyResearch(
                    one_liner="Research completed but the brief failed validation.",
                    recent_news=[json.dumps(block.input)[:500]],
                )
    text = "\n".join(
        getattr(b, "text", "") for b in final.content if getattr(b, "type", None) == "text"
    ).strip()
    if text:
        return CompanyResearch(one_liner=text.splitlines()[0])
    return None


def _research_no_search(client, prompt: str) -> CompanyResearch:
    """Fallback: produce a brief without web search."""
    fallback_prompt = (
        prompt
        + "\n\nIMPORTANT: web_search is unavailable. Produce the brief from "
        "the contact info alone. Qualify any guesses with 'Likely…' or "
        "'Possibly…'. Three opening questions are still required."
    )
    final = stream_to_terminal(
        client,
        label="research-fallback",
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM,
        tools=[SUBMIT_TOOL],
        tool_choice={"type": "tool", "name": "submit_research_brief"},
        messages=[{"role": "user", "content": fallback_prompt}],
    )
    return _parse_brief(final, fallback_one_liner="No brief returned.") or CompanyResearch(
        one_liner="No structured brief returned by the model."
    )


def research_company(
    card: ExtractedCard,
    company_context: Optional[str] = None,
) -> CompanyResearch:
    """Run the research agent and return a structured brief.

    Tries web_search first, falls back to a search-less brief on any error.
    """
    if not card.company:
        return CompanyResearch(
            one_liner="No company name was extracted from the card; research skipped."
        )

    client = make_client(timeout=180.0)
    prompt = _build_prompt(
        card,
        company_context or os.getenv("COMPANY_CONTEXT", DEFAULT_COMPANY_CONTEXT),
    )

    # ---- attempt 1: with web_search --------------------------------------
    try:
        final = stream_to_terminal(
            client,
            label="research",
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=[WEB_SEARCH_TOOL, SUBMIT_TOOL],
            messages=[{"role": "user", "content": prompt}],
        )
    except APIStatusError as exc:
        log.warning(
            "research with web_search failed (status=%s, message=%s); "
            "falling back to search-less brief.",
            getattr(exc, "status_code", "?"),
            str(exc)[:200],
        )
        try:
            return _research_no_search(client, prompt)
        except Exception as inner:  # noqa: BLE001
            log.exception("fallback research also failed")
            return CompanyResearch(
                one_liner=f"Research failed: {inner.__class__.__name__}: {inner}"
            )
    except Exception as exc:  # noqa: BLE001
        log.exception("research call hit an unexpected error")
        return CompanyResearch(
            one_liner=f"Research failed: {exc.__class__.__name__}: {exc}"
        )

    parsed = _parse_brief(final, fallback_one_liner="No brief returned.")
    if parsed is not None:
        return parsed

    # The with-search call returned no submit_research_brief — try fallback.
    log.warning("with-search call did not produce a brief; running fallback.")
    try:
        return _research_no_search(client, prompt)
    except Exception as exc:  # noqa: BLE001
        log.exception("fallback research failed after empty primary response")
        return CompanyResearch(
            one_liner=f"Research failed: {exc.__class__.__name__}: {exc}"
        )
