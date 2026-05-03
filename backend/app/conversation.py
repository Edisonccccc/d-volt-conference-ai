"""Sales conversation summarization. Transcript -> Claude -> structured brief.

Streams to stderr like the other AI calls. Pulls in linked card context
(extracted contact + research) when available so the summary is sharper.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from .llm import make_client, stream_to_terminal
from .models import CardRecord, ConversationSummary


log = logging.getLogger("conference-ai.conversation")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

DEFAULT_COMPANY_CONTEXT = (
    "d-volt sells voltage regulation and power conditioning hardware to "
    "industrial and commercial customers."
)


SUBMIT_TOOL = {
    "name": "submit_conversation_summary",
    "description": (
        "Deliver the structured summary of the sales conversation. Call "
        "exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "One short paragraph (3-5 sentences) summarizing the "
                    "conversation."
                ),
            },
            "key_topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Main topics discussed, one per bullet.",
            },
            "customer_concerns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Pain points, objections, or concerns the CUSTOMER raised."
                ),
            },
            "commitments": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Concrete commitments either side made (e.g. 'Sales will "
                    "send a quote by Friday', 'Customer will share their "
                    "spec sheet')."
                ),
            },
            "next_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Recommended next steps for the salesperson to advance "
                    "the deal. Be specific and actionable."
                ),
            },
            "follow_up_email": {
                "type": "string",
                "description": (
                    "A polite, concrete draft follow-up email the "
                    "salesperson can send. Plain text, ~150-250 words. "
                    "Reference specific points from the conversation."
                ),
            },
        },
        "required": ["summary"],
    },
}


SYSTEM = (
    "You are a sales coach. You read transcripts of sales conversations and "
    "produce sharp, actionable summaries. You favor concrete details over "
    "filler, never invent commitments that weren't actually made, and make "
    "next steps specific (with names and timelines when possible)."
)


def _card_context_block(card: Optional[CardRecord]) -> str:
    if card is None:
        return "(no linked customer card)"
    lines = []
    if card.extracted:
        if card.extracted.name:
            lines.append(f"Contact: {card.extracted.name}")
        if card.extracted.title:
            lines.append(f"Title: {card.extracted.title}")
        if card.extracted.company:
            lines.append(f"Company: {card.extracted.company}")
    if card.research:
        if card.research.company_category:
            lines.append(f"Category: {card.research.company_category}")
        if card.research.one_liner:
            lines.append(f"Company: {card.research.one_liner}")
        if card.research.industry:
            lines.append(f"Industry: {card.research.industry}")
        if card.research.pain_points:
            lines.append(
                "Likely pain points (pre-call):\n  - "
                + "\n  - ".join(card.research.pain_points)
            )
    return "\n".join(lines) if lines else "(card linked but empty)"


def _build_prompt(transcript: str, card: Optional[CardRecord]) -> str:
    company_context = os.getenv("COMPANY_CONTEXT", DEFAULT_COMPANY_CONTEXT)
    return f"""\
Your seller is at d-volt. Their context:

{company_context}

Customer context (from earlier card scan, may be empty):

{_card_context_block(card)}

Transcript of the sales conversation:

\"\"\"
{transcript.strip()}
\"\"\"

Produce a structured summary by calling `submit_conversation_summary`.

Rules:
- Summarize what actually happened, not what could happen.
- For `customer_concerns`, only include things the customer actually said
  or implied — not the seller.
- For `commitments`, distinguish "Sales will…" vs "Customer will…" in each
  bullet so it's clear who's on the hook.
- For `next_steps`, be specific and actionable: include who, what, by when.
- The follow-up email should reference at least two specific points from
  the conversation so it feels personal.

Customer-context rules (apply when the customer context block above is not
empty):
- The follow-up email MUST address the contact by name (e.g. "Hi Daniel,")
  and reference their company by name when those values are present in the
  customer context. Do not use generic placeholders like "Hi there" if a
  name is available.
- Tailor the email to the company category. A pitch to a Utility reads
  differently from one to an EPC, a Distributor, an End User, or a Vendor.
  Mention reliability/grid impact for utilities; deployment timelines for
  EPCs; margins/inventory for distributors; downtime/cost-per-hour for end
  users.
- If the customer said anything that matches a pre-call pain point, call
  that connection out explicitly in either `next_steps` or the email body
  — it shows the seller did their homework and earns trust.
"""


def summarize_conversation(
    transcript: str, card: Optional[CardRecord] = None
) -> ConversationSummary:
    """Summarize a sales conversation. Returns a ConversationSummary.

    Falls back to a one-line error message in the summary if Claude fails.
    """
    if not transcript or not transcript.strip():
        return ConversationSummary(
            summary="No transcript was captured for this conversation."
        )

    client = make_client(timeout=120.0)
    prompt = _build_prompt(transcript, card)

    try:
        final = stream_to_terminal(
            client,
            label="summary",
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=[SUBMIT_TOOL],
            tool_choice={"type": "tool", "name": "submit_conversation_summary"},
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("summarize_conversation failed")
        return ConversationSummary(
            summary=f"Summarization failed: {exc.__class__.__name__}: {exc}"
        )

    for block in final.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "submit_conversation_summary"
        ):
            try:
                return ConversationSummary.model_validate(block.input)
            except Exception:  # noqa: BLE001
                log.exception("summary input failed schema validation")
                return ConversationSummary(
                    summary="Summary returned but failed schema validation.",
                    key_topics=[json.dumps(block.input)[:500]],
                )

    text = "\n".join(
        getattr(b, "text", "") for b in final.content if getattr(b, "type", None) == "text"
    ).strip()
    return ConversationSummary(
        summary=text.splitlines()[0] if text else "No summary returned."
    )
