"""Business-card photo -> structured fields, via Claude vision + forced tool use.

Streams tokens to stderr (Claude-Code style) so you can watch the model work
in your uvicorn terminal.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path

from .costing import anthropic_cost
from .llm import make_client, stream_to_terminal
from .models import ExtractedCard


log = logging.getLogger("conference-ai.extraction")

# Card extraction is essentially OCR + light field labeling — a tiny task
# Haiku handles ~as well as Sonnet at roughly 1/4 the cost. Override via
# ANTHROPIC_EXTRACTION_MODEL if you ever see quality issues.
MODEL = os.getenv(
    "ANTHROPIC_EXTRACTION_MODEL",
    os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
)


# JSON Schema for the tool. Forcing tool use is the most reliable way
# to get strictly structured output from Claude.
EXTRACT_TOOL = {
    "name": "record_card_fields",
    "description": (
        "Record the contact fields read from a business card photo. "
        "Leave fields blank if they are not visible. Do not invent values."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's full name"},
            "title": {"type": "string", "description": "Job title"},
            "company": {"type": "string", "description": "Company name"},
            "emails": {"type": "array", "items": {"type": "string"}},
            "phones": {"type": "array", "items": {"type": "string"}},
            "website": {"type": "string"},
            "address": {"type": "string"},
            "linkedin": {"type": "string"},
            "notes": {
                "type": "string",
                "description": "Anything else legible on the card not captured above",
            },
        },
        "required": [],
    },
}


def _image_block(photo_path: str) -> dict:
    p = Path(photo_path)
    media_type, _ = mimetypes.guess_type(p.name)
    if not media_type or not media_type.startswith("image/"):
        media_type = "image/jpeg"
    data = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


def extract_card(photo_path: str):
    """Extract contact fields from a business card photo.

    Returns ``(ExtractedCard, cost_usd)``. Missing fields are left as None / [].
    Raises whatever the SDK raises on a hard failure so the pipeline can
    record it.
    """
    client = make_client(timeout=90.0)

    system = (
        "You are a careful information extractor. You are given a photo of a "
        "business card. Read every legible field and call the "
        "`record_card_fields` tool exactly once with the values you read. "
        "Never guess. If a field is unreadable or absent, omit it."
    )

    final = stream_to_terminal(
        client,
        label="extract",
        model=MODEL,
        max_tokens=1024,
        system=system,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "record_card_fields"},
        messages=[
            {
                "role": "user",
                "content": [
                    _image_block(photo_path),
                    {
                        "type": "text",
                        "text": (
                            "Extract every readable field from this business card. "
                            "Return them via the tool."
                        ),
                    },
                ],
            }
        ],
    )

    cost = anthropic_cost(getattr(final, "usage", None), MODEL)

    for block in final.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_card_fields":
            return ExtractedCard.model_validate(block.input), cost

    log.warning("extract: no tool_use block found in final message; returning empty.")
    return ExtractedCard(notes="Model did not return structured fields."), cost
