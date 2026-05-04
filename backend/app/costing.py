"""Cost computation for Anthropic + OpenAI calls.

All prices are USD per the listed unit. Update PRICES_USD when vendor
rates change. Functions accept the SDK's usage object (or any object that
exposes the usual `input_tokens` / `output_tokens` / etc. attributes) and
return cost in dollars as a float.

The numbers below are the *list* prices a small team would pay before
volume discounts. They're a reasonable upper bound; actual cost will
match or be slightly lower.
"""

from __future__ import annotations

import logging
from typing import Any, Optional


log = logging.getLogger("conference-ai.costing")


# Per-million-token rates for the Anthropic models we use, plus the rate
# for each web_search server-tool call and per-minute Whisper.
# Source: anthropic.com/pricing, openai.com/pricing as of May 2026.
PRICES_USD = {
    # Claude Sonnet (default)
    "sonnet": {
        "input":      3.00,
        "output":    15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    # Claude Opus
    "opus": {
        "input":     15.00,
        "output":    75.00,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    # Claude Haiku
    "haiku": {
        "input":      0.80,
        "output":     4.00,
        "cache_read": 0.08,
        "cache_write": 1.00,
    },
    # Web search: $10 per 1,000 searches.
    "web_search_per_call": 0.01,
    # OpenAI Whisper: $0.006/min.
    "whisper_per_minute": 0.006,
}


def _tier(model: str) -> dict:
    m = (model or "").lower()
    if "opus" in m:    return PRICES_USD["opus"]
    if "haiku" in m:   return PRICES_USD["haiku"]
    return PRICES_USD["sonnet"]


def anthropic_cost(
    usage: Any,
    model: str,
    *,
    web_search_calls: int = 0,
) -> float:
    """Cost in USD for one Anthropic call.

    `usage` is whatever .usage attribute the SDK returns (it has
    input_tokens, output_tokens, and optionally
    cache_read_input_tokens / cache_creation_input_tokens).
    """
    if usage is None:
        return round(web_search_calls * PRICES_USD["web_search_per_call"], 6)
    rate = _tier(model)
    in_tok   = getattr(usage, "input_tokens", 0) or 0
    out_tok  = getattr(usage, "output_tokens", 0) or 0
    cache_r  = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_w  = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cost = (
        in_tok   * rate["input"]      / 1_000_000
        + out_tok * rate["output"]     / 1_000_000
        + cache_r * rate["cache_read"] / 1_000_000
        + cache_w * rate["cache_write"]/ 1_000_000
        + web_search_calls * PRICES_USD["web_search_per_call"]
    )
    return round(cost, 6)


def whisper_cost(duration_seconds: Optional[float]) -> float:
    if not duration_seconds or duration_seconds < 0:
        return 0.0
    return round((duration_seconds / 60.0) * PRICES_USD["whisper_per_minute"], 6)


def count_web_search_calls(message: Any) -> int:
    """Count server_tool_use blocks named 'web_search' in a Message response."""
    if message is None:
        return 0
    n = 0
    for block in getattr(message, "content", []) or []:
        bt = getattr(block, "type", None)
        if bt == "server_tool_use" and getattr(block, "name", None) == "web_search":
            n += 1
    return n
