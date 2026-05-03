"""Shared Anthropic client + streaming pretty-printer.

The printer mimics Claude Code's terminal output: as the model streams text,
calls tools, or runs server tools (like web_search), each event is labeled
and printed live so you can see what's happening inside long-running calls.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

from anthropic import Anthropic


log = logging.getLogger("conference-ai.llm")


def make_client(
    *, max_retries: int = 1, timeout: float = 120.0
) -> Anthropic:
    """Build an Anthropic client tuned for fail-fast debugging.

    The default SDK retries up to 2 times with backoff, which can hide a
    400-class error for 90+ seconds. We keep one retry for transient
    network blips but no more.
    """
    return Anthropic(max_retries=max_retries, timeout=timeout)


class _Printer:
    """Tiny state machine that turns streaming events into readable terminal output."""

    def __init__(self, label: str, max_text_chars: int = 4000) -> None:
        self.label = label
        self.max_text_chars = max_text_chars
        self._line_dirty = False
        self._chars_in_block = 0
        self._truncated = False

    # ---- low-level helpers ------------------------------------------------
    def _newline(self) -> None:
        if self._line_dirty:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._line_dirty = False

    def _line(self, msg: str) -> None:
        self._newline()
        sys.stderr.write(f"[{self.label}] {msg}\n")
        sys.stderr.flush()

    def _start_block(self, kind: str) -> None:
        self._newline()
        sys.stderr.write(f"[{self.label}] {kind}: ")
        sys.stderr.flush()
        self._line_dirty = True
        self._chars_in_block = 0
        self._truncated = False

    def _write_streaming(self, text: str) -> None:
        if not text:
            return
        if self._truncated:
            return
        remaining = self.max_text_chars - self._chars_in_block
        if remaining <= 0:
            sys.stderr.write(" […truncated]")
            sys.stderr.flush()
            self._truncated = True
            return
        if len(text) > remaining:
            text = text[:remaining]
            self._truncated = True
        sys.stderr.write(text)
        sys.stderr.flush()
        self._chars_in_block += len(text)
        self._line_dirty = True

    # ---- event entrypoint -------------------------------------------------
    def handle(self, event: Any) -> None:
        et = getattr(event, "type", None)

        if et == "message_start":
            usage = getattr(getattr(event, "message", None), "usage", None)
            in_tok = getattr(usage, "input_tokens", None) if usage else None
            self._line(
                f"=== Claude streaming begin"
                + (f" (input_tokens={in_tok})" if in_tok else "")
                + " ==="
            )

        elif et == "content_block_start":
            block = event.content_block
            bt = getattr(block, "type", None)
            if bt == "text":
                self._start_block("thinking")
            elif bt == "tool_use":
                self._start_block(f"calling tool {block.name!r} with input")
            elif bt == "server_tool_use":
                self._start_block(f"server_tool {block.name!r} input")
            elif bt == "web_search_tool_result":
                # Server tool results arrive in a single block. The printer
                # prints a one-line summary; full results stay inside the
                # final Message for the parser.
                content = getattr(block, "content", None)
                count: Optional[int] = None
                if isinstance(content, list):
                    count = len(content)
                self._line(
                    "web_search results received"
                    + (f" ({count} hits)" if count is not None else "")
                )
            else:
                self._start_block(f"block {bt!r}")

        elif et == "content_block_delta":
            delta = event.delta
            dt = getattr(delta, "type", None)
            if dt == "text_delta":
                self._write_streaming(getattr(delta, "text", ""))
            elif dt == "input_json_delta":
                self._write_streaming(getattr(delta, "partial_json", ""))

        elif et == "message_stop":
            self._line("=== done ===")

        elif et == "error":
            err = getattr(event, "error", None)
            self._line(f"!! stream error: {err}")


def stream_to_terminal(
    client: Anthropic, *, label: str, **kwargs: Any
):
    """Run `client.messages.stream(**kwargs)` while printing live events.

    Returns the final ``Message`` (with all blocks aggregated) so the caller
    can parse tool_use blocks normally.
    """
    printer = _Printer(label)
    log.info("[%s] starting stream model=%s max_tokens=%s tools=%s",
             label, kwargs.get("model"), kwargs.get("max_tokens"),
             [t.get("name") for t in (kwargs.get("tools") or [])])

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            printer.handle(event)
        final = stream.get_final_message()

    log.info("[%s] finished. stop_reason=%s usage=%s",
             label, getattr(final, "stop_reason", None),
             getattr(final, "usage", None))
    return final
