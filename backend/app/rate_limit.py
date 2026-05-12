"""In-process token bucket for Anthropic-style minute-window rate limits.

Anthropic's ITPM (input tokens per minute) and OTPM (output tokens per minute)
are enforced as sliding 60-second windows. The SDK retries on 429 with
backoff, but if we burst hard enough the SDK gives up and surfaces an
exception that ends up as an 'error' row in attendee_scoring. This module
provides proactive throttling: acquire a reservation *before* making a
call, then `record()` the actual usage after the call so the bucket
is accurate.

Design:
- `TokenBucket.acquire(est_input, est_output)` provisionally books the
  estimate against both ITPM and OTPM windows. It blocks (sleeps 1s and
  rechecks) until both budgets allow the call. Returns a mutable
  reservation list.
- `TokenBucket.record(reservation, actual_input, actual_output)` rewrites
  the reservation with actuals, so the bucket reflects reality.
- A single module-level singleton is exposed via `attendee_bucket()`.
  Configured via env vars `ANTHROPIC_ITPM_LIMIT` / `ANTHROPIC_OTPM_LIMIT`.

The bucket is per-process: if you run multiple uvicorn workers, each has
its own counter, so set limits per worker (e.g. with 2 workers, configure
each to use 50% of the real tier limit). For single-worker Render
deployments — the default — it's accurate.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Deque, List, Optional


log = logging.getLogger("conference-ai.rate-limit")


class TokenBucket:
    """Sliding-window dual budget (input + output tokens per minute)."""

    def __init__(self, input_tpm: int, output_tpm: int, *, name: str = "anthropic"):
        self.input_cap = max(1, int(input_tpm))
        self.output_cap = max(1, int(output_tpm))
        self.name = name
        # window items are 3-element lists: [ts, input_tokens, output_tokens]
        # Lists (not tuples) so `record()` can mutate them in place.
        self._window: Deque[List[float]] = deque()
        self._lock = threading.Lock()

    def _trim_locked(self) -> None:
        cutoff = time.monotonic() - 60.0
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _used_locked(self):
        self._trim_locked()
        in_used = 0
        out_used = 0
        for entry in self._window:
            in_used += entry[1]
            out_used += entry[2]
        return in_used, out_used

    def acquire(
        self, est_input: int, est_output: int, *, label: str = "",
    ) -> List[float]:
        """Block until both ITPM and OTPM windows have room for the call.

        Books the *estimate* immediately so concurrent callers see the
        reservation. Caller MUST eventually call `record()` (typically
        inside a finally) so the bucket trues up with the actuals.
        """
        est_input  = max(0, int(est_input))
        est_output = max(0, int(est_output))
        attempt = 0
        while True:
            attempt += 1
            now = time.monotonic()
            with self._lock:
                in_used, out_used = self._used_locked()
                if (in_used + est_input <= self.input_cap and
                    out_used + est_output <= self.output_cap):
                    reservation: List[float] = [now, est_input, est_output]
                    self._window.append(reservation)
                    return reservation
                # Compute how long until the oldest entry rolls out of the
                # 60s window so we sleep just long enough — but cap at 5s
                # so we re-check often as other concurrent calls finish.
                if self._window:
                    wait = 60.0 - (now - self._window[0][0])
                else:
                    wait = 1.0
                wait = max(0.5, min(wait, 5.0))
            if attempt <= 3 or attempt % 10 == 0:
                log.info(
                    "rate-limit[%s]: throttling %s — in=%d/%d out=%d/%d, sleep %.1fs",
                    self.name, label or "call",
                    in_used, self.input_cap, out_used, self.output_cap, wait,
                )
            time.sleep(wait)

    def record(
        self, reservation: List[float], actual_input: int, actual_output: int,
    ) -> None:
        """Replace the reservation's estimate with actuals.

        Safe to call even if the reservation has already rolled out of the
        window — we mutate the list in place; the trim pass will drop it
        on a subsequent call.
        """
        actual_input  = max(0, int(actual_input))
        actual_output = max(0, int(actual_output))
        with self._lock:
            reservation[1] = actual_input
            reservation[2] = actual_output


# ---- Module-level singleton --------------------------------------------------

_attendee_bucket: Optional[TokenBucket] = None
_attendee_bucket_lock = threading.Lock()


def attendee_bucket() -> TokenBucket:
    """Return the shared bucket used by attendee_scoring. Lazy-init so env
    vars can override limits without requiring import order."""
    global _attendee_bucket
    if _attendee_bucket is None:
        with _attendee_bucket_lock:
            if _attendee_bucket is None:
                # Defaults match Anthropic's Tier-1 limits for claude-sonnet-4-6
                # (30K input / 8K output tokens per minute) so out-of-the-box
                # the bucket throttles correctly. Bump these env vars to match
                # your actual tier (Tier 2 = 80K/16K, Tier 4 = 400K/80K, etc.).
                itpm = int(os.getenv("ANTHROPIC_ITPM_LIMIT", "30000"))
                otpm = int(os.getenv("ANTHROPIC_OTPM_LIMIT",  "8000"))
                # Apply a safety margin (default 80%) so we don't push right
                # up to the API edge — the SDK has its own retries but a
                # margin keeps the bucket from racing other in-flight calls.
                margin = float(os.getenv("ANTHROPIC_RATE_LIMIT_MARGIN", "0.85"))
                margin = max(0.5, min(margin, 1.0))
                itpm_eff = int(itpm * margin)
                otpm_eff = int(otpm * margin)
                _attendee_bucket = TokenBucket(itpm_eff, otpm_eff, name="attendee")
                log.info(
                    "rate-limit[attendee]: bucket initialized "
                    "(ITPM=%d, OTPM=%d; raw tier %d/%d, margin %.2f)",
                    itpm_eff, otpm_eff, itpm, otpm, margin,
                )
    return _attendee_bucket
