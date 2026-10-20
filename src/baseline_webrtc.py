"""
Reserved Baseline Wrapper for Future Phases.

Per CDX-02 controller orders, external baselines (such as WebRTC AEC) are reserved
until S1 static-room LMS vs NLMS verification passes CDX review.
"""

from __future__ import annotations


def info() -> str:
    return "Reserved module for post-S1 baseline comparisons."
