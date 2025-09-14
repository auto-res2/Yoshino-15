# src/preprocess.py
"""No-op data pre-processing stub.

The real pipeline tokenises and sanitises large datasets.  For the smoke-test
we only need the function signatures so that the rest of the codebase can
import them.  The *process()* function simply echoes the input.
"""
from __future__ import annotations

from typing import List

__all__ = ["process"]


def process(raw: List[str]):
    """Return the *raw* list unchanged (placeholder implementation)."""

    return raw
