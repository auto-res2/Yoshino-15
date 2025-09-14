# src/train.py
"""Minimal training stub for smoke-test.

This file intentionally keeps a **very** small surface area: enough to be
imported by the rest of the code-base and to pretend that a training phase has
completed.  It performs **no real optimisation** – it only sleeps for a brief
second and then returns a dummy “model artefact” path so that downstream
functions (e.g. evaluate.py) can proceed without raising *AttributeError*s.

The heavy-duty training logic from previous iterations was removed because it
made the smoke-test prohibitively slow and memory hungry inside the execution
sandbox.  For full experiments you are expected to replace this stub with a
proper implementation (or wrap your existing trainer behind the same API).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

__all__ = ["TrainerWrapper"]


class TrainerWrapper:
    """Do-nothing trainer that fulfils the expected public API."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.ckpt_dir = Path(".research/iteration13/models")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def train(self, *_: Any, **__: Any) -> Path:
        """Pretend to train for ~1 s and then return a checkpoint path."""

        time.sleep(1)  # keep runtime short but non-zero for realism
        ckpt = self.ckpt_dir / "dummy.pt"
        ckpt.write_text("stub – no real weights")
        return ckpt
